"""
LLM-based training example generator using Claude API.

Generates instruction-following training examples from legal documents.
"""

import random
from dataclasses import dataclass, field

import anthropic

from config.settings import settings
from config.llm_prompts import TaskType, get_prompt, get_all_task_types
from processors.text_cleaner import clean_text
from utils.logger import get_logger
from utils.rate_limiter import AdaptiveRateLimiter

logger = get_logger(__name__)


@dataclass
class GeneratedExample:
    """A single generated training example."""

    cnr: str
    task_type: TaskType
    instruction: str
    input_text: str
    output_text: str
    input_tokens: int
    output_tokens: int


@dataclass
class GenerationResult:
    """Result from generating examples for a document."""

    cnr: str
    success: bool
    examples: list[GeneratedExample] = field(default_factory=list)
    error_message: str | None = None
    total_input_tokens: int = 0
    total_output_tokens: int = 0


class LLMGenerator:
    """Generate training examples using Claude API."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = settings.LLM_MODEL,
        max_tokens: int = settings.LLM_MAX_TOKENS,
        temperature: float = settings.LLM_TEMPERATURE,
    ):
        """Initialize LLM generator.

        Args:
            api_key: Anthropic API key. Uses settings if not provided.
            model: Model to use (default: claude-3-haiku).
            max_tokens: Maximum output tokens.
            temperature: Sampling temperature.
        """
        self.api_key = api_key or settings.ANTHROPIC_API_KEY
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY not set in environment or settings")

        self.client = anthropic.Anthropic(api_key=self.api_key)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

        # Rate limiting
        self.rate_limiter = AdaptiveRateLimiter(
            min_interval=settings.LLM_MIN_REQUEST_INTERVAL,
            max_interval=settings.LLM_MIN_REQUEST_INTERVAL * 2,
        )

        # Cost tracking
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    def generate_example(
        self,
        document_text: str,
        task_type: TaskType,
    ) -> tuple[str | None, int, int]:
        """Generate a single training example.

        Args:
            document_text: The document content.
            task_type: Type of example to generate.

        Returns:
            Tuple of (generated_output, input_tokens, output_tokens).
            Returns (None, 0, 0) on failure.
        """
        prompt_template = get_prompt(task_type)

        # Clean and truncate document
        cleaned_text = clean_text(
            document_text, max_chars=settings.TRAINING_MAX_INPUT_CHARS
        )
        if not cleaned_text:
            logger.warning("Document text empty after cleaning")
            return None, 0, 0

        user_message = prompt_template.user_template.format(
            document_text=cleaned_text
        )

        try:
            self.rate_limiter.wait()

            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=prompt_template.system_prompt,
                messages=[{"role": "user", "content": user_message}],
            )

            self.rate_limiter.record_success()

            output = response.content[0].text
            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens

            # Update totals
            self.total_input_tokens += input_tokens
            self.total_output_tokens += output_tokens

            return output, input_tokens, output_tokens

        except anthropic.RateLimitError as e:
            logger.warning(f"Rate limited: {e}")
            self.rate_limiter.record_error(is_rate_limit=True)
            self.rate_limiter.backoff(60)
            return None, 0, 0

        except anthropic.APIError as e:
            logger.error(f"API error: {e}")
            self.rate_limiter.record_error(is_rate_limit=False)
            return None, 0, 0

        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return None, 0, 0

    def generate_for_document(
        self,
        cnr: str,
        document_text: str,
        task_types: list[TaskType],
    ) -> GenerationResult:
        """Generate examples for a single document.

        Args:
            cnr: Document CNR.
            document_text: Full document text.
            task_types: List of task types to generate.

        Returns:
            GenerationResult with all examples or error.
        """
        result = GenerationResult(cnr=cnr, success=True)

        for task_type in task_types:
            output, input_tokens, output_tokens = self.generate_example(
                document_text, task_type
            )

            if output:
                prompt_template = get_prompt(task_type)
                cleaned_input = clean_text(
                    document_text, max_chars=settings.TRAINING_MAX_INPUT_CHARS
                )

                example = GeneratedExample(
                    cnr=cnr,
                    task_type=task_type,
                    instruction=prompt_template.instruction,
                    input_text=cleaned_input or "",
                    output_text=output,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                )
                result.examples.append(example)
                result.total_input_tokens += input_tokens
                result.total_output_tokens += output_tokens
            else:
                result.success = False
                result.error_message = f"Failed to generate {task_type.value}"
                logger.warning(f"Failed to generate {task_type.value} for {cnr}")

        return result

    def get_estimated_cost(self) -> float:
        """Calculate estimated cost based on tokens used."""
        input_cost = (
            self.total_input_tokens / 1000
        ) * settings.HAIKU_INPUT_COST_PER_1K
        output_cost = (
            self.total_output_tokens / 1000
        ) * settings.HAIKU_OUTPUT_COST_PER_1K
        return input_cost + output_cost

    def reset_token_counts(self) -> None:
        """Reset token counters."""
        self.total_input_tokens = 0
        self.total_output_tokens = 0


class TaskAssigner:
    """Assigns task types to documents for balanced distribution."""

    def __init__(
        self,
        target_per_type: int = settings.TRAINING_TARGET_PER_TYPE,
        seed: int = 42,
    ):
        """Initialize task assigner.

        Args:
            target_per_type: Target examples per task type.
            seed: Random seed.
        """
        self.target_per_type = target_per_type
        self.task_types = get_all_task_types()
        self.examples_per_doc = 2  # Generate 2 examples per document

        # Track counts per type
        self.counts = {t: 0 for t in self.task_types}

        self.rng = random.Random(seed)

    def assign_tasks(self, cnr: str) -> list[TaskType]:
        """Assign task types for a document.

        Ensures balanced distribution across task types.

        Args:
            cnr: Document CNR (for deterministic assignment).

        Returns:
            List of 2 task types to generate for this document.
        """
        # Get types that haven't reached target
        available = [
            t for t in self.task_types if self.counts[t] < self.target_per_type
        ]

        if len(available) < self.examples_per_doc:
            # All types near target, pick any
            available = self.task_types.copy()

        # Pick 2 different task types
        self.rng.shuffle(available)
        selected = available[: self.examples_per_doc]

        # Update counts
        for t in selected:
            self.counts[t] += 1

        return selected

    def get_distribution(self) -> dict[str, int]:
        """Get current distribution of task types."""
        return {t.value: c for t, c in self.counts.items()}


def create_generator() -> LLMGenerator:
    """Factory function to create an LLM generator."""
    return LLMGenerator()
