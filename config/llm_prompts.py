"""
Prompt templates for LLM-based training data generation.

Each task type has:
- SYSTEM_PROMPT: Role and context for the model
- USER_TEMPLATE: Template with {document_text} placeholder
- INSTRUCTION: The instruction that will be used in the training data
"""

from enum import Enum
from dataclasses import dataclass


class TaskType(str, Enum):
    """Types of training examples to generate."""

    SUMMARIZATION = "summarization"
    RESEARCH_QA = "research_qa"
    OUTCOME_ANALYSIS = "outcome_analysis"
    INFO_EXTRACTION = "info_extraction"


@dataclass
class PromptTemplate:
    """Template for a specific task type."""

    task_type: TaskType
    system_prompt: str
    user_template: str
    instruction: str


SUMMARIZATION_PROMPT = PromptTemplate(
    task_type=TaskType.SUMMARIZATION,
    system_prompt="""You are an expert Indian legal analyst. Your task is to create a concise, accurate summary of legal judgments.

Focus on:
- Key legal issues
- Arguments from both parties
- Court's reasoning and legal principles applied
- Final verdict and orders

Write in clear, professional legal language. Be objective and factual.""",
    user_template="""Summarize the following Indian court judgment in 200-400 words:

{document_text}

Provide a structured summary covering: (1) Case background, (2) Key legal issues, (3) Court's analysis, (4) Verdict.""",
    instruction="Summarize this Indian court judgment, covering the case background, key legal issues, court's analysis, and verdict.",
)


RESEARCH_QA_PROMPT = PromptTemplate(
    task_type=TaskType.RESEARCH_QA,
    system_prompt="""You are an Indian legal research assistant. Your task is to answer legal questions based on court judgments.

Guidelines:
- Cite specific sections, acts, and precedents mentioned
- Explain legal principles in accessible language
- Be precise about the court's interpretation
- Distinguish between facts and legal reasoning""",
    user_template="""Based on the following judgment, generate a relevant legal research question and provide a detailed answer:

{document_text}

Format:
QUESTION: [A substantive legal question that this judgment addresses]
ANSWER: [Detailed answer with citations to acts, sections, and legal principles from the judgment]""",
    instruction="Generate a legal research question from this judgment and provide a detailed, well-cited answer.",
)


OUTCOME_ANALYSIS_PROMPT = PromptTemplate(
    task_type=TaskType.OUTCOME_ANALYSIS,
    system_prompt="""You are an expert in Indian legal outcome analysis. Your task is to analyze why cases reached particular outcomes.

Focus on:
- Decisive legal factors
- How evidence was evaluated
- Application of specific legal provisions
- Precedents that influenced the decision""",
    user_template="""Analyze the outcome of the following judgment:

{document_text}

Explain:
1. What was the final outcome (allowed/dismissed/modified)?
2. What were the key factors that determined this outcome?
3. Which legal provisions or precedents were decisive?
4. What could have changed the outcome?""",
    instruction="Analyze the outcome of this judgment, explaining the decisive factors, applicable legal provisions, and what could have changed the result.",
)


INFO_EXTRACTION_PROMPT = PromptTemplate(
    task_type=TaskType.INFO_EXTRACTION,
    system_prompt="""You are a legal information extraction specialist. Your task is to extract structured information from Indian court judgments.

Extract accurately without adding information not present in the judgment. Use "Not mentioned" for unavailable information.""",
    user_template="""Extract key information from the following judgment:

{document_text}

Provide structured extraction:
- Case Type: [Criminal/Civil/Constitutional/Tax/etc.]
- Petitioner: [Name and description]
- Respondent: [Name and description]
- Court: [Court name and bench]
- Key Statutes: [List of acts and sections cited]
- Key Precedents: [Cases cited with relevance]
- Relief Sought: [What the petitioner asked for]
- Relief Granted: [What the court ordered]
- Key Legal Principles: [Principles established or applied]""",
    instruction="Extract structured information from this judgment including parties, statutes cited, precedents, relief sought, and key legal principles.",
)


# Registry of all prompts
PROMPTS: dict[TaskType, PromptTemplate] = {
    TaskType.SUMMARIZATION: SUMMARIZATION_PROMPT,
    TaskType.RESEARCH_QA: RESEARCH_QA_PROMPT,
    TaskType.OUTCOME_ANALYSIS: OUTCOME_ANALYSIS_PROMPT,
    TaskType.INFO_EXTRACTION: INFO_EXTRACTION_PROMPT,
}


def get_prompt(task_type: TaskType) -> PromptTemplate:
    """Get prompt template for a task type."""
    return PROMPTS[task_type]


def get_all_task_types() -> list[TaskType]:
    """Get all available task types."""
    return list(TaskType)
