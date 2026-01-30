#!/usr/bin/env python3
"""
Nyay AI Model Evaluation Script

Evaluates the fine-tuned model on diverse legal test cases and generates
comprehensive metrics including accuracy, hallucination rate, and quality scores.

Usage:
    python scripts/evaluate_model.py --checkpoint models/nyay-ai-checkpoints-v4/0003000_adapters.safetensors
    python scripts/evaluate_model.py --checkpoint models/nyay-ai-checkpoints-v4/0003000_adapters.safetensors --interactive
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import re

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mlx_lm import load, generate


class LegalModelEvaluator:
    """Evaluates fine-tuned legal AI model on test cases"""

    def __init__(self, base_model_path: str, checkpoint_path: str):
        """Initialize evaluator with model paths"""
        self.base_model_path = base_model_path
        self.checkpoint_path = checkpoint_path
        self.model = None
        self.tokenizer = None

        # Evaluation results
        self.results = []
        self.metrics = {}

    def load_model(self):
        """Load base model and fine-tuned checkpoint"""
        print("="*70)
        print("LOADING MODEL")
        print("="*70)
        print(f"Base model: {self.base_model_path}")
        print(f"Checkpoint: {self.checkpoint_path}")
        print()

        print("Loading base model...")
        self.model, self.tokenizer = load(self.base_model_path)
        print("✓ Base model loaded")

        print(f"Loading checkpoint...")
        self.model.load_weights(self.checkpoint_path, strict=False)
        print("✓ Checkpoint loaded")
        print()

    def load_test_cases(self, test_file: str) -> List[Dict]:
        """Load test cases from JSONL file"""
        test_cases = []
        with open(test_file, 'r') as f:
            for line in f:
                test_cases.append(json.loads(line))
        return test_cases

    def generate_response(self, prompt: str, max_tokens: int = 200) -> str:
        """Generate response from model"""
        response = generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            verbose=False
        )
        return response.strip()

    def check_keyword_presence(self, response: str, keywords: List[str]) -> Tuple[int, int]:
        """Check how many expected keywords are present in response"""
        response_lower = response.lower()
        found = 0
        for keyword in keywords:
            if keyword.lower() in response_lower:
                found += 1
        return found, len(keywords)

    def detect_hallucination_markers(self, response: str) -> List[str]:
        """Detect potential hallucination markers in response"""
        markers = []

        # Check for made-up case names (common hallucination)
        if re.search(r'\b[A-Z][a-z]+ v\.? [A-Z][a-z]+\b', response):
            # Very simplified check - would need more sophisticated validation
            pass

        # Check for specific years/dates (might be fabricated)
        years = re.findall(r'\b(19|20)\d{2}\b', response)
        if len(years) > 3:
            markers.append("Multiple specific years mentioned (verify accuracy)")

        # Check for overly specific numbers
        if re.search(r'\b\d{1,3}(,\d{3})+\b', response):
            markers.append("Specific numbers mentioned (verify if accurate)")

        # Check for hedging language (good - shows uncertainty)
        hedging_words = ['generally', 'typically', 'usually', 'may', 'can', 'might']
        hedging_count = sum(1 for word in hedging_words if word in response.lower())
        if hedging_count == 0 and len(response.split()) > 50:
            markers.append("No hedging language (overly confident?)")

        return markers

    def assess_quality(self, response: str, test_case: Dict) -> Dict:
        """Assess response quality across multiple dimensions"""
        quality = {}

        # 1. Keyword coverage
        found, total = self.check_keyword_presence(response, test_case.get('expected_keywords', []))
        quality['keyword_coverage'] = found / total if total > 0 else 0
        quality['keywords_found'] = found
        quality['keywords_total'] = total

        # 2. Response length adequacy
        words = len(response.split())
        quality['word_count'] = words
        quality['length_adequate'] = 30 <= words <= 300  # Reasonable length

        # 3. Coherence check (basic)
        sentences = re.split(r'[.!?]+', response)
        quality['sentence_count'] = len([s for s in sentences if s.strip()])
        quality['coherent'] = quality['sentence_count'] >= 2 and words > 20

        # 4. Legal terminology presence
        legal_terms = ['court', 'act', 'section', 'article', 'petition', 'jurisdiction',
                      'constitutional', 'statute', 'law', 'rights', 'justice']
        legal_term_count = sum(1 for term in legal_terms if term in response.lower())
        quality['legal_terminology'] = legal_term_count >= 2
        quality['legal_term_count'] = legal_term_count

        # 5. Hallucination markers
        quality['hallucination_markers'] = self.detect_hallucination_markers(response)
        quality['has_hallucination_risk'] = len(quality['hallucination_markers']) > 0

        # 6. Overall score (0-100)
        score = 0
        score += quality['keyword_coverage'] * 40  # 40% weight
        score += (20 if quality['length_adequate'] else 0)  # 20% weight
        score += (20 if quality['coherent'] else 0)  # 20% weight
        score += (20 if quality['legal_terminology'] else 0)  # 20% weight
        score -= (len(quality['hallucination_markers']) * 5)  # Penalty for hallucination risk

        quality['overall_score'] = max(0, min(100, score))

        return quality

    def evaluate_test_case(self, test_case: Dict, interactive: bool = False) -> Dict:
        """Evaluate a single test case"""
        print(f"\n{'='*70}")
        print(f"TEST CASE: {test_case['id']}")
        print(f"Task: {test_case['task']}")
        print(f"Difficulty: {test_case.get('difficulty', 'N/A')}")
        print(f"{'='*70}")
        print(f"\nPrompt: {test_case['prompt']}")
        print(f"\nGenerating response...")

        # Generate response
        response = self.generate_response(test_case['prompt'])

        print(f"\nResponse:")
        print("-"*70)
        print(response)
        print("-"*70)

        # Assess quality
        quality = self.assess_quality(response, test_case)

        print(f"\nAutomatic Assessment:")
        print(f"  Keyword Coverage: {quality['keywords_found']}/{quality['keywords_total']} ({quality['keyword_coverage']*100:.1f}%)")
        print(f"  Word Count: {quality['word_count']}")
        print(f"  Legal Terms: {quality['legal_term_count']}")
        print(f"  Coherent: {'✓' if quality['coherent'] else '✗'}")
        print(f"  Hallucination Risk: {'⚠️  Yes' if quality['has_hallucination_risk'] else '✓ Low'}")
        if quality['hallucination_markers']:
            for marker in quality['hallucination_markers']:
                print(f"    - {marker}")
        print(f"  Overall Score: {quality['overall_score']:.1f}/100")

        # Manual review if interactive
        manual_score = None
        manual_feedback = None

        if interactive:
            print(f"\n{'='*70}")
            print("MANUAL REVIEW")
            print(f"{'='*70}")
            print("Rate the response (0-5):")
            print("  5 = Excellent (accurate, complete, well-explained)")
            print("  4 = Good (accurate, mostly complete)")
            print("  3 = Acceptable (correct but incomplete or unclear)")
            print("  2 = Poor (partially incorrect or confusing)")
            print("  1 = Very Poor (mostly incorrect)")
            print("  0 = Completely Wrong")

            try:
                manual_score = int(input("\nYour rating (0-5): ").strip())
                manual_feedback = input("Optional feedback: ").strip()
            except (ValueError, KeyboardInterrupt):
                print("\nSkipping manual review")

        # Compile result
        result = {
            'id': test_case['id'],
            'task': test_case['task'],
            'difficulty': test_case.get('difficulty', 'N/A'),
            'prompt': test_case['prompt'],
            'response': response,
            'expected_keywords': test_case.get('expected_keywords', []),
            'quality': quality,
            'manual_score': manual_score,
            'manual_feedback': manual_feedback,
            'timestamp': datetime.now().isoformat()
        }

        return result

    def calculate_aggregate_metrics(self):
        """Calculate aggregate metrics across all test cases"""
        if not self.results:
            return

        # Overall statistics
        self.metrics['total_tests'] = len(self.results)
        self.metrics['avg_score'] = sum(r['quality']['overall_score'] for r in self.results) / len(self.results)
        self.metrics['avg_keyword_coverage'] = sum(r['quality']['keyword_coverage'] for r in self.results) / len(self.results) * 100
        self.metrics['avg_word_count'] = sum(r['quality']['word_count'] for r in self.results) / len(self.results)

        # Success rates
        self.metrics['coherent_rate'] = sum(1 for r in self.results if r['quality']['coherent']) / len(self.results) * 100
        self.metrics['legal_terminology_rate'] = sum(1 for r in self.results if r['quality']['legal_terminology']) / len(self.results) * 100
        self.metrics['hallucination_rate'] = sum(1 for r in self.results if r['quality']['has_hallucination_risk']) / len(self.results) * 100

        # Score distribution
        score_bins = {
            'excellent': sum(1 for r in self.results if r['quality']['overall_score'] >= 80),
            'good': sum(1 for r in self.results if 60 <= r['quality']['overall_score'] < 80),
            'acceptable': sum(1 for r in self.results if 40 <= r['quality']['overall_score'] < 60),
            'poor': sum(1 for r in self.results if r['quality']['overall_score'] < 40)
        }
        self.metrics['score_distribution'] = score_bins

        # By task type
        task_scores = {}
        for result in self.results:
            task = result['task']
            if task not in task_scores:
                task_scores[task] = []
            task_scores[task].append(result['quality']['overall_score'])

        self.metrics['by_task'] = {
            task: {
                'count': len(scores),
                'avg_score': sum(scores) / len(scores)
            }
            for task, scores in task_scores.items()
        }

        # Manual review stats (if available)
        manual_scores = [r['manual_score'] for r in self.results if r['manual_score'] is not None]
        if manual_scores:
            self.metrics['manual_review'] = {
                'count': len(manual_scores),
                'avg_score': sum(manual_scores) / len(manual_scores),
                'avg_score_out_of_100': sum(manual_scores) / len(manual_scores) * 20  # Convert 0-5 to 0-100
            }

    def print_summary(self):
        """Print evaluation summary"""
        print("\n" + "="*70)
        print("EVALUATION SUMMARY")
        print("="*70)
        print()

        print(f"Total Tests: {self.metrics['total_tests']}")
        print(f"Average Score: {self.metrics['avg_score']:.1f}/100")
        print(f"Average Keyword Coverage: {self.metrics['avg_keyword_coverage']:.1f}%")
        print(f"Average Response Length: {self.metrics['avg_word_count']:.0f} words")
        print()

        print("Quality Rates:")
        print(f"  Coherent Responses: {self.metrics['coherent_rate']:.1f}%")
        print(f"  Legal Terminology: {self.metrics['legal_terminology_rate']:.1f}%")
        print(f"  Hallucination Risk: {self.metrics['hallucination_rate']:.1f}%")
        print()

        print("Score Distribution:")
        dist = self.metrics['score_distribution']
        print(f"  Excellent (≥80): {dist['excellent']} ({dist['excellent']/self.metrics['total_tests']*100:.1f}%)")
        print(f"  Good (60-79):    {dist['good']} ({dist['good']/self.metrics['total_tests']*100:.1f}%)")
        print(f"  Acceptable (40-59): {dist['acceptable']} ({dist['acceptable']/self.metrics['total_tests']*100:.1f}%)")
        print(f"  Poor (<40):      {dist['poor']} ({dist['poor']/self.metrics['total_tests']*100:.1f}%)")
        print()

        print("Performance by Task:")
        for task, stats in self.metrics['by_task'].items():
            print(f"  {task:30s}: {stats['avg_score']:.1f}/100 ({stats['count']} tests)")
        print()

        if 'manual_review' in self.metrics:
            mr = self.metrics['manual_review']
            print("Manual Review:")
            print(f"  Reviewed: {mr['count']} tests")
            print(f"  Average: {mr['avg_score']:.1f}/5 ({mr['avg_score_out_of_100']:.1f}/100)")
            print()

        # Overall assessment
        avg_score = self.metrics['avg_score']
        print("Overall Assessment:")
        if avg_score >= 75:
            print("  ✅ EXCELLENT - Model is production-ready")
        elif avg_score >= 60:
            print("  ✓ GOOD - Model performs well, minor improvements possible")
        elif avg_score >= 40:
            print("  ⚠️  ACCEPTABLE - Model needs improvement before deployment")
        else:
            print("  ❌ POOR - Model requires significant retraining")
        print()

    def save_results(self, output_file: str):
        """Save detailed results to JSON file"""
        output_data = {
            'checkpoint': self.checkpoint_path,
            'evaluation_time': datetime.now().isoformat(),
            'metrics': self.metrics,
            'results': self.results
        }

        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"✓ Detailed results saved to: {output_file}")

    def run_evaluation(self, test_file: str, interactive: bool = False, limit: int = None):
        """Run complete evaluation"""
        # Load model
        self.load_model()

        # Load test cases
        print("="*70)
        print("LOADING TEST CASES")
        print("="*70)
        test_cases = self.load_test_cases(test_file)
        print(f"Loaded {len(test_cases)} test cases")

        if limit:
            test_cases = test_cases[:limit]
            print(f"Limited to {limit} test cases")
        print()

        # Evaluate each test case
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n{'#'*70}")
            print(f"# TEST {i}/{len(test_cases)}")
            print(f"{'#'*70}")

            result = self.evaluate_test_case(test_case, interactive)
            self.results.append(result)

            if interactive and i < len(test_cases):
                input("\nPress Enter to continue to next test case...")

        # Calculate metrics
        self.calculate_aggregate_metrics()

        # Print summary
        self.print_summary()


def main():
    parser = argparse.ArgumentParser(description='Evaluate Nyay AI model')
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='models/nyay-ai-checkpoints-v4/0003000_adapters.safetensors',
        help='Path to checkpoint file'
    )
    parser.add_argument(
        '--base-model',
        type=str,
        default='models/llama-3.2-3b-instruct-mlx',
        help='Path to base model'
    )
    parser.add_argument(
        '--test-file',
        type=str,
        default='data/evaluation/test_cases.jsonl',
        help='Path to test cases file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output file for results (default: auto-generated)'
    )
    parser.add_argument(
        '--interactive',
        action='store_true',
        help='Enable interactive mode for manual review'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit number of test cases to evaluate'
    )

    args = parser.parse_args()

    # Auto-generate output filename if not provided
    if args.output is None:
        checkpoint_name = Path(args.checkpoint).stem
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.output = f'scripts/evaluation_results/{checkpoint_name}_{timestamp}.json'

    # Ensure output directory exists
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    # Run evaluation
    evaluator = LegalModelEvaluator(args.base_model, args.checkpoint)
    evaluator.run_evaluation(args.test_file, args.interactive, args.limit)

    # Save results
    evaluator.save_results(args.output)

    print("\n" + "="*70)
    print("EVALUATION COMPLETE")
    print("="*70)


if __name__ == '__main__':
    main()
