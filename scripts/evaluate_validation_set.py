#!/usr/bin/env python3
"""
Automated Evaluation Script - 810 Validation Examples

Evaluates the deployed nyay-ai model on all validation examples to generate
comprehensive performance metrics and reports.

Usage:
    python scripts/evaluate_validation_set.py
    python scripts/evaluate_validation_set.py --limit 100  # Test with subset
    python scripts/evaluate_validation_set.py --model nyay-ai  # Specify model
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import re
from collections import defaultdict, Counter
import statistics
import requests

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class ValidationSetEvaluator:
    """Evaluates deployed model on validation examples"""

    def __init__(self, model_name: str = "nyay-ai", val_file: str = "data/training/val.jsonl"):
        """Initialize evaluator"""
        self.model_name = model_name
        self.val_file = Path(val_file)
        self.results = []
        self.start_time = None
        self.end_time = None

    def load_validation_examples(self, limit: int = None) -> List[Dict]:
        """Load validation examples from JSONL"""
        print("="*70)
        print("LOADING VALIDATION EXAMPLES")
        print("="*70)
        print(f"Source: {self.val_file}")

        if not self.val_file.exists():
            raise FileNotFoundError(f"Validation file not found: {self.val_file}")

        examples = []
        with open(self.val_file, 'r') as f:
            for line in f:
                if line.strip():
                    examples.append(json.loads(line))
                if limit and len(examples) >= limit:
                    break

        print(f"Loaded: {len(examples)} examples")

        # Show distribution
        task_types = Counter(ex.get('metadata', {}).get('task_type', 'unknown') for ex in examples)
        print(f"\nTask Type Distribution:")
        for task_type, count in sorted(task_types.items()):
            print(f"  - {task_type}: {count}")

        print()
        return examples

    def query_ollama_model(self, prompt: str, timeout: int = 120) -> Dict:
        """Query Ollama model via HTTP API and return response"""
        try:
            # Use Ollama HTTP API (handles long prompts better than CLI)
            url = "http://localhost:11434/api/generate"

            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": 500,  # Max tokens to generate
                    "temperature": 0.1,
                }
            }

            response = requests.post(url, json=payload, timeout=timeout)

            if response.status_code == 200:
                data = response.json()
                return {
                    "success": True,
                    "response": data.get("response", "").strip(),
                    "error": None
                }
            else:
                return {
                    "success": False,
                    "response": None,
                    "error": f"HTTP {response.status_code}: {response.text}"
                }
        except requests.Timeout:
            return {
                "success": False,
                "response": None,
                "error": f"Timeout after {timeout}s"
            }
        except requests.ConnectionError:
            return {
                "success": False,
                "response": None,
                "error": "Cannot connect to Ollama. Is it running? (ollama serve)"
            }
        except Exception as e:
            return {
                "success": False,
                "response": None,
                "error": str(e)
            }

    def calculate_keyword_coverage(self, response: str, expected: str) -> float:
        """Calculate keyword coverage between response and expected output"""
        # Extract significant words (>3 chars, not common stopwords)
        stopwords = {'the', 'and', 'for', 'that', 'with', 'this', 'from', 'are', 'was', 'were'}

        def extract_keywords(text: str) -> set:
            words = re.findall(r'\b\w{4,}\b', text.lower())
            return set(w for w in words if w not in stopwords)

        expected_keywords = extract_keywords(expected)
        response_keywords = extract_keywords(response)

        if not expected_keywords:
            return 0.0

        overlap = len(expected_keywords & response_keywords)
        coverage = (overlap / len(expected_keywords)) * 100
        return coverage

    def check_coherence(self, response: str) -> bool:
        """Check if response is coherent"""
        if not response or len(response.strip()) < 10:
            return False

        # Check for common refusal patterns
        refusal_patterns = [
            "i cannot", "i don't know", "i'm not sure",
            "unable to", "cannot provide", "don't have information"
        ]

        response_lower = response.lower()
        if any(pattern in response_lower for pattern in refusal_patterns):
            return False

        # Check for sentence structure
        has_punctuation = any(p in response for p in ['.', '!', '?', ':', ';'])
        if not has_punctuation:
            return False

        return True

    def check_legal_terminology(self, response: str) -> bool:
        """Check if response contains legal terminology"""
        legal_terms = [
            'section', 'act', 'article', 'court', 'provision', 'law',
            'statute', 'judgment', 'petition', 'appeal', 'writ', 'jurisdiction',
            'case', 'plaintiff', 'defendant', 'counsel', 'bench', 'high court',
            'supreme court', 'constitution', 'code', 'procedure', 'criminal',
            'civil', 'penal', 'relief', 'order', 'decree'
        ]

        response_lower = response.lower()
        return any(term in response_lower for term in legal_terms)

    def detect_hallucination_risk(self, response: str) -> str:
        """Detect potential hallucination markers"""
        markers = []

        # Check for overly specific dates/numbers
        years = re.findall(r'\b(19|20)\d{2}\b', response)
        if len(years) > 5:
            markers.append("many_specific_years")

        # Check for made-up looking case citations
        citations = re.findall(r'\(\d{4}\)\s*\d+\s*[A-Z]+', response)
        if len(citations) > 3:
            markers.append("many_citations")

        # Check for repetitive content
        sentences = response.split('.')
        if len(sentences) > 3:
            sentence_set = set(s.strip() for s in sentences if s.strip())
            if len(sentence_set) < len(sentences) * 0.7:
                markers.append("repetitive")

        # Check for contradictions (very basic)
        if "not" in response.lower() and "is" in response.lower():
            # This is overly simplistic, but gives a basic check
            pass

        if len(markers) >= 2:
            return "High"
        elif len(markers) == 1:
            return "Medium"
        else:
            return "Low"

    def check_completeness(self, response: str) -> bool:
        """Check if response appears complete"""
        # Too short
        if len(response) < 50:
            return False

        # Ends abruptly
        if not response.strip()[-1] in ['.', '!', '?']:
            return False

        return True

    def evaluate_response(self, example: Dict, response: str) -> Dict:
        """Evaluate a single response against expected output"""
        expected = example['output']
        metadata = example.get('metadata', {})

        # Calculate metrics
        keyword_coverage = self.calculate_keyword_coverage(response, expected)
        is_coherent = self.check_coherence(response)
        has_legal_terms = self.check_legal_terminology(response)
        hallucination_risk = self.detect_hallucination_risk(response)
        is_complete = self.check_completeness(response)

        # Calculate overall score (0-100)
        score = 0.0

        # Keyword coverage (40%)
        score += keyword_coverage * 0.4

        # Coherence (20%)
        score += 20 if is_coherent else 0

        # Legal terminology (20%)
        score += 20 if has_legal_terms else 0

        # Hallucination check (10%)
        if hallucination_risk == "Low":
            score += 10
        elif hallucination_risk == "Medium":
            score += 5
        # High risk = 0 points

        # Completeness (10%)
        score += 10 if is_complete else 0

        return {
            "score": round(score, 1),
            "keyword_coverage": round(keyword_coverage, 1),
            "coherent": is_coherent,
            "legal_terminology": has_legal_terms,
            "hallucination_risk": hallucination_risk,
            "complete": is_complete,
            "response_length": len(response.split()),
            "expected_length": len(expected.split())
        }

    def run_evaluation(self, limit: int = None):
        """Run evaluation on all validation examples"""
        print("="*70)
        print(f"EVALUATING MODEL: {self.model_name}")
        print("="*70)
        print()

        # Load examples
        examples = self.load_validation_examples(limit)
        total = len(examples)

        self.start_time = datetime.now()
        print(f"Started: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Estimated time: ~{total * 10 / 60:.1f} minutes")
        print()

        # Process each example
        for i, example in enumerate(examples, 1):
            print(f"[{i}/{total}] Processing example {i}...")

            # Build prompt
            instruction = example['instruction']
            input_text = example['input']
            prompt = f"{instruction}\n\n{input_text}" if input_text else instruction

            # Query model
            result = self.query_ollama_model(prompt)

            if not result['success']:
                print(f"  ❌ Error: {result['error']}")
                self.results.append({
                    "index": i,
                    "cnr": example.get('metadata', {}).get('cnr', 'unknown'),
                    "task_type": example.get('metadata', {}).get('task_type', 'unknown'),
                    "court": example.get('metadata', {}).get('court', 'unknown'),
                    "success": False,
                    "error": result['error'],
                    "instruction": instruction[:100] + "...",
                    "score": 0,
                })
                continue

            response = result['response']

            # Evaluate response
            evaluation = self.evaluate_response(example, response)

            # Store result
            self.results.append({
                "index": i,
                "cnr": example.get('metadata', {}).get('cnr', 'unknown'),
                "task_type": example.get('metadata', {}).get('task_type', 'unknown'),
                "court": example.get('metadata', {}).get('court', 'unknown'),
                "success": True,
                "instruction": instruction[:100] + "...",
                "expected_output": example['output'][:200] + "...",
                "model_output": response[:200] + "..." if len(response) > 200 else response,
                "full_output": response,
                **evaluation
            })

            # Progress update
            print(f"  Score: {evaluation['score']:.1f}/100 | "
                  f"Coverage: {evaluation['keyword_coverage']:.1f}% | "
                  f"Risk: {evaluation['hallucination_risk']}")

            # Show periodic summary
            if i % 50 == 0:
                self._print_interim_summary(i, total)

        self.end_time = datetime.now()
        print()
        print(f"Completed: {self.end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Duration: {(self.end_time - self.start_time).total_seconds() / 60:.1f} minutes")
        print()

    def _print_interim_summary(self, current: int, total: int):
        """Print interim summary during evaluation"""
        successful = [r for r in self.results if r.get('success', False)]
        if not successful:
            return

        scores = [r['score'] for r in successful]
        avg_score = statistics.mean(scores)

        print()
        print(f"  === Interim Summary ({current}/{total}) ===")
        print(f"  Average Score: {avg_score:.1f}/100")
        print(f"  Success Rate: {len(successful)}/{current} ({len(successful)/current*100:.1f}%)")
        print()

    def calculate_overall_metrics(self) -> Dict:
        """Calculate overall metrics from results"""
        successful = [r for r in self.results if r.get('success', False)]

        if not successful:
            return {"error": "No successful evaluations"}

        scores = [r['score'] for r in successful]

        metrics = {
            "total_examples": len(self.results),
            "successful": len(successful),
            "failed": len(self.results) - len(successful),
            "success_rate": len(successful) / len(self.results) * 100,

            # Score statistics
            "avg_score": statistics.mean(scores),
            "median_score": statistics.median(scores),
            "std_dev": statistics.stdev(scores) if len(scores) > 1 else 0,
            "min_score": min(scores),
            "max_score": max(scores),
            "percentile_25": statistics.quantiles(scores, n=4)[0] if len(scores) >= 4 else None,
            "percentile_75": statistics.quantiles(scores, n=4)[2] if len(scores) >= 4 else None,
            "percentile_90": statistics.quantiles(scores, n=10)[8] if len(scores) >= 10 else None,

            # Pass rates
            "pass_rate_60": sum(1 for s in scores if s >= 60) / len(scores) * 100,
            "pass_rate_70": sum(1 for s in scores if s >= 70) / len(scores) * 100,
            "pass_rate_80": sum(1 for s in scores if s >= 80) / len(scores) * 100,

            # Quality metrics
            "coherence_rate": sum(1 for r in successful if r['coherent']) / len(successful) * 100,
            "legal_terminology_rate": sum(1 for r in successful if r['legal_terminology']) / len(successful) * 100,
            "avg_keyword_coverage": statistics.mean(r['keyword_coverage'] for r in successful),
            "complete_rate": sum(1 for r in successful if r['complete']) / len(successful) * 100,

            # Hallucination
            "hallucination_low": sum(1 for r in successful if r['hallucination_risk'] == 'Low') / len(successful) * 100,
            "hallucination_medium": sum(1 for r in successful if r['hallucination_risk'] == 'Medium') / len(successful) * 100,
            "hallucination_high": sum(1 for r in successful if r['hallucination_risk'] == 'High') / len(successful) * 100,

            # Response length
            "avg_response_length": statistics.mean(r['response_length'] for r in successful),
            "median_response_length": statistics.median(r['response_length'] for r in successful),
        }

        return metrics

    def calculate_task_type_metrics(self) -> Dict:
        """Calculate metrics by task type"""
        successful = [r for r in self.results if r.get('success', False)]

        by_task = defaultdict(list)
        for result in successful:
            task_type = result['task_type']
            by_task[task_type].append(result)

        task_metrics = {}
        for task_type, results in by_task.items():
            scores = [r['score'] for r in results]

            task_metrics[task_type] = {
                "count": len(results),
                "avg_score": statistics.mean(scores),
                "median_score": statistics.median(scores),
                "min_score": min(scores),
                "max_score": max(scores),
                "pass_rate": sum(1 for s in scores if s >= 60) / len(scores) * 100,
                "coherence_rate": sum(1 for r in results if r['coherent']) / len(results) * 100,
                "legal_terminology_rate": sum(1 for r in results if r['legal_terminology']) / len(results) * 100,
                "avg_keyword_coverage": statistics.mean(r['keyword_coverage'] for r in results),
            }

        return task_metrics

    def calculate_court_metrics(self) -> Dict:
        """Calculate metrics by court"""
        successful = [r for r in self.results if r.get('success', False)]

        by_court = defaultdict(list)
        for result in successful:
            court = result['court']
            by_court[court].append(result)

        court_metrics = {}
        for court, results in by_court.items():
            scores = [r['score'] for r in results]

            court_metrics[court] = {
                "count": len(results),
                "avg_score": statistics.mean(scores),
                "median_score": statistics.median(scores),
                "pass_rate": sum(1 for s in scores if s >= 60) / len(scores) * 100,
                "coherence_rate": sum(1 for r in results if r['coherent']) / len(results) * 100,
            }

        return court_metrics

    def get_top_bottom_examples(self, n: int = 10) -> Tuple[List, List]:
        """Get top and bottom N examples by score"""
        successful = [r for r in self.results if r.get('success', False)]
        sorted_results = sorted(successful, key=lambda x: x['score'], reverse=True)

        top_n = sorted_results[:n]
        bottom_n = sorted_results[-n:]

        return top_n, bottom_n

    def export_results(self):
        """Export results to multiple formats"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = Path("scripts/evaluation_results")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Calculate metrics
        overall_metrics = self.calculate_overall_metrics()
        task_metrics = self.calculate_task_type_metrics()
        court_metrics = self.calculate_court_metrics()
        top_10, bottom_10 = self.get_top_bottom_examples(10)

        # Export JSON
        json_file = output_dir / f"val_810_evaluation_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump({
                "metadata": {
                    "model": self.model_name,
                    "evaluation_date": datetime.now().isoformat(),
                    "start_time": self.start_time.isoformat() if self.start_time else None,
                    "end_time": self.end_time.isoformat() if self.end_time else None,
                    "duration_minutes": (self.end_time - self.start_time).total_seconds() / 60 if self.start_time and self.end_time else None,
                    "total_examples": len(self.results),
                },
                "overall_metrics": overall_metrics,
                "by_task_type": task_metrics,
                "by_court": court_metrics,
                "top_10": top_10,
                "bottom_10": bottom_10,
                "all_results": self.results
            }, f, indent=2)

        print(f"✓ Exported JSON: {json_file}")

        # Export CSV
        csv_file = output_dir / f"val_810_evaluation_{timestamp}.csv"
        with open(csv_file, 'w') as f:
            # Header
            f.write("Index,CNR,Task Type,Court,Success,Score,Keyword Coverage,Coherent,Legal Terms,Hallucination Risk,Complete,Response Length\n")

            # Data
            for r in self.results:
                f.write(f"{r['index']},{r['cnr']},{r['task_type']},{r['court']},")
                if r.get('success'):
                    f.write(f"Yes,{r['score']},{r['keyword_coverage']},{r['coherent']},{r['legal_terminology']},{r['hallucination_risk']},{r['complete']},{r['response_length']}\n")
                else:
                    f.write(f"No,0,0,False,False,N/A,False,0\n")

        print(f"✓ Exported CSV: {csv_file}")

        # Generate markdown report
        self.generate_markdown_report(overall_metrics, task_metrics, court_metrics, top_10, bottom_10, timestamp)

        return json_file, csv_file

    def generate_markdown_report(self, overall_metrics, task_metrics, court_metrics, top_10, bottom_10, timestamp):
        """Generate comprehensive markdown report"""
        report_file = Path("docs") / "AUTOMATED_EVALUATION_REPORT_810.md"

        # Calculate duration
        if self.start_time and self.end_time:
            duration_str = f"{(self.end_time - self.start_time).total_seconds() / 60:.1f}"
        else:
            duration_str = "N/A"

        with open(report_file, 'w') as f:
            f.write(f"""# Automated Evaluation Report - 810 Validation Examples

**Model**: {self.model_name}
**Evaluation Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Duration**: {duration_str} minutes
**Total Examples**: {overall_metrics['total_examples']}

---

## Executive Summary

### Key Findings

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| **Overall Score** | **{overall_metrics['avg_score']:.1f}/100** | >60 | {'✅ PASS' if overall_metrics['avg_score'] >= 60 else '❌ FAIL'} |
| **Pass Rate (>60)** | **{overall_metrics['pass_rate_60']:.1f}%** | >70% | {'✅' if overall_metrics['pass_rate_60'] >= 70 else '❌'} |
| **Coherence Rate** | **{overall_metrics['coherence_rate']:.1f}%** | >80% | {'✅' if overall_metrics['coherence_rate'] >= 80 else '❌'} |
| **Legal Terminology** | **{overall_metrics['legal_terminology_rate']:.1f}%** | >70% | {'✅' if overall_metrics['legal_terminology_rate'] >= 70 else '❌'} |
| **Keyword Coverage** | **{overall_metrics['avg_keyword_coverage']:.1f}%** | >30% | {'✅' if overall_metrics['avg_keyword_coverage'] >= 30 else '❌'} |
| **Hallucination (High Risk)** | **{overall_metrics['hallucination_high']:.1f}%** | <50% | {'✅' if overall_metrics['hallucination_high'] < 50 else '❌'} |

### Verdict

**{'✅ PASS' if overall_metrics['avg_score'] >= 60 and overall_metrics['pass_rate_60'] >= 70 else '⚠️ NEEDS IMPROVEMENT'}**

The model {'meets' if overall_metrics['avg_score'] >= 60 else 'does not meet'} the baseline quality threshold on the 810-example validation set.

---

## Overall Performance

### Score Distribution

| Score Range | Count | Percentage |
|-------------|-------|------------|
""")

            # Calculate score distribution
            successful = [r for r in self.results if r.get('success', False)]
            score_ranges = [
                (90, 100, "90-100 (Excellent)"),
                (80, 89, "80-89 (Very Good)"),
                (70, 79, "70-79 (Good)"),
                (60, 69, "60-69 (Fair)"),
                (50, 59, "50-59 (Poor)"),
                (0, 49, "0-49 (Fail)")
            ]

            for low, high, label in score_ranges:
                count = sum(1 for r in successful if low <= r['score'] <= high)
                pct = count / len(successful) * 100 if successful else 0
                f.write(f"| {label} | {count} | {pct:.1f}% |\n")

            f.write(f"""

**Median Score**: {overall_metrics['median_score']:.1f}/100
**Standard Deviation**: {overall_metrics['std_dev']:.1f}
**25th Percentile**: {overall_metrics.get('percentile_25', 'N/A')}
**75th Percentile**: {overall_metrics.get('percentile_75', 'N/A')}
**90th Percentile**: {overall_metrics.get('percentile_90', 'N/A')}

---

## Performance by Task Type

| Task Type | Count | Avg Score | Pass Rate | Coherence | Legal Terms | Coverage |
|-----------|-------|-----------|-----------|-----------|-------------|----------|
""")

            for task_type, metrics in sorted(task_metrics.items()):
                f.write(f"| {task_type} | {metrics['count']} | {metrics['avg_score']:.1f} | {metrics['pass_rate']:.1f}% | {metrics['coherence_rate']:.1f}% | {metrics['legal_terminology_rate']:.1f}% | {metrics['avg_keyword_coverage']:.1f}% |\n")

            f.write(f"""

---

## Performance by Court

| Court | Count | Avg Score | Pass Rate | Coherence |
|-------|-------|-----------|-----------|-----------|
""")

            for court, metrics in sorted(court_metrics.items()):
                f.write(f"| {court} | {metrics['count']} | {metrics['avg_score']:.1f} | {metrics['pass_rate']:.1f}% | {metrics['coherence_rate']:.1f}% |\n")

            f.write(f"""

---

## Top 10 Examples (Highest Scores)

| Rank | CNR | Task Type | Score |
|------|-----|-----------|-------|
""")

            for i, example in enumerate(top_10, 1):
                f.write(f"| {i} | {example['cnr']} | {example['task_type']} | {example['score']:.1f} |\n")

            f.write(f"""

---

## Bottom 10 Examples (Lowest Scores)

| Rank | CNR | Task Type | Score | Issue |
|------|-----|-----------|-------|-------|
""")

            for i, example in enumerate(bottom_10, 1):
                issue = "Short response" if example['response_length'] < 50 else \
                        "High hallucination risk" if example['hallucination_risk'] == 'High' else \
                        "Low keyword coverage" if example['keyword_coverage'] < 20 else \
                        "Incoherent" if not example['coherent'] else "Unknown"
                f.write(f"| {i} | {example['cnr']} | {example['task_type']} | {example['score']:.1f} | {issue} |\n")

            f.write(f"""

---

## Comparison with 20-Test Baseline

| Metric | 20-Test Baseline | 810-Example Eval | Change |
|--------|------------------|------------------|--------|
| Overall Score | 63.9/100 | {overall_metrics['avg_score']:.1f}/100 | {overall_metrics['avg_score'] - 63.9:+.1f} |
| Coherence Rate | 90% | {overall_metrics['coherence_rate']:.1f}% | {overall_metrics['coherence_rate'] - 90:+.1f}% |
| Legal Terminology | 75% | {overall_metrics['legal_terminology_rate']:.1f}% | {overall_metrics['legal_terminology_rate'] - 75:+.1f}% |

---

## Recommendations

Based on this comprehensive evaluation of {overall_metrics['total_examples']} examples:

1. **Overall Performance**: {'The model meets baseline quality standards.' if overall_metrics['avg_score'] >= 60 else 'The model needs improvement to meet baseline standards.'}

2. **Strongest Task Type**: {max(task_metrics.items(), key=lambda x: x[1]['avg_score'])[0]} (Avg: {max(task_metrics.items(), key=lambda x: x[1]['avg_score'])[1]['avg_score']:.1f}/100)

3. **Weakest Task Type**: {min(task_metrics.items(), key=lambda x: x[1]['avg_score'])[0]} (Avg: {min(task_metrics.items(), key=lambda x: x[1]['avg_score'])[1]['avg_score']:.1f}/100)

4. **Key Findings**:
   - {overall_metrics['pass_rate_60']:.1f}% of examples scored above 60/100
   - {overall_metrics['hallucination_high']:.1f}% flagged as high hallucination risk
   - Average response length: {overall_metrics['avg_response_length']:.0f} words

---

**Full Results**: `scripts/evaluation_results/val_810_evaluation_{timestamp}.json`
**CSV Export**: `scripts/evaluation_results/val_810_evaluation_{timestamp}.csv`

**Last Updated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
""")

        print(f"✓ Generated report: {report_file}")
        return report_file

    def print_summary(self):
        """Print summary to console"""
        print("="*70)
        print("EVALUATION SUMMARY")
        print("="*70)

        metrics = self.calculate_overall_metrics()

        print(f"\nTotal Examples: {metrics['total_examples']}")
        print(f"Successful: {metrics['successful']} ({metrics['success_rate']:.1f}%)")
        print(f"Failed: {metrics['failed']}")

        print(f"\nOverall Score: {metrics['avg_score']:.1f}/100")
        print(f"Median Score: {metrics['median_score']:.1f}/100")
        print(f"Std Dev: {metrics['std_dev']:.1f}")

        print(f"\nPass Rates:")
        print(f"  >60: {metrics['pass_rate_60']:.1f}%")
        print(f"  >70: {metrics['pass_rate_70']:.1f}%")
        print(f"  >80: {metrics['pass_rate_80']:.1f}%")

        print(f"\nQuality Metrics:")
        print(f"  Coherence: {metrics['coherence_rate']:.1f}%")
        print(f"  Legal Terminology: {metrics['legal_terminology_rate']:.1f}%")
        print(f"  Avg Keyword Coverage: {metrics['avg_keyword_coverage']:.1f}%")
        print(f"  Complete Responses: {metrics['complete_rate']:.1f}%")

        print(f"\nHallucination Risk:")
        print(f"  Low: {metrics['hallucination_low']:.1f}%")
        print(f"  Medium: {metrics['hallucination_medium']:.1f}%")
        print(f"  High: {metrics['hallucination_high']:.1f}%")

        print()


def main():
    parser = argparse.ArgumentParser(description="Evaluate model on 810 validation examples")
    parser.add_argument("--model", default="nyay-ai", help="Ollama model name")
    parser.add_argument("--val-file", default="data/training/val.jsonl", help="Validation file")
    parser.add_argument("--limit", type=int, help="Limit number of examples (for testing)")

    args = parser.parse_args()

    # Initialize evaluator
    evaluator = ValidationSetEvaluator(
        model_name=args.model,
        val_file=args.val_file
    )

    # Run evaluation
    evaluator.run_evaluation(limit=args.limit)

    # Print summary
    evaluator.print_summary()

    # Export results
    print("="*70)
    print("EXPORTING RESULTS")
    print("="*70)
    evaluator.export_results()

    print()
    print("="*70)
    print("✓ EVALUATION COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
