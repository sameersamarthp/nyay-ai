#!/usr/bin/env python3
"""
Manual Quality Review Script
Tests the deployed nyay-ai model with 20 diverse legal queries
"""

import json
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List


# Test queries covering different legal domains
TEST_QUERIES = [
    # Constitutional Law
    {
        "id": 1,
        "category": "Constitutional Law",
        "query": "What is the scope of Article 226 of the Indian Constitution?",
        "expected_topics": ["High Court", "writ jurisdiction", "fundamental rights", "legal rights"]
    },
    {
        "id": 2,
        "category": "Constitutional Law",
        "query": "Explain the difference between Articles 32 and 226.",
        "expected_topics": ["Supreme Court", "High Court", "writs", "fundamental rights", "jurisdiction"]
    },
    {
        "id": 3,
        "category": "Constitutional Law",
        "query": "What are the grounds for issuing a writ of mandamus?",
        "expected_topics": ["public duty", "legal right", "public authority", "enforcement"]
    },

    # Criminal Law - IPC
    {
        "id": 4,
        "category": "Criminal Law - IPC",
        "query": "What is the punishment for Section 304B IPC (dowry death)?",
        "expected_topics": ["imprisonment", "not less than 7 years", "life imprisonment", "dowry"]
    },
    {
        "id": 5,
        "category": "Criminal Law - IPC",
        "query": "Explain the ingredients of Section 420 IPC (cheating).",
        "expected_topics": ["dishonestly", "inducement", "property", "deception", "mens rea"]
    },

    # Criminal Procedure - CrPC
    {
        "id": 6,
        "category": "Criminal Procedure",
        "query": "Under which section can High Court quash an FIR?",
        "expected_topics": ["Section 482", "inherent powers", "abuse of process", "High Court"]
    },
    {
        "id": 7,
        "category": "Criminal Procedure",
        "query": "What is the time limit for filing a chargesheet in CrPC?",
        "expected_topics": ["60 days", "90 days", "custody", "investigation", "Section 167"]
    },
    {
        "id": 8,
        "category": "Criminal Procedure",
        "query": "What is anticipatory bail and under which section is it granted?",
        "expected_topics": ["Section 438", "apprehension of arrest", "pre-arrest bail", "conditions"]
    },

    # Civil Law - CPC
    {
        "id": 9,
        "category": "Civil Procedure",
        "query": "What is the limitation period for filing a suit for recovery of money?",
        "expected_topics": ["3 years", "Limitation Act", "debt", "contract"]
    },
    {
        "id": 10,
        "category": "Civil Procedure",
        "query": "Explain the doctrine of res judicata under CPC.",
        "expected_topics": ["Section 11", "finality", "same parties", "same matter", "bar"]
    },

    # Contract Law
    {
        "id": 11,
        "category": "Contract Law",
        "query": "What makes a contract void under the Indian Contract Act?",
        "expected_topics": ["consideration", "competency", "free consent", "lawful object", "Section 10"]
    },
    {
        "id": 12,
        "category": "Contract Law",
        "query": "What is specific performance of contract?",
        "expected_topics": ["Specific Relief Act", "actual performance", "damages inadequate", "discretion"]
    },

    # Property Law
    {
        "id": 13,
        "category": "Property Law",
        "query": "What is the difference between lease and license?",
        "expected_topics": ["exclusive possession", "transfer of interest", "easement", "permission"]
    },

    # Evidence Law
    {
        "id": 14,
        "category": "Evidence Law",
        "query": "What is the burden of proof in criminal cases?",
        "expected_topics": ["prosecution", "beyond reasonable doubt", "presumption of innocence", "Section 101"]
    },

    # Labor Law
    {
        "id": 15,
        "category": "Labor Law",
        "query": "What is the notice period required for retrenchment under Industrial Disputes Act?",
        "expected_topics": ["one month", "Section 25F", "compensation", "notice"]
    },

    # Tax Law
    {
        "id": 16,
        "category": "Tax Law",
        "query": "What is the time limit for income tax assessment?",
        "expected_topics": ["assessment year", "9 months", "21 months", "search cases"]
    },

    # Family Law
    {
        "id": 17,
        "category": "Family Law",
        "query": "What are the grounds for divorce under Hindu Marriage Act?",
        "expected_topics": ["adultery", "cruelty", "desertion", "conversion", "mental disorder"]
    },

    # Consumer Law
    {
        "id": 18,
        "category": "Consumer Law",
        "query": "What is the pecuniary jurisdiction of District Consumer Forum?",
        "expected_topics": ["up to 1 crore", "rupees", "State Commission", "National Commission"]
    },

    # Legal Procedures
    {
        "id": 19,
        "category": "Legal Procedures",
        "query": "What is the procedure for filing a PIL (Public Interest Litigation)?",
        "expected_topics": ["Article 32", "Article 226", "public interest", "locus standi", "Supreme Court", "High Court"]
    },

    # Recent Developments
    {
        "id": 20,
        "category": "Recent Legislation",
        "query": "What are the key changes in the Bharatiya Nyaya Sanhita compared to IPC?",
        "expected_topics": ["BNS", "IPC replacement", "2023", "new provisions", "amendments"]
    }
]


def query_ollama_model(prompt: str, model: str = "nyay-ai") -> Dict:
    """Query the Ollama model and return response with metadata."""
    try:
        cmd = [
            "ollama", "run", model,
            prompt
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120  # 2 minute timeout
        )

        if result.returncode == 0:
            return {
                "success": True,
                "response": result.stdout.strip(),
                "error": None
            }
        else:
            return {
                "success": False,
                "response": None,
                "error": result.stderr.strip()
            }
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "response": None,
            "error": "Query timed out after 120 seconds"
        }
    except Exception as e:
        return {
            "success": False,
            "response": None,
            "error": str(e)
        }


def assess_response_quality(response: str, expected_topics: List[str]) -> Dict:
    """Manually assess response quality based on expected topics."""
    response_lower = response.lower()

    # Check coverage of expected topics
    topics_covered = [
        topic for topic in expected_topics
        if any(word in response_lower for word in topic.lower().split())
    ]

    coverage_score = len(topics_covered) / len(expected_topics) * 100

    # Basic quality checks
    checks = {
        "not_empty": len(response) > 0,
        "sufficient_length": len(response) > 100,
        "has_structure": any(marker in response for marker in ['\n', '.', ':']),
        "no_refusal": not any(phrase in response_lower for phrase in [
            "i cannot", "i don't know", "i'm not sure", "unable to"
        ]),
        "has_legal_terms": any(term in response_lower for term in [
            "section", "act", "article", "court", "provision", "law"
        ])
    }

    quality_score = sum(checks.values()) / len(checks) * 100

    return {
        "coverage_score": coverage_score,
        "quality_score": quality_score,
        "topics_covered": topics_covered,
        "topics_missed": [t for t in expected_topics if t not in topics_covered],
        "checks": checks,
        "response_length": len(response),
        "overall_rating": "EXCELLENT" if quality_score >= 80 and coverage_score >= 70 else
                         "GOOD" if quality_score >= 60 and coverage_score >= 50 else
                         "FAIR" if quality_score >= 40 else "POOR"
    }


def run_manual_quality_review():
    """Run the complete manual quality review."""
    print("=" * 80)
    print("MANUAL QUALITY REVIEW - Nyay AI Model")
    print("=" * 80)
    print(f"\nStarted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total queries: {len(TEST_QUERIES)}")
    print(f"Model: nyay-ai (deployed with Ollama)\n")

    results = []

    for i, test_case in enumerate(TEST_QUERIES, 1):
        print(f"\n[{i}/{len(TEST_QUERIES)}] Testing: {test_case['category']}")
        print(f"Query: {test_case['query']}")
        print("-" * 80)

        # Query the model
        response_data = query_ollama_model(test_case['query'])

        if not response_data['success']:
            print(f"❌ ERROR: {response_data['error']}")
            results.append({
                **test_case,
                "success": False,
                "error": response_data['error'],
                "timestamp": datetime.now().isoformat()
            })
            continue

        response = response_data['response']

        # Assess quality
        assessment = assess_response_quality(response, test_case['expected_topics'])

        # Print results
        print(f"\n✅ Response received ({assessment['response_length']} chars)")
        print(f"\nRating: {assessment['overall_rating']}")
        print(f"  - Quality Score: {assessment['quality_score']:.1f}%")
        print(f"  - Topic Coverage: {assessment['coverage_score']:.1f}%")
        print(f"  - Topics Covered: {', '.join(assessment['topics_covered']) if assessment['topics_covered'] else 'None'}")
        if assessment['topics_missed']:
            print(f"  - Topics Missed: {', '.join(assessment['topics_missed'])}")

        # Show first 300 chars of response
        preview = response[:300] + "..." if len(response) > 300 else response
        print(f"\nResponse Preview:\n{preview}")

        # Store results
        results.append({
            **test_case,
            "success": True,
            "response": response,
            "assessment": assessment,
            "timestamp": datetime.now().isoformat()
        })

    # Generate summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    successful = [r for r in results if r.get('success', False)]
    failed = [r for r in results if not r.get('success', True)]

    print(f"\nTotal Queries: {len(TEST_QUERIES)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")

    if successful:
        ratings = [r['assessment']['overall_rating'] for r in successful]
        rating_counts = {
            "EXCELLENT": ratings.count("EXCELLENT"),
            "GOOD": ratings.count("GOOD"),
            "FAIR": ratings.count("FAIR"),
            "POOR": ratings.count("POOR")
        }

        avg_quality = sum(r['assessment']['quality_score'] for r in successful) / len(successful)
        avg_coverage = sum(r['assessment']['coverage_score'] for r in successful) / len(successful)

        print(f"\nRating Distribution:")
        print(f"  - EXCELLENT: {rating_counts['EXCELLENT']} ({rating_counts['EXCELLENT']/len(successful)*100:.1f}%)")
        print(f"  - GOOD: {rating_counts['GOOD']} ({rating_counts['GOOD']/len(successful)*100:.1f}%)")
        print(f"  - FAIR: {rating_counts['FAIR']} ({rating_counts['FAIR']/len(successful)*100:.1f}%)")
        print(f"  - POOR: {rating_counts['POOR']} ({rating_counts['POOR']/len(successful)*100:.1f}%)")

        print(f"\nAverage Scores:")
        print(f"  - Quality Score: {avg_quality:.1f}%")
        print(f"  - Topic Coverage: {avg_coverage:.1f}%")

        # Category breakdown
        print(f"\nPerformance by Category:")
        categories = {}
        for r in successful:
            cat = r['category']
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(r['assessment'])

        for cat, assessments in sorted(categories.items()):
            avg_cat_quality = sum(a['quality_score'] for a in assessments) / len(assessments)
            avg_cat_coverage = sum(a['coverage_score'] for a in assessments) / len(assessments)
            print(f"  - {cat}: Quality={avg_cat_quality:.1f}%, Coverage={avg_cat_coverage:.1f}%")

    # Save results to file
    output_dir = Path("data/training")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "manual_quality_review_results.json"

    with open(output_file, 'w') as f:
        json.dump({
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "model": "nyay-ai",
                "total_queries": len(TEST_QUERIES),
                "successful": len(successful),
                "failed": len(failed)
            },
            "summary": {
                "rating_distribution": rating_counts if successful else {},
                "average_quality_score": avg_quality if successful else 0,
                "average_coverage_score": avg_coverage if successful else 0
            },
            "results": results
        }, f, indent=2)

    print(f"\n✅ Results saved to: {output_file}")
    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    run_manual_quality_review()
