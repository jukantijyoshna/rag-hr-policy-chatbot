import json
from datetime import datetime, timezone
from pathlib import Path

import ollama
from rag import retrieve

BASE_DIR = Path(__file__).resolve().parent
CASES_FILE = BASE_DIR / "evaluation_cases.json"
REPORT_FILE = BASE_DIR / "evaluation_report.json"

MAX_DISTANCE = 0.8
TOP_K = 3
MODEL = "llama3.2:3b"

ABSTENTION_MESSAGE = (
    "I don't have enough relevant information in the HR policy "
    "documents to answer that question."
)


def generate_answer(question: str) -> dict:
    """Retrieve relevant context and generate an answer."""

    retrieved = retrieve(question, top_k=TOP_K)

    accepted = [
        item for item in retrieved
        if item["distance"] <= MAX_DISTANCE
    ]

    if not accepted:
        return {
            "context": "",
            "answer": ABSTENTION_MESSAGE,
            "answered": False,
            "distances": [
                round(item["distance"], 4) for item in retrieved
            ],
        }

    context = "\n\n".join(item["text"] for item in accepted)

    response = ollama.chat(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "Answer only using the supplied context. "
                    "Do not invent facts. If the context is "
                    "insufficient, explicitly say you do not "
                    "have enough information to answer."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"CONTEXT:\n{context}\n\n"
                    f"QUESTION:\n{question}"
                ),
            },
        ],
    )

    answer = response["message"]["content"].strip()

    # Human confirmation below will determine whether the
    # response actually answered or abstained.
    return {
        "context": context,
        "answer": answer,
        "answered": None,
        "distances": [
            round(item["distance"], 4) for item in accepted
        ],
    }


def ask_yes_no(prompt: str, allow_na: bool = False):
    """Collect a human evaluation label."""

    options = "Y/N/NA" if allow_na else "Y/N"

    while True:
        value = input(f"{prompt} ({options}): ").strip().upper()

        if value in ("Y", "N"):
            return value == "Y"

        if allow_na and value in ("NA", "N/A"):
            return None

        print(f"Please enter {options}.")


def main():
    with CASES_FILE.open("r", encoding="utf-8") as file:
        cases = json.load(file)

    records = []

    print(f"Loaded {len(cases)} evaluation cases.")
    print(f"Distance threshold: {MAX_DISTANCE}")
    print(f"Model: {MODEL}")

    for index, case in enumerate(cases, start=1):
        result = generate_answer(case["question"])

        print("\n" + "=" * 65)
        print(f"Case {index}/{len(cases)}: {case['id']}")
        print(f"Question: {case['question']}")
        print(f"Expected answer: {case['should_answer']}")
        print(f"Expected facts: {case['expected_facts']}")
        print(f"\nRetrieved context:\n{result['context'] or '(none accepted)'}")
        print(f"\nGenerated answer:\n{result['answer']}")

# Determine whether the generated response is an abstention.
# This matches the exact fallback message used by our pipeline.
        result["answered"] = (
            result["answer"].strip() != ABSTENTION_MESSAGE
        )

        print(
            "Detected response:",
            "Substantive answer" if result["answered"] else "Abstention",
        )
        # An answerable question should be answered.
        # An unsupported question should be refused.
        abstention_correct = (
            result["answered"] == case["should_answer"]
        )

        print(
            "Abstention decision:",
            "CORRECT" if abstention_correct else "INCORRECT",
        )

        faithful = ask_yes_no(
            "Are all factual claims supported by the context?",
            allow_na=True,
        )

        relevant = ask_yes_no(
            "Does the response appropriately address the question?",
            allow_na=True,
        )

        records.append({
            "id": case["id"],
            "question": case["question"],
            "should_answer": case["should_answer"],
            "expected_facts": case["expected_facts"],
            "retrieved_context": result["context"],
            "retrieval_distances": result["distances"],
            "answer": result["answer"],
            "answered": result["answered"],
            "faithful": faithful,
            "answer_relevant": relevant,
            "abstention_correct": abstention_correct,
        })

    def accuracy(field):
        labelled = [r[field] for r in records if r[field] is not None]

        if not labelled:
            return None

        return round(
            sum(labelled) / len(labelled) * 100, 2
        )

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": MODEL,
        "max_distance": MAX_DISTANCE,
        "top_k": TOP_K,
        "total_cases": len(records),
        "summary": {
            "faithfulness_percent": accuracy("faithful"),
            "answer_relevance_percent": accuracy("answer_relevant"),
            "abstention_correctness_percent": accuracy(
                "abstention_correct"
            ),
            "faithfulness_labelled": sum(
                r["faithful"] is not None for r in records
            ),
            "relevance_labelled": sum(
                r["answer_relevant"] is not None for r in records
            ),
            "abstention_cases_correct": sum(
                r["abstention_correct"] for r in records
            ),
        },
        "results": records,
    }

    REPORT_FILE.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n" + "=" * 65)
    print("EVALUATION SUMMARY")
    print(f"Total cases: {len(records)}")
    print(f"Faithfulness: {accuracy('faithful')}%")
    print(f"Answer relevance: {accuracy('answer_relevant')}%")
    print(f"Abstention correctness: {accuracy('abstention_correct')}%")
    print(f"Report saved to: {REPORT_FILE}")


if __name__ == "__main__":
    main()
