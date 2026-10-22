
from rag import retrieve

tests = [
    {
        "question": "How many annual leave days do employees receive?",
        "should_answer": True,
    },
    {
        "question": "How many days can employees work remotely?",
        "should_answer": True,
    },
    {
        "question": "How many unused annual leave days can be carried forward?",
        "should_answer": True,
    },
    {
        "question": "What is the company's maternity leave policy?",
        "should_answer": False,
    },
    {
        "question": "Who is the CEO of the company?",
        "should_answer": False,
    },
]

thresholds = [0.4, 0.6, 0.8, 1.0, 1.2]


def evaluate_threshold(threshold):
    false_accepts = 0
    false_rejects = 0
    correct = 0

    for test in tests:
        # Use only the closest chunk for this experiment.
        results = retrieve(test["question"], top_k=1)

        accepted = (
            bool(results)
            and results[0]["distance"] <= threshold
        )

        expected = test["should_answer"]

        if accepted == expected:
            correct += 1
        elif accepted:
            false_accepts += 1
        else:
            false_rejects += 1

    return correct, false_accepts, false_rejects


if __name__ == "__main__":
    print("THRESHOLD CALIBRATION")
    print("-" * 55)

    for threshold in thresholds:
        correct, false_accepts, false_rejects = (
            evaluate_threshold(threshold)
        )

        print(
            f"Threshold: {threshold:.1f} | "
            f"Correct: {correct}/{len(tests)} | "
            f"False accepts: {false_accepts} | "
            f"False rejects: {false_rejects}"
        )
