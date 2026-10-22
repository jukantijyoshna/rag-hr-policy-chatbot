
from rag import retrieve

MAX_DISTANCE = 0.8

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


def evaluate_abstention():
    correct = 0

    for test in tests:
        question = test["question"]
        should_answer = test["should_answer"]

        results = retrieve(question, top_k=3)

        accepted = any(
            item["distance"] <= MAX_DISTANCE
            for item in results
        )

        is_correct = accepted == should_answer
        correct += int(is_correct)

        if accepted:
            decision = "ANSWER"
        else:
            decision = "ABSTAIN"

        print(f"\nQuestion: {question}")
        print(f"Expected: {'ANSWER' if should_answer else 'ABSTAIN'}")
        print(f"Actual:   {decision}")
        print(f"Correct:  {is_correct}")

    accuracy = correct / len(tests)

    print("\n===== ABSTENTION EVALUATION =====")
    print(f"Correct decisions: {correct}/{len(tests)}")
    print(f"Decision accuracy: {accuracy:.2%}")


if __name__ == "__main__":
    evaluate_abstention()
