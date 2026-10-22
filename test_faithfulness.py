
def check_faithfulness(answer, context):
    """
    A simple manual-review helper.
    It does not automatically prove that an answer is faithful.
    """
    print("CONTEXT:")
    print(context)

    print("\nANSWER:")
    print(answer)

    print("\nReview these questions:")
    print("1. Is every factual claim supported by the context?")
    print("2. Are any numbers or policies invented?")
    print("3. Does the answer contradict the context?")


context = (
    "Employees receive 20 days of annual leave each year."
)

faithful_answer = (
    "Employees receive 20 days of annual leave each year."
)

unfaithful_answer = (
    "Employees receive 25 days of annual leave and "
    "10 paid sick days."
)

print("===== TEST 1 =====")
check_faithfulness(faithful_answer, context)

print("\n===== TEST 2 =====")
check_faithfulness(unfaithful_answer, context)
