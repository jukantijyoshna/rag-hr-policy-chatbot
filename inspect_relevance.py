
from rag import retrieve

questions = [
    "How many annual leave days do employees receive?",
    "What is the company's maternity leave policy?",
    "Who is the CEO of the company?",
    "What's my yearly leave allowance?"
]

for question in questions:
    print(f"\nQuestion: {question}")

    results = retrieve(question, top_k=3)

    for rank, item in enumerate(results, start=1):
        print(
            f"{rank}. Distance: {item['distance']:.4f} | "
            f"{item['text']}"
        )
