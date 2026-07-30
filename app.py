from utils.search import search
from utils.gemini_api import ask_gemini

query = input("Ask your legal question: ")

results = search(query)

if not results:
    print("No results found.")
    exit()

# Use only the best result
best = results[0]

print("\nBest Match")
print("=" * 70)
print("Source:", best["source"])
print("Page:", best["page"])
print(best["text"])

answer = ask_gemini(best["text"], query)

print("\n" + "=" * 70)
print("LEGAL ASSISTANT")
print("=" * 70)
print(answer)