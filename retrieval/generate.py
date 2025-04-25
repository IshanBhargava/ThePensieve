# models/generate.py (OpenAI GPT-based answer generator using search results)
import os
import openai
import logging
from dotenv import load_dotenv

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from retrieval.search import run_query  # Adjust the import based on your project structure

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4")

# ─── Generate Answer from Search Context ───────────────────────────────
def generate_answer(user_query: str) -> str:
    matches = run_query(user_query, top_k=20)
    context = "\n\n".join([m["metadata"]["text"] for m in matches])

# Answer the question below using only the context provided and infer information from it.
# Do not add any external knowledge or guess.
# If the answer is not in the context, respond with: "I don't know based on the given information."

    prompt = f"""
You are a thoughtful assistant. Use the provided context to analyze or interpret the question.
Base your answer on the evidence, but you're allowed to synthesize across multiple examples.
Context:
{context}

Question: {user_query}
Instructions: If the answer is not clear in the context, offer a reasoned opinion based on what is implied or evident in the text.
Answer:
"""

    response = openai.ChatCompletion.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7
    )

    answer = response["choices"][0]["message"]["content"]
    logging.info("🧠 Answer Generated:\n" + answer)
    return answer


# ─── Entrypoint ───────────────────────────────────────────────────────
if __name__ == "__main__":
    generate_answer("Is Voldemort's personality justified based on his childhood?")
    # generate_answer("write a summary of the book Harry Potter and the Philosopher's Stone")
