from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

def ask_groq(context, question):

    prompt = f"""
You are an expert Indian Legal Assistant.

Answer ONLY using the legal context below.

If the answer is not available in the context, reply:

"I could not find this information in the provided legal documents."

Legal Context:
{context}

Question:
{question}

Answer:
"""

    try:

        response = client.chat.completions.create(

            model="llama-3.3-70b-versatile",

            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],

            temperature=0.2,
            max_tokens=1024

        )

        return response.choices[0].message.content

    except Exception as e:

        return f"Error: {e}"