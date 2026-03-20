import os
import re
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def is_random_text(answer):
    """
    Detect clearly meaningless answers like 'asdf', 'qwerty', etc.
    """

    cleaned = answer.strip()

    # Too short (almost certainly meaningless)
    if len(cleaned) <= 5:
        return True

    # Check if answer contains real words
    words = re.findall(r"[a-zA-Z]{3,}", cleaned)

    if len(words) == 0:
        return True

    return False


def evaluate_answer(answer):

    # -------- RANDOM TEXT CHECK --------

    if is_random_text(answer):

        return """
Score: 0/10

Strengths:
- None

Weaknesses:
- The answer appears to be random or meaningless text.
- It does not demonstrate understanding of the question.

Suggestions:
- Provide a clear explanation related to the question.
- Use meaningful technical terms and examples.
"""

    # -------- NORMAL AI EVALUATION --------

    prompt = f"""
You are a technical interviewer.

Evaluate the following answer.

Answer:
{answer}

Scoring rules:
- 0–3: Incorrect or irrelevant
- 4–6: Partially correct
- 7–8: Good answer
- 9–10: Excellent answer

Return:

Score: (0-10)

Strengths:
- bullet points

Weaknesses:
- bullet points

Suggestions:
- bullet points
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )

    return response.choices[0].message.content