# AI-Mock-Interviewer

An AI-powered mock interview system using RAG + LLaMA 3.1.

## Setup
1. Clone the repo
2. Install dependencies: `pip install -r requirements.txt`
3. Create a `.env` file with your Groq API key:
   GROQ_API_KEY=your_key_here
4. Run: `streamlit run app.py`

# Brief Description of the project
# Objective: 
To build a personalized, AI-driven mock interview platform that generates interview questions from a user's own study material and evaluates their responses automatically — eliminating reliance on static quiz banks or human interviewers.
Methodology: The system follows a Retrieval-Augmented Generation (RAG) pipeline. A user uploads a document (PDF, DOCX, or TXT), which is split into overlapping 500-character chunks and converted into vector embeddings using the all-MiniLM-L6-v2 sentence transformer. These embeddings are stored in ChromaDB for fast similarity search. When generating questions, the most relevant chunks are retrieved and passed to a Groq-hosted LLaMA 3.1-8B model, which produces difficulty-adaptive questions at Beginner, Intermediate, or Advanced levels. After the candidate submits a response, the same LLM evaluates it against a structured rubric and returns a numeric score (0–10), strengths, weaknesses, and improvement suggestions. The entire flow is orchestrated through a Streamlit web interface with live score tracking.

# key Ouctcomes:
The system produces contextually accurate, document-grounded questions rather than generic ones. Evaluations are structured and actionable. It runs entirely on a standard laptop CPU with no GPU or custom model training required, making it practical and accessible. Experimental results showed that detailed responses consistently scored 8–10, while incomplete answers scored 3–5, confirming the evaluator's rubric alignment.

