import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import chromadb

load_dotenv()

CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_store")
COLLECTION_NAME = "business_context"

BUSINESS_CONTEXT = [
    "Arnav Mehta is the Head of IT. He is the highest-ranking person in the IT department.",
    "A 'high performer' is any employee with a performance_rating greater than 4.5.",
    "A 'low performer' is any employee with a performance_rating below 4.0.",
    "Sneha Iyer is the Sales Manager and leads the Sales department.",
    "Vikram Singh is the HR Manager and leads the HR department.",
    "'Senior' employees are those hired before January 1, 2020 (i.e. hire_date < 2020-01-01).",
    "'New hires' refers to employees hired within the last 12 months from today.",
    "Salary figures in the database are annual gross salary in USD, not monthly.",
    "The company has exactly three departments: IT, Sales, and HR. There is no 'Finance' or 'Marketing' department in this database.",
    "When a user asks about 'headcount', they mean the total count of employees, optionally filtered by department.",
]


def build_index():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(COLLECTION_NAME)
    vectors = embeddings.embed_documents(BUSINESS_CONTEXT)

    collection.add(
        ids=[f"fact_{i}" for i in range(len(BUSINESS_CONTEXT))],
        documents=BUSINESS_CONTEXT,
        embeddings=vectors,
    )

    print(f"Indexed {len(BUSINESS_CONTEXT)} business-context facts into "
          f"Chroma collection '{COLLECTION_NAME}' at '{CHROMA_DIR}'.")


if __name__ == "__main__":
    build_index()
