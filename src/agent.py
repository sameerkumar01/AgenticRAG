import os
import re
import uuid
from typing import TypedDict, Optional, Literal

import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, text
from fpdf import FPDF

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langgraph.graph import StateGraph, END
import chromadb

load_dotenv()

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "company_db")
DB_USER = os.getenv("DB_USER", "botuser")
DB_PASSWORD = os.getenv("DB_PASSWORD", "change_this_password")
CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_store")
MAX_SQL_RETRIES = 3

DATABASE_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
engine = create_engine(DATABASE_URL)

llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0)
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)

SCHEMA_DESCRIPTION = """
Table: departments
  - department_id (INTEGER, PRIMARY KEY)
  - name (VARCHAR)
  - location (VARCHAR)

Table: employees
  - employee_id (INTEGER, PRIMARY KEY)
  - full_name (VARCHAR)
  - email (VARCHAR)
  - role (VARCHAR)
  - department_id (INTEGER, FOREIGN KEY -> departments.department_id)
  - hire_date (DATE)
  - performance_rating (FLOAT, 1.0 to 5.0)
  - salary (INTEGER, annual USD)
"""


class AgentState(TypedDict):
    question: str
    rag_context: str
    sql_query: str
    sql_error: Optional[str]
    df_result: Optional[pd.DataFrame]
    pdf_path: Optional[str]
    retry_count: int


class RouteDecision(BaseModel):
    route: Literal["vector_search", "direct_sql"] = Field(
        description=(
            "'vector_search' if the question needs business context/definitions "
            "not obvious from raw column names (e.g. 'high performer', 'who leads IT'). "
            "'direct_sql' if the question can be answered purely from the schema "
            "(e.g. simple counts, filters on explicit columns)."
        )
    )


def router_node(state: AgentState) -> dict:
    structured_llm = llm.with_structured_output(RouteDecision)
    prompt = (
        "You are a routing agent for a text-to-SQL system.\n"
        f"Database schema:\n{SCHEMA_DESCRIPTION}\n"
        f"User question: {state['question']}\n"
        "Decide whether this question requires extra business context "
        "(vector_search) or can be answered directly from the schema (direct_sql)."
    )
    decision: RouteDecision = structured_llm.invoke(prompt)
    return {"_route": decision.route}


def rag_node(state: AgentState) -> dict:
    collection = chroma_client.get_collection("business_context")
    query_vector = embeddings.embed_query(state["question"])
    results = collection.query(query_embeddings=[query_vector], n_results=3)
    docs = results.get("documents", [[]])[0]
    context = "\n".join(f"- {d}" for d in docs)
    return {"rag_context": context}


def _extract_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and "text" in block:
                parts.append(block["text"])
        return "".join(parts)
    return str(content)


def sql_gen_node(state: AgentState) -> dict:
    error_note = ""
    if state.get("sql_error"):
        error_note = (
            f"\nThe previous SQL attempt failed with this error:\n{state['sql_error']}\n"
            f"Previous query was:\n{state.get('sql_query', '')}\n"
            "Fix the query so it runs correctly against the schema below."
        )

    prompt = (
        "You are an expert PostgreSQL query writer.\n"
        f"Schema:\n{SCHEMA_DESCRIPTION}\n"
        f"Business context (may be empty):\n{state.get('rag_context', '')}\n"
        f"User question: {state['question']}\n"
        f"{error_note}\n"
        "Return ONLY the raw SQL query. No markdown fences, no explanation, "
        "no trailing semicolon commentary. Just the SQL statement."
    )
    response = llm.invoke(prompt)
    raw_sql = _extract_text(response.content).strip()

    raw_sql = re.sub(r"^```(sql)?", "", raw_sql, flags=re.IGNORECASE).strip()
    raw_sql = re.sub(r"```$", "", raw_sql).strip()

    return {
        "sql_query": raw_sql,
        "sql_error": None,
        "retry_count": state.get("retry_count", 0) + 1,
    }


def execution_node(state: AgentState) -> dict:
    query = state["sql_query"]

    forbidden = ("drop ", "delete ", "truncate ", "alter ", "update ", "insert ")
    if query.strip().lower().startswith(forbidden):
        return {"sql_error": "Refused: only read (SELECT) queries are permitted."}

    try:
        with engine.connect() as conn:
            df = pd.read_sql_query(text(query), conn)
        return {"df_result": df, "sql_error": None}
    except Exception as e:
        return {"sql_error": str(e), "df_result": None}


def execution_router(state: AgentState) -> str:
    if state.get("sql_error") and state.get("retry_count", 0) < MAX_SQL_RETRIES:
        return "retry"
    return "continue"


def pdf_node(state: AgentState) -> dict:
    df = state.get("df_result")
    if df is None or df.empty:
        df = pd.DataFrame({"message": ["No results found for this query."]})

    pdf = FPDF(orientation="L", unit="mm", format="A4")
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "Query Report", ln=True)
    pdf.set_font("Helvetica", "", 9)
    pdf.multi_cell(0, 6, f"Question: {state['question']}")
    pdf.ln(2)

    col_width = max(20, (pdf.w - 20) / max(len(df.columns), 1))
    row_height = 7

    pdf.set_font("Helvetica", "B", 9)
    for col in df.columns:
        pdf.cell(col_width, row_height, str(col)[:25], border=1)
    pdf.ln(row_height)

    pdf.set_font("Helvetica", "", 8)
    for _, row in df.iterrows():
        for val in row:
            pdf.cell(col_width, row_height, str(val)[:25], border=1)
        pdf.ln(row_height)

    out_path = f"/tmp/report_{uuid.uuid4().hex[:8]}.pdf"
    pdf.output(out_path)
    return {"pdf_path": out_path}


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("router", router_node)
    graph.add_node("rag_search", rag_node)
    graph.add_node("sql_gen", sql_gen_node)
    graph.add_node("execute", execution_node)
    graph.add_node("make_pdf", pdf_node)

    graph.set_entry_point("router")

    graph.add_conditional_edges(
        "router",
        lambda s: s.get("_route", "direct_sql"),
        {"vector_search": "rag_search", "direct_sql": "sql_gen"},
    )
    graph.add_edge("rag_search", "sql_gen")
    graph.add_edge("sql_gen", "execute")

    graph.add_conditional_edges(
        "execute",
        execution_router,
        {"retry": "sql_gen", "continue": "make_pdf"},
    )
    graph.add_edge("make_pdf", END)

    return graph.compile()


app_graph = build_graph()


if __name__ == "__main__":
    import sys
    question = " ".join(sys.argv[1:]) or "Who are the high performers in IT?"
    result = app_graph.invoke({"question": question, "retry_count": 0})
    print("SQL used:", result.get("sql_query"))
    print("PDF at:", result.get("pdf_path"))
