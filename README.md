# Agentic RAG Text-to-SQL Telegram Bot

Live bot: https://t.me/agenticragdb_bot

A Telegram bot that turns natural language questions into SQL queries, runs them
against a Postgres database, and returns the results as a PDF report.

Built with LangGraph, Gemini, and ChromaDB.

## How it works

1. A router node decides if the question needs extra business context (e.g. what
   counts as a "high performer") or can be answered from the schema alone.
2. If it needs context, a RAG step pulls relevant facts from a ChromaDB collection.
3. Gemini generates a SQL query from the schema + context.
4. The query runs against Postgres. If it fails, the error is fed back to the model
   and it retries (up to 3 times).
5. Results are rendered into a PDF and sent back on Telegram.

## Stack

- LangGraph for the agent graph
- Gemini (`gemini-3.5-flash`) for routing and SQL generation
- ChromaDB + Gemini embeddings for RAG
- PostgreSQL as the target database
- python-telegram-bot for the Telegram interface

## RAG setup

The vector store is ChromaDB, running as a local persistent client (`./chroma_store`)
on the same instance as the bot — no separate vector DB service to manage.

Chunking here is one fact per chunk rather than splitting a long document. Each
entry in `BUSINESS_CONTEXT` (`src/rag_index.py`) is a single, self-contained
statement ("Arnav Mehta is the Head of IT.", "A 'high performer' is any employee
with a performance_rating greater than 4.5."). Keeping each chunk atomic means a
similarity search pulls back one clean fact instead of a paragraph with the
relevant part buried in it, which matters more here than chunk size since none
of the source facts are long enough to need splitting.

**Example trace** for the question *"Who are the high performers in IT?"*:

1. Router sees the phrase "high performer" isn't a column name in the schema and
   routes to `vector_search`.
2. The question is embedded and matched against the Chroma collection. Top hits:
   - `"A 'high performer' is any employee with a performance_rating greater than 4.5."`
   - `"Arnav Mehta is the Head of IT."`
3. Those facts are appended to the SQL-generation prompt alongside the schema.
4. Gemini writes:
   ```sql
   SELECT full_name, performance_rating
   FROM employees e
   JOIN departments d ON e.department_id = d.department_id
   WHERE d.name = 'IT' AND e.performance_rating > 4.5;
   ```
5. The query runs, and the result is rendered to PDF.

Without the RAG step, the model has no way to know what "high performer" means
in this schema — `performance_rating` alone doesn't imply a threshold.

## PDF generation and delivery

The query result (a pandas DataFrame) is rendered into a PDF with `fpdf2`, in the
`pdf_node` function in `src/agent.py`. It builds a landscape A4 page, writes the
question at the top, then draws the DataFrame as a bordered grid — one header
row from `df.columns`, then one row per record, each cell truncated to 25
characters so it fits. The file is written to `/tmp/report_<random>.pdf`.

That path comes back out of the graph as `pdf_path`. `src/bot.py` opens the file
and sends it with `context.bot.send_document(...)` from python-telegram-bot,
then deletes it from `/tmp` in a `finally` block once it's been sent.

## Deployment

Deployed on a single cloud VM (EC2 in this case, but any Ubuntu instance works
the same way): Postgres and ChromaDB both run locally on the box, the bot process
is kept alive with the systemd unit in `deploy/`, and it only needs outbound
access to Telegram's and Google's APIs — no inbound ports beyond SSH.

## Project structure

```
telegram-sql-bot/
├── src/
│   ├── agent.py       # LangGraph state machine
│   ├── bot.py          # Telegram entry point
│   └── rag_index.py    # builds the ChromaDB business-context index
├── deploy/
│   └── telegram-sql-bot.service
├── requirements.txt
└── .env.example
```

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add your keys and DB credentials
python src/rag_index.py
python src/bot.py
```

## Running as a service

```bash
sudo cp deploy/telegram-sql-bot.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now telegram-sql-bot
```

## Notes

- Only `SELECT` queries are allowed to run — the execution node blocks
  destructive statements as a first line of defense. For a real deployment,
  the DB user should also be a read-only Postgres role.
- Schema and business-context facts in `rag_index.py` are placeholders —
  swap them for your own.
