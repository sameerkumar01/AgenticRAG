"""Validation helpers for model-generated PostgreSQL queries."""

from sqlglot import exp, parse
from sqlglot.errors import ParseError

_BLOCKED_NODE_NAMES = {
    "Alter",
    "Command",
    "Copy",
    "Create",
    "Delete",
    "Drop",
    "Grant",
    "Insert",
    "Into",
    "Merge",
    "Revoke",
    "Transaction",
    "TruncateTable",
    "Update",
}


def validate_read_only_query(query: str) -> str:
    """Return normalized SQL when *query* is exactly one read-only query."""
    if not query or not query.strip():
        raise ValueError("SQL query is empty.")

    try:
        statements = [statement for statement in parse(query, dialect="postgres") if statement]
    except ParseError as exc:
        raise ValueError("SQL query could not be parsed.") from exc

    if len(statements) != 1:
        raise ValueError("Exactly one SQL statement is allowed.")

    statement = statements[0]
    if not isinstance(statement, exp.Query):
        raise ValueError("Only read-only SELECT queries are allowed.")

    for node in statement.walk():
        if node.__class__.__name__ in _BLOCKED_NODE_NAMES:
            raise ValueError("Only read-only SELECT queries are allowed.")

    if statement.find(exp.Select) is None:
        raise ValueError("Only read-only SELECT queries are allowed.")

    return statement.sql(dialect="postgres")
