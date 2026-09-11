import pytest

from src.sql_safety import validate_read_only_query


@pytest.mark.parametrize(
    "query",
    [
        "SELECT * FROM employees",
        "select count(*) from employees;",
        "WITH active AS (SELECT * FROM employees WHERE performance_rating > 4) SELECT * FROM active",
        "SELECT name FROM departments UNION SELECT role FROM employees",
    ],
)
def test_accepts_read_only_queries(query):
    normalized = validate_read_only_query(query)
    assert "SELECT" in normalized.upper()


@pytest.mark.parametrize(
    "query",
    [
        "",
        "DELETE FROM employees",
        "UPDATE employees SET salary = 0",
        "INSERT INTO employees (full_name) VALUES ('test')",
        "DROP TABLE employees",
        "SELECT 1; DROP TABLE employees",
        "WITH removed AS (DELETE FROM employees RETURNING *) SELECT * FROM removed",
        "CREATE TABLE copied AS SELECT * FROM employees",
    ],
)
def test_rejects_unsafe_or_invalid_queries(query):
    with pytest.raises(ValueError):
        validate_read_only_query(query)


def test_allows_dangerous_words_inside_string_literals():
    normalized = validate_read_only_query("SELECT 'drop table employees' AS example")
    assert "drop table employees" in normalized
