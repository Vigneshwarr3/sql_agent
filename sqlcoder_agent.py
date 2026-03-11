"""
sqlcoder_agent.py  —  SQLCoder-based SQL Pipeline
==================================================
Answers natural-language questions against a SQL database using the
SQLCoder model (defog/sqlcoder-7b-2 or similar).

Unlike the LangGraph agent in sql_agent.py, SQLCoder is a *completion*
model that does NOT support function / tool calls.  The entire schema is
injected directly into the prompt and the model emits plain SQL text.

Pipeline:
    1. Fetch full DDL schema from the database         (get_schema)
    2. Build the SQLCoder-formatted prompt              (build_sqlcoder_prompt)
    3. Call the model → raw text with embedded SQL
    4. Extract the SQL statement from the response     (extract_sql)
    5. Execute against the database via LangChain      (run_sqlcoder)
    6. (Optional) Summarise with a chat-capable LLM   (summarise_result)

Public API
----------
get_schema(db, table_names=None)         → DDL string
build_sqlcoder_prompt(question, schema)  → prompt string
extract_sql(text)                        → SQL string | None
run_sqlcoder(model, db, question, ...)   → result dict
summarise_result(chat_model, question, sql, result) → natural-language str
run_sqlcoder_with_summary(...)           → result dict + "answer" key
"""

import re

from langchain_community.utilities import SQLDatabase

# ─────────────────────────────────────────────────────────────────────────────
# Prompt template
# ─────────────────────────────────────────────────────────────────────────────

# SQLCoder-2 prompt format
# Reference: https://huggingface.co/defog/sqlcoder-7b-2
_SQLCODER_PROMPT = """\
### Task
Generate a SQL query to answer [QUESTION]{question}[/QUESTION]

### Database Schema
The query will run on a database with the following schema:
{schema}

### Answer
Given the database schema, here is the SQL query that answers \
[QUESTION]{question}[/QUESTION]
[SQL]
"""


# ─────────────────────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────────────────────

def get_schema(db: SQLDatabase, table_names: list | None = None) -> str:
    """
    Return CREATE TABLE DDL for the requested tables.

    Parameters
    ----------
    db          : LangChain SQLDatabase wrapper
    table_names : list of table names to include; None → all usable tables

    Returns
    -------
    str  DDL block understood by SQLCoder
    """
    if table_names is None:
        table_names = db.get_usable_table_names()
    return db.get_table_info(table_names)


def build_sqlcoder_prompt(question: str, schema: str) -> str:
    """
    Format the SQLCoder-2 prompt.

    Parameters
    ----------
    question : natural-language question from the user
    schema   : DDL string (output of get_schema)

    Returns
    -------
    str  ready-to-send prompt
    """
    return _SQLCODER_PROMPT.format(question=question, schema=schema)


def extract_sql(text: str) -> str | None:
    """
    Extract the first SQL statement from SQLCoder output.

    Handles (in order of priority):
      1. [SQL] … [/SQL]  tags   (SQLCoder-2 native format)
      2. ```sql … ```    fenced  (markdown wrapping)
      3. First SELECT/WITH/…    (plain-text fallback)

    Returns None if no SQL is found.
    """
    if not text:
        return None

    # 1. [SQL] tag
    m = re.search(r"\[SQL\](.*?)(?:\[/SQL\]|\Z)", text, re.DOTALL | re.IGNORECASE)
    if m:
        sql = re.sub(r"^```sql\s*|```\s*$", "", m.group(1).strip(),
                     flags=re.IGNORECASE).strip()
        if sql:
            return sql

    # 2. Markdown fenced block
    m = re.search(r"```sql\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if m:
        sql = m.group(1).strip()
        if sql:
            return sql

    # 3. Plain SQL starting with a keyword
    m = re.search(
        r"((?:SELECT|WITH|INSERT|UPDATE|DELETE)\b.*?)(?:;\s*$|\Z)",
        text, re.IGNORECASE | re.DOTALL,
    )
    return m.group(1).strip() if m else None


def run_sqlcoder(
    model,
    db: SQLDatabase,
    question: str,
    table_names: list | None = None,
    verbose: bool = True,
) -> dict:
    """
    Run the SQLCoder pipeline for a single question.

    Parameters
    ----------
    model       : LangChain LLM / ChatModel wrapping a SQLCoder Ollama endpoint
    db          : LangChain SQLDatabase wrapper
    question    : natural-language question
    table_names : tables to include in schema; None → all tables
    verbose     : print progress / intermediate results

    Returns
    -------
    dict with keys:
        question  str        the original question
        schema    str        DDL sent to the model
        prompt    str        full prompt sent to the model
        raw       str        raw model output
        sql       str|None   extracted SQL
        result    str|None   query result (stringified rows)
        error     str|None   exception message if execution failed
    """
    schema = get_schema(db, table_names)
    prompt = build_sqlcoder_prompt(question, schema)

    if verbose:
        print(f"\n{'='*60}")
        print(f"Question : {question}")
        print(f"{'='*60}")
        print(f"[Schema  : {len(schema)} chars | Tables: {db.get_usable_table_names()}]")

    # ── 1. Call the SQLCoder model ──────────────────────────────────────────
    response = model.invoke(prompt)
    raw = response.content if hasattr(response, "content") else str(response)

    if verbose:
        print(f"\n── SQLCoder raw output ─────────────────────────────")
        print(raw)

    # ── 2. Extract SQL ──────────────────────────────────────────────────────
    sql = extract_sql(raw)

    if verbose:
        print(f"\n── Extracted SQL ───────────────────────────────────")
        print(sql or "(none found)")

    # ── 3. Execute SQL ──────────────────────────────────────────────────────
    result = None
    error  = None

    if sql:
        try:
            result = db.run(sql)
            if verbose:
                print(f"\n── Query Result ────────────────────────────────────")
                print(result)
        except Exception as exc:
            error = str(exc)
            if verbose:
                print(f"\n── Execution Error ─────────────────────────────────")
                print(error)
    else:
        error = "No SQL could be extracted from the model response."
        if verbose:
            print(f"\n[Error] {error}")

    return {
        "question": question,
        "schema":   schema,
        "prompt":   prompt,
        "raw":      raw,
        "sql":      sql,
        "result":   result,
        "error":    error,
    }


def summarise_result(
    chat_model,
    question: str,
    sql: str | None,
    result: str | None,
) -> str:
    """
    Use a chat-capable LLM to turn raw query results into a plain-English answer.

    This is useful when you have a separate chat model (e.g. llama3.1) available
    alongside SQLCoder for the final answer generation step.

    Parameters
    ----------
    chat_model : any LangChain chat model (ChatOllama, ChatGoogleGenerativeAI, …)
    question   : the original user question
    sql        : the SQL that was executed
    result     : the raw result string returned by the database

    Returns
    -------
    str  natural-language answer
    """
    if not result:
        return f"No results were returned. SQL: {sql}\n"

    prompt = (
        "You have run a SQL query to answer a user's question.\n"
        f"Question : {question}\n"
        f"SQL      : {sql or '(none)'}\n"
        f"Result   : {result}\n\n"
        "Give a concise, direct answer to the user's question based on the result above. "
        "Do not run any more queries."
    )
    resp = chat_model.invoke(prompt)
    return resp.content if hasattr(resp, "content") else str(resp)


def run_sqlcoder_with_summary(
    sqlcoder_model,
    db: SQLDatabase,
    question: str,
    chat_model=None,
    table_names: list | None = None,
    verbose: bool = True,
) -> dict:
    """
    Full pipeline: SQLCoder generates SQL → DB executes → chat LLM summarises.

    If chat_model is None, the "answer" key will contain the raw DB result.

    Parameters
    ----------
    sqlcoder_model : Ollama model wrapping SQLCoder (no tool-call support needed)
    db             : LangChain SQLDatabase wrapper
    question       : natural-language question
    chat_model     : (optional) separate chat LLM for final summarisation
    table_names    : tables to expose to SQLCoder; None → all tables
    verbose        : print progress

    Returns
    -------
    dict — same keys as run_sqlcoder(), plus:
        answer  str  natural-language answer (from chat_model or raw result)
    """
    out = run_sqlcoder(sqlcoder_model, db, question,
                       table_names=table_names, verbose=verbose)

    if chat_model and out["sql"] and out["result"]:
        if verbose:
            print(f"\n── Chat-model summary ──────────────────────────────")
        out["answer"] = summarise_result(chat_model, question, out["sql"], out["result"])
        if verbose:
            print(out["answer"])
    else:
        out["answer"] = out["result"] or out["error"] or "No answer produced."

    return out
