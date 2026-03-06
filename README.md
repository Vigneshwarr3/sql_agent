# SQL Agent (LangGraph + PostgreSQL)

This project contains a notebook-based SQL agent in `sql_agent.ipynb`.
It uses LangGraph + LangChain tools to:
- inspect your DB schema,
- generate SQL from natural language,
- run queries,
- return final answers.

## Prerequisites

- Python 3.10+
- PostgreSQL access (already configured in the notebook)
- Ollama (for local model mode)

## 1) Install Ollama

Install Ollama from: https://ollama.com/download

Verify installation:

```bash
ollama --version
```

## 2) Pull and run the required model

This notebook is configured to work with:

```bash
ollama pull llama3.1:8b
```

Start Ollama server (if not already running):

```bash
ollama serve
```

## 3) Install Python dependencies

From the project folder:

```bash
pip install -r requirements.txt
```

If `requirements.txt` is not present, run the install cells in `sql_agent.ipynb`.

## 4) Run the notebook

1. Open `sql_agent.ipynb` in VS Code.
2. Run cells from top to bottom.
3. Keep `MODEL_BACKEND = "sqlcoder"` and:
   - `OLLAMA_MODEL = "llama3.1:8b"`
   - `OLLAMA_BASE_URL = "http://localhost:11434"`
4. Run the example question cell in section **7 · Run the Agent**.

## Optional: Gemini backend

You can switch to Gemini by setting:

- `MODEL_BACKEND = "gemini"`
- adding your `GEMINI_API_KEY` in the config cell.

## Troubleshooting

- If you only see SQL text and no DB results, re-run all cells after kernel restart.
- Confirm Ollama is running and model is pulled:

```bash
ollama list
```

- Confirm DB connection cell prints available tables before running the agent.
