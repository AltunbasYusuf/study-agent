\# Study Agent (PDF Q\&A + Summary Generator)



A local agent built with LangGraph that helps a student work with a course PDF: it answers direct questions from the material, and generates saved Markdown study summaries on request. Built as part of AI/agent engineering prep.



\## How it works



The agent has two tools, and decides which one to use based on the user's request:



1\. \*\*`ask\_pdf(question)`\*\* — answers a specific question using RAG (retrieval over the PDF, embedded with `bge-m3`, stored in a local Chroma DB). Used for questions like \*"What is the difference between verification and validation?"\*



2\. \*\*`save\_summary(topic)`\*\* — retrieves relevant content on a topic, writes a structured Markdown summary, and saves it to `summaries/`. Used for requests like \*"Make me a summary of testing strategies."\*



Both tools are strictly grounded in the PDF content — if the source material doesn't cover something well, the agent says so instead of inventing an answer.



The agent itself is built with LangGraph's `create\_react\_agent` (ReAct pattern: reason → act → observe → repeat), running entirely locally via Ollama.



\## Requirements



\- Python 3.11+

\- Ollama (https://ollama.com) installed and running, with these models pulled:

&#x20; - `ollama pull qwen2.5:7b`

&#x20; - `ollama pull bge-m3`



\## Setup



1\. Create and activate a virtual environment:

&#x20;  - `python -m venv venv`

&#x20;  - `venv\\Scripts\\activate` (Windows)

2\. Install dependencies:

&#x20;  - `pip install -r requirements.txt`

3\. Build the local vector DB from the PDF in `data/`:

&#x20;  - `python setup\_rag.py`



\## Usage



Edit the question inside `agent\_v1.py`, then run:



\- `python agent\_v1.py`



\## Design notes / lessons learned



\- \*\*Tool description quality matters as much as tool logic.\*\* The model chooses between tools based on their docstrings — vague descriptions lead to wrong tool selection.

\- \*\*Models don't reliably relay raw tool output verbatim by default.\*\* In an early version, the agent invented a slightly different (uncropped) filename than the one the tool actually returned. Fixing this required an explicit system-level instruction: "quote tool results exactly, never guess or complete them."

\- \*\*Retrieval breadth needs to match query breadth.\*\* A broad summary request (covering multiple sub-topics) needed a higher `n\_results` than a single specific question — a fixed retrieval count doesn't fit every kind of query.

\- \*\*Source material quality bounds summary quality.\*\* When the underlying slides only had one or two sentences on a sub-topic, the agent correctly reported that rather than padding the summary with invented detail.

\- \*\*Deterministic file naming avoids duplicate outputs.\*\* Filenames are derived from a normalized (lowercased, stripped) version of the topic string, so near-identical requests overwrite the same file instead of creating near-duplicates.



\## Known limitations



\- File-name-based deduplication is approximate — meaningfully different phrasings of the same topic will still produce separate files.

\- `create\_react\_agent` is deprecated as of LangGraph v1.0 in favor of `langchain.agents.create\_agent`; this project will likely need a small migration in the future.



\## Project structure



\- `data/` — source PDF(s)

\- `chroma\_db/` — vector DB (gitignored, rebuild with `setup\_rag.py`)

\- `summaries/` — generated study summaries

\- `tools.py` — `ask\_pdf` and `save\_summary` tool definitions

\- `setup\_rag.py` — PDF to chunks to embeddings to Chroma

\- `agent\_v1.py` — single agent, two tools (LangGraph ReAct agent)

\- `requirements.txt`

\- `README.md`

