# Study Agent (PDF Q&A + Summary Generator)

A local agent built with LangGraph that helps a student work with a course PDF: it answers direct questions from the material, and generates saved Markdown study summaries on request. Built as part of AI/agent engineering prep.

## How it works

The agent has two tools, and decides which one to use based on the user's request:

1. ask_pdf(question) - answers a specific question using RAG (retrieval over the PDF, embedded with bge-m3, stored in a local Chroma DB). Used for questions like "What is the difference between verification and validation?"

2. save_summary(topic) - retrieves relevant content on a topic, writes a structured Markdown summary, and saves it to summaries/. Used for requests like "Make me a summary of testing strategies."

Both tools are strictly grounded in the PDF content - if the source material doesn't cover something well, the agent says so instead of inventing an answer.

The agent itself is built with LangGraph's create_react_agent (ReAct pattern: reason, act, observe, repeat), running entirely locally via Ollama.

## Requirements

- Python 3.11+
- Ollama (https://ollama.com) installed and running, with these models pulled:
  - ollama pull qwen2.5:7b
  - ollama pull bge-m3

## Setup

1. Create and activate a virtual environment:
   - python -m venv venv
   - venv\Scripts\activate (Windows)
2. Install dependencies:
   - pip install -r requirements.txt
3. Add your own PDF to the data/ folder (not included in this repo - see note below).
4. Build the local vector DB from the PDF:
   - python setup_rag.py

## Usage

Edit the question inside agent_v1.py, then run:

- python agent_v1.py

## Note on the data/ folder

This repo does not include any PDF, since course material is typically copyrighted 
and not meant for public redistribution. Add your own PDF to data/ and update the 
PDF_PATH variable in setup_rag.py if needed, then run setup_rag.py to build the 
vector database.

## Design notes / lessons learned

- Tool description quality matters as much as tool logic. The model chooses between tools based on their docstrings - vague descriptions lead to wrong tool selection.
- Models don't reliably relay raw tool output verbatim by default. In an early version, the agent invented a slightly different (uncropped) filename than the one the tool actually returned. Fixing this required an explicit system-level instruction: "quote tool results exactly, never guess or complete them."
- Retrieval breadth needs to match query breadth. A broad summary request (covering multiple sub-topics) needed a higher n_results than a single specific question - a fixed retrieval count doesn't fit every kind of query.
- Source material quality bounds summary quality. When the underlying slides only had one or two sentences on a sub-topic, the agent correctly reported that rather than padding the summary with invented detail.
- Deterministic file naming avoids duplicate outputs. Filenames are derived from a normalized (lowercased, stripped) version of the topic string, so near-identical requests overwrite the same file instead of creating near-duplicates.

## Known limitations

- File-name-based deduplication is approximate - meaningfully different phrasings of the same topic will still produce separate files.
- create_react_agent is deprecated as of LangGraph v1.0 in favor of langchain.agents.create_agent; this project will likely need a small migration in the future.

## Project structure

- data/ - place your own source PDF(s) here (gitignored)
- chroma_db/ - vector DB (gitignored, rebuild with setup_rag.py)
- summaries/ - generated study summaries
- tools.py - ask_pdf and save_summary tool definitions
- setup_rag.py - PDF to chunks to embeddings to Chroma
- agent_v1.py - single agent, two tools (LangGraph ReAct agent)
- requirements.txt
- README.md
