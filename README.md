# HR Products price-list chatbot

A small local chatbot that answers questions about the **HR Products Price List 2026-27** (PDF). Ask for a product by
description or supplier code and it returns the supplier code, list price, group and page, with a short answer written by
a local LLM (Ollama). Everything runs on your machine.

## How it works
1. `ingest.py` reads the PDF with `pdftotext`, parses every table row (description, supplier/order code, quantities, list
   price), repairs wrapped descriptions, records the printed page number, and builds an embedding index in `data/`.
2. `search.py` ranks rows with a mix of exact code match, BM25 keyword search and sentence-embedding similarity.
3. `llm.py` talks to Ollama: it cleans up the query, writes a short answer from the top rows (the model never supplies
   prices or codes itself), and can tidy descriptions during ingest.
4. `app.py` is the Streamlit chat page: greeting handling, clarifying questions, store contact fallback, results table
   (top 10 + "show all"), and a Stop button. Without Ollama it falls back to plain search.

## Setup
Requirements: Python 3.10+, `pdftotext` (`brew install poppler`), and optionally [Ollama](https://ollama.com).

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# optional, for chat answers
brew install ollama
ollama serve                 # leave running
ollama pull qwen2.5:7b

.venv/bin/python ingest.py   # builds data/ from the PDF (re-run if the PDF changes)
.venv/bin/streamlit run app.py
```

Set `OLLAMA_MODEL` or `OLLAMA_URL` to use another model or host.

## Notes
- Prices are list prices (ex GST); for discounts contact HR Products Perth (details are in the app).
- The price-list PDF belongs to HR Products. It is available at
  https://www.hrproducts.com.au/product-category/price-list/
