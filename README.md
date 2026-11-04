# MR. Analyst

A Streamlit app for analysing one uploaded PDF or CSV using Groq.

## Features
- PDF Q&A using retrieval-augmented generation (RAG): PDF text extraction, chunking, lexical retrieval, and Groq answer generation with page references.
- CSV analysis using Pandas, Groq-generated read-only DuckDB SQL, and a Groq explanation of the computed result.
- Dataset dimensions, missing-cell count, column types, and a small preview.

## Project files
- `mr_analyst_app.py` — complete app
- `requirements.txt` — dependencies
- `.env.example` — example key configuration
- `.gitignore` — excludes secrets and local files

## Run locally
1. Use Python 3.11.
2. Install: `pip install -r requirements.txt`
3. Copy `.env.example` to `.env` and add your key:
   `GROQ_API_KEY=your_actual_key`
4. Run: `streamlit run mr_analyst_app.py`

You can also paste the Groq key in the app sidebar. Never commit a real API key or `.env` file.

## Deploy on Streamlit Community Cloud
1. Upload all project files to GitHub.
2. Set the main file path to `mr_analyst_app.py`.
3. In Advanced settings, choose Python 3.11 if available.
4. Add this in Streamlit Secrets:
   `GROQ_API_KEY = "your_actual_key"`
5. Deploy.

## Notes
- The configured Groq model is `openai/gpt-oss-20b`.
- Scanned/image-only PDFs are not OCR-processed.
- CSV queries are restricted to one read-only SELECT/WITH statement.
- PDF retrieval uses lightweight lexical ranking. This version intentionally removes ChromaDB, Sentence Transformers, PyTorch, and CUDA dependencies to make deployment smaller and simpler.
- The app uses `st.table` and `st.text_input` for previews and questions rather than interactive DataFrame and multiline text-area widgets.
