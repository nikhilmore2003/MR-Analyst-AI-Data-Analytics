import hashlib
import math
import os
import re
from collections import Counter

import duckdb
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from pypdf import PdfReader

load_dotenv()

st.set_page_config(page_title="MR. Analyst", page_icon="📊", layout="wide")
st.markdown("""
<style>
.block-container {padding-top: 1.8rem; padding-bottom: 2.5rem;}
[data-testid="stSidebar"] {border-right: 1px solid rgba(128,128,128,.2);}
</style>
""", unsafe_allow_html=True)

MODEL = "openai/gpt-oss-20b"
PREVIEW_ROWS = 10
RESULT_ROWS = 50

def client_for(key):
    return Groq(api_key=key)

def ask_groq(client, system, prompt, temperature=0.1):
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": prompt}],
        temperature=temperature,
    )
    return response.choices[0].message.content.strip()

def fingerprint(upload):
    return hashlib.sha256(upload.getvalue()).hexdigest()

def split_page(text, page, size=1100, overlap=180):
    text = re.sub(r"\s+", " ", text).strip()
    result, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        piece = text[start:end].strip()
        if piece:
            result.append({"page": page, "text": piece})
        if end == len(text):
            break
        start = max(end - overlap, start + 1)
    return result

def read_pdf(upload):
    reader = PdfReader(upload)
    chunks = []
    for number, page in enumerate(reader.pages, 1):
        chunks.extend(split_page(page.extract_text() or "", number))
    if not chunks:
        raise ValueError("No selectable text found. Scanned PDFs need OCR, which this app does not include.")
    return chunks

def terms(text):
    return re.findall(r"[a-zA-Z0-9_]+", text.lower())

def retrieve(question, chunks, top_k=5):
    query = set(terms(question))
    if not query:
        return []
    token_lists = [terms(c["text"]) for c in chunks]
    df = Counter()
    for words in token_lists:
        df.update(set(words))
    scored = []
    n = max(len(chunks), 1)
    for chunk, words in zip(chunks, token_lists):
        counts = Counter(words)
        score = sum(
            (1 + math.log(counts[t])) * math.log(1 + n / (1 + df[t]))
            for t in query if counts[t]
        )
        if score:
            scored.append((score, chunk))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [item for _, item in scored[:top_k]]

def pdf_answer(client, question, chunks):
    found = retrieve(question, chunks)
    if not found:
        return "I could not find relevant information in the uploaded PDF.", []
    context = "\n\n".join(f"[Page {c['page']}]\n{c['text']}" for c in found)
    system = ("Answer only from the supplied PDF context. If the answer is absent, say so. "
              "Cite supporting page numbers like [Page 3]. Never invent facts.")
    answer = ask_groq(client, system, f"PDF CONTEXT:\n{context}\n\nQUESTION:\n{question}", 0.2)
    return answer, sorted({c["page"] for c in found})

def schema_for(df):
    return "\n".join(f"- {col}: {dtype}" for col, dtype in df.dtypes.items())

def make_sql(client, question, df):
    system = """You are an expert DuckDB SQL analyst.
Return exactly one read-only SQL query and nothing else.
The table is uploaded_data. Use only columns in the supplied schema.
Only SELECT or WITH queries are allowed.
Never use INSERT, UPDATE, DELETE, DROP, CREATE, ALTER, COPY, ATTACH,
DETACH, INSTALL, LOAD, EXPORT, IMPORT, PRAGMA, CALL, or multiple statements.
Do not use Markdown fences."""
    prompt = (f"Table: uploaded_data\nSchema:\n{schema_for(df)}\n\n"
              f"Question: {question}\nReturn one DuckDB query. Limit row-level output to 100 rows.")
    sql = ask_groq(client, system, prompt, 0)
    sql = re.sub(r"^```(?:sql)?\s*|\s*```$", "", sql.strip(), flags=re.I).strip()
    return sql.rstrip(";").strip()

def check_sql(sql):
    if not sql or ";" in sql:
        raise ValueError("Empty query or multiple SQL statements are not allowed.")
    if not re.match(r"^(SELECT|WITH)\b", sql, re.I):
        raise ValueError("Only SELECT or WITH queries are allowed.")
    forbidden = r"\b(INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|COPY|ATTACH|DETACH|INSTALL|LOAD|EXPORT|IMPORT|PRAGMA|CALL|EXECUTE|TRUNCATE)\b"
    if re.search(forbidden, sql, re.I):
        raise ValueError("The generated query contains a blocked operation.")
    if not re.search(r"\buploaded_data\b", sql, re.I):
        raise ValueError("The query must use uploaded_data.")

def run_sql(df, sql):
    check_sql(sql)
    con = duckdb.connect(":memory:")
    try:
        con.register("uploaded_data", df)
        return con.execute(sql).fetchdf().head(500)
    finally:
        con.close()

def explain_sql(client, question, sql, result):
    system = ("Explain the computed SQL result accurately in plain language. "
              "Do not invent values or claim causation.")
    prompt = (f"Question: {question}\nSQL:\n{sql}\n\nComputed result:\n"
              f"{result.head(30).to_string(index=False)}\n\nExplain the result clearly.")
    return ask_groq(client, system, prompt, 0.2)

def init_state():
    defaults = {
        "loaded_hash": None, "kind": None, "csv": None, "chunks": [],
        "answer": None, "sql": None, "result": None, "pages": []
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_state()

with st.sidebar:
    st.title("👋 Hey User")
    st.subheader("Step 1 · Your details")
    st.text_input("Name", placeholder="Enter your name")
    st.text_input("Email", placeholder="Enter your email")
    st.divider()
    st.subheader("Step 2 · Groq API")
    key = st.text_input(
        "Groq API key", value=os.getenv("GROQ_API_KEY", ""),
        type="password", placeholder="Enter your Groq API key"
    )
    if key:
        st.success("Groq API key detected.")
    st.divider()
    st.subheader("Step 3 · Upload a file")
    upload = st.file_uploader("Choose one PDF or CSV file", type=["pdf", "csv"])
    if upload:
        st.caption(f"Selected: {upload.name} · {upload.size / 1024:.1f} KB")
        if st.button("Process file", type="primary", use_container_width=True):
            if not key:
                st.error("Enter your Groq API key first.")
            else:
                try:
                    with st.spinner("Processing file..."):
                        if upload.name.lower().endswith(".pdf"):
                            st.session_state.chunks = read_pdf(upload)
                            st.session_state.csv = None
                            st.session_state.kind = "pdf"
                        else:
                            data = pd.read_csv(upload)
                            if data.empty:
                                raise ValueError("The CSV has no data rows.")
                            st.session_state.csv = data
                            st.session_state.chunks = []
                            st.session_state.kind = "csv"
                        st.session_state.loaded_hash = fingerprint(upload)
                        st.session_state.answer = None
                        st.session_state.sql = None
                        st.session_state.result = None
                        st.session_state.pages = []
                    st.success("File processed successfully.")
                except Exception as exc:
                    st.error(f"File processing failed: {exc}")
    if st.session_state.loaded_hash:
        st.divider()
        st.subheader("Current file")
        st.success(f"{st.session_state.kind.upper()} ready")

st.title("📊 MR. Analyst")
st.subheader("Your AI-powered data analysis companion. Upload your file, ask questions and discover insights.")

if not key:
    st.info("Enter your Groq API key in the sidebar to begin.")
elif upload is not None and st.session_state.loaded_hash != fingerprint(upload):
    st.info("Your file is selected. Click **Process file** in the sidebar.")
elif st.session_state.kind == "csv" and st.session_state.csv is not None:
    df = st.session_state.csv
    st.header("📊 CSV Analysis · Pandas + SQL + Groq")
    a, b, c = st.columns(3)
    a.metric("Total rows", f"{len(df):,}")
    b.metric("Total columns", f"{len(df.columns):,}")
    c.metric("Missing cells", f"{int(df.isna().sum().sum()):,}")
    with st.expander("Dataset details", expanded=True):
        st.markdown("**Column names and data types**")
        st.table(pd.DataFrame({"Column": df.columns.astype(str),
                               "Data type": [str(x) for x in df.dtypes]}))
        st.markdown(f"**Dataset preview — first {PREVIEW_ROWS} rows**")
        st.table(df.head(PREVIEW_ROWS))
    st.subheader("Ask a question based on your CSV")
    question = st.text_input("Your question", placeholder="Example: What are total sales by region?")
    if st.button("Analyze data", type="primary"):
        if not question.strip():
            st.warning("Enter a question first.")
        else:
            try:
                ai = client_for(key)
                with st.spinner("Generating SQL and calculating the answer..."):
                    sql = make_sql(ai, question, df)
                    result = run_sql(df, sql)
                    explanation = explain_sql(ai, question, sql, result)
                st.session_state.answer = explanation
                st.session_state.sql = sql
                st.session_state.result = result
                st.session_state.pages = []
            except Exception as exc:
                st.error(f"Analysis failed: {exc}")
    if st.session_state.answer and st.session_state.sql:
        st.divider()
        st.subheader("✨ Your AI Insights")
        st.markdown(st.session_state.answer)
        st.markdown("**Generated SQL Query**")
        st.code(st.session_state.sql, language="sql")
        result = st.session_state.result
        st.markdown(f"**Rows returned: {len(result)}**")
        if not result.empty:
            st.table(result.head(RESULT_ROWS))
        else:
            st.info("The query returned no rows.")
elif st.session_state.kind == "pdf" and st.session_state.chunks:
    st.header("📄 PDF Analysis · RAG + Groq")
    st.caption(f"Retrieved index: {len(st.session_state.chunks)} text chunks")
    question = st.text_input("Ask a question about your PDF",
                             placeholder="Example: Summarize the main points.")
    if st.button("Ask MR. Analyst", type="primary"):
        if not question.strip():
            st.warning("Enter a question first.")
        else:
            try:
                with st.spinner("Searching the PDF and preparing an answer..."):
                    answer, pages = pdf_answer(client_for(key), question, st.session_state.chunks)
                st.session_state.answer = answer
                st.session_state.pages = pages
                st.session_state.sql = None
                st.session_state.result = None
            except Exception as exc:
                st.error(f"PDF question failed: {exc}")
    if st.session_state.answer and st.session_state.sql is None:
        st.subheader("✨ Answer")
        st.markdown(st.session_state.answer)
        if st.session_state.pages:
            st.caption("Retrieved pages: " + ", ".join(map(str, st.session_state.pages)))
else:
    st.info("Upload a PDF or CSV in the sidebar and click Process file.")

st.divider()
st.caption("MR. Analyst · PDF uses lightweight retrieval-augmented generation (RAG). "
           "CSV uses Pandas and read-only DuckDB SQL. Verify AI-generated answers before business-critical use.")
