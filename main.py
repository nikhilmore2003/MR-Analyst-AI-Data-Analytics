import hashlib
import os
from io import BytesIO

import chromadb
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader


# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(
    page_title="Mr. Analyst",
    page_icon="📊",
    layout="wide"
)

load_dotenv()

MODEL_NAME = "openai/gpt-oss-20b"
CHROMA_PATH = "./my_database"


# ---------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------
def get_groq_client():
    """Create a Groq client using the key stored in the .env file."""
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY was not found. Add it to your .env file."
        )

    return Groq(api_key=api_key)


def get_collection(collection_name):
    """Get or create a persistent ChromaDB collection."""
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

    return chroma_client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )


def make_collection_name(file_bytes, prefix):
    """Create a stable collection name for the uploaded file."""
    file_hash = hashlib.md5(file_bytes).hexdigest()[:12]
    return f"{prefix}_{file_hash}"


def extract_pdf_text(pdf_bytes):
    """Extract selectable text from every page of an uploaded PDF."""
    reader = PdfReader(BytesIO(pdf_bytes))

    page_texts = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""

        if text.strip():
            page_texts.append(
                f"[Page {page_number}]\n{text.strip()}"
            )

    return "\n\n".join(page_texts), len(reader.pages)


def process_pdf(uploaded_file):
    """Extract, split and store PDF text chunks in ChromaDB."""
    pdf_bytes = uploaded_file.getvalue()
    full_text, total_pages = extract_pdf_text(pdf_bytes)

    if not full_text.strip():
        raise ValueError(
            "No selectable text was found in this PDF. "
            "It may be a scanned PDF and require OCR."
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )

    chunks = splitter.split_text(full_text)

    if not chunks:
        raise ValueError("No text chunks could be created from this PDF.")

    collection_name = make_collection_name(pdf_bytes, "pdf")
    collection = get_collection(collection_name)

    # Add chunks only if this file has not already been indexed.
    if collection.count() == 0:
        collection.add(
            ids=[f"chunk_{i}" for i in range(len(chunks))],
            documents=chunks,
            metadatas=[
                {
                    "source": uploaded_file.name,
                    "chunk_number": i + 1
                }
                for i in range(len(chunks))
            ]
        )

    return {
        "type": "pdf",
        "name": uploaded_file.name,
        "collection_name": collection_name,
        "total_pages": total_pages,
        "total_chunks": len(chunks),
        "full_text": full_text
    }


def dataframe_to_documents(df, rows_per_chunk=20):
    """Convert CSV rows into text documents for semantic retrieval."""
    documents = []
    metadatas = []

    for start in range(0, len(df), rows_per_chunk):
        end = min(start + rows_per_chunk, len(df))
        section = df.iloc[start:end]

        text = (
            f"CSV rows {start + 1} to {end}\n"
            f"Columns: {', '.join(map(str, df.columns))}\n"
            f"{section.to_string(index=False)}"
        )

        documents.append(text)
        metadatas.append({
            "start_row": start + 1,
            "end_row": end
        })

    return documents, metadatas


def process_csv(uploaded_file):
    """Load CSV with pandas and index row chunks in ChromaDB."""
    csv_bytes = uploaded_file.getvalue()

    try:
        df = pd.read_csv(BytesIO(csv_bytes))
    except UnicodeDecodeError:
        df = pd.read_csv(BytesIO(csv_bytes), encoding="latin-1")

    if df.empty:
        raise ValueError("The uploaded CSV has no data rows.")

    documents, metadatas = dataframe_to_documents(df)

    collection_name = make_collection_name(csv_bytes, "csv")
    collection = get_collection(collection_name)

    if collection.count() == 0 and documents:
        collection.add(
            ids=[f"row_chunk_{i}" for i in range(len(documents))],
            documents=documents,
            metadatas=metadatas
        )

    return {
        "type": "csv",
        "name": uploaded_file.name,
        "collection_name": collection_name,
        "dataframe": df,
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "total_chunks": len(documents)
    }


def retrieve_context(collection_name, question, number_of_results=4):
    """Retrieve the most relevant chunks from ChromaDB."""
    collection = get_collection(collection_name)
    total_items = collection.count()

    if total_items == 0:
        return []

    results = collection.query(
        query_texts=[question],
        n_results=min(number_of_results, total_items)
    )

    return results.get("documents", [[]])[0]


def ask_groq(question, context, data_type):
    """Send retrieved context and the user's question to Groq."""
    client = get_groq_client()

    if data_type == "pdf":
        system_prompt = """
You are Mr. Analyst, a helpful document question-answering assistant.
Answer using only the supplied PDF context.
If the answer is not present in the context, clearly say that you could
not find the answer in the uploaded PDF.
Do not invent facts. Keep the answer clear and structured.
Mention page numbers when they are available in the context.
"""
    else:
        system_prompt = """
You are Mr. Analyst, a helpful CSV data analysis assistant.
Use the supplied dataset information and retrieved CSV rows to answer.
Be clear about whether a result is based on the full dataset, a summary,
or only the retrieved rows. Do not invent calculations or claim that a
sample represents the entire dataset.
When useful, explain which columns support your answer.
"""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    f"USER QUESTION:\n{question}\n\n"
                    f"AVAILABLE DATA / CONTEXT:\n{context}"
                )
            }
        ],
        temperature=0.2
    )

    return response.choices[0].message.content


def build_csv_context(df, question, collection_name):
    """Combine pandas dataset information with retrieved CSV chunks."""
    retrieved_chunks = retrieve_context(
        collection_name,
        question,
        number_of_results=1
    )

    column_details = "\n".join(
        [f"- {column}: {dtype}" for column, dtype in df.dtypes.items()]
    )

    null_counts = df.isnull().sum().to_string()
    sample_rows = df.head(8).to_string(index=False)

    # Useful overview. Some describe() values may be non-numeric or missing.
    try:
        summary = df.describe(include="all").fillna("").to_string()
    except Exception:
        summary = "Could not generate a statistical summary."

    context_parts = [
        f"Dataset shape: {df.shape[0]} rows and {df.shape[1]} columns",
        f"Column names and data types:\n{column_details}",
        f"Missing values per column:\n{null_counts}",
        f"Statistical summary:\n{summary}",
        f"First 8 rows:\n{sample_rows}",
        "Semantically retrieved CSV row chunks:\n"
        + ("\n\n".join(retrieved_chunks) if retrieved_chunks else "No matching rows found.")
    ]

    # Include the complete CSV when it is small enough to fit comfortably.
    full_csv_text = df.to_csv(index=False)

    if len(full_csv_text) <= 12000:
        context_parts.append(
            "Complete CSV data (small dataset):\n" + full_csv_text
        )
    else:
        context_parts.append(
            "Note: This is a large dataset. The complete CSV was not sent "
            "to the LLM. Use the summary and retrieved row chunks; do not "
            "claim exact full-dataset calculations unless supported."
        )

    return "\n\n".join(context_parts)


# ---------------------------------------------------------
# SESSION STATE
# ---------------------------------------------------------
if "user_name" not in st.session_state:
    st.session_state.user_name = ""

if "user_email" not in st.session_state:
    st.session_state.user_email = ""

if "file_data" not in st.session_state:
    st.session_state.file_data = None

if "answer" not in st.session_state:
    st.session_state.answer = ""

if "processed_file_signature" not in st.session_state:
    st.session_state.processed_file_signature = None


# ---------------------------------------------------------
# SIDEBAR: USER DETAILS AND FILE UPLOAD
# ---------------------------------------------------------
st.sidebar.title("Hey User 👋")
st.sidebar.subheader("Step 1: Your details")

with st.sidebar.form("user_details_form"):
    name = st.text_input(
        "Name",
        value=st.session_state.user_name,
        placeholder="Enter your name"
    )

    email = st.text_input(
        "Email",
        value=st.session_state.user_email,
        placeholder="Enter your email"
    )

    submitted = st.form_submit_button("Save details")

    if submitted:
        st.session_state.user_name = name.strip()
        st.session_state.user_email = email.strip()

        if not st.session_state.user_name or not st.session_state.user_email:
            st.sidebar.warning("Please enter both your name and email.")
        else:
            st.sidebar.success("Your details have been saved.")


st.sidebar.divider()
st.sidebar.subheader("Step 2: Upload your data")

uploaded_file = st.sidebar.file_uploader(
    "Upload a PDF or CSV file",
    type=["pdf", "csv"],
    help="PDF files are processed using RAG. CSV files are processed using pandas and semantic retrieval."
)

if uploaded_file is not None:
    file_signature = hashlib.md5(uploaded_file.getvalue()).hexdigest()

    if st.session_state.processed_file_signature != file_signature:
        st.session_state.file_data = None
        st.session_state.answer = ""

    if st.sidebar.button("Process file", use_container_width=True):
        try:
            with st.spinner("Processing your file..."):
                if uploaded_file.name.lower().endswith(".pdf"):
                    file_info = process_pdf(uploaded_file)
                elif uploaded_file.name.lower().endswith(".csv"):
                    file_info = process_csv(uploaded_file)
                else:
                    raise ValueError("Only PDF and CSV files are supported.")

                st.session_state.file_data = file_info
                st.session_state.processed_file_signature = file_signature
                st.session_state.answer = ""

            st.sidebar.success("File processed successfully!")

        except Exception as error:
            st.sidebar.error(f"Could not process file: {error}")


# ---------------------------------------------------------
# MAIN SCREEN
# ---------------------------------------------------------
st.title("Mr. Analyst")
st.header("Your personal data analysis assistant.")
st.write(
    "Upload a PDF or CSV file, ask a question, and let Mr. Analyst "
    "find relevant information using AI."
)

if st.session_state.user_name:
    st.caption(f"Welcome, {st.session_state.user_name}!")

file_data = st.session_state.file_data

if file_data:
    st.success(f"Currently loaded: {file_data['name']}")

    if file_data["type"] == "pdf":
        col1, col2 = st.columns(2)
        col1.metric("PDF pages", file_data["total_pages"])
        col2.metric("Text chunks indexed", file_data["total_chunks"])

    elif file_data["type"] == "csv":
        df = file_data["dataframe"]

        col1, col2, col3 = st.columns(3)
        col1.metric("Total rows", file_data["total_rows"])
        col2.metric("Total columns", file_data["total_columns"])
        col3.metric("Chunks indexed", file_data["total_chunks"])

        with st.expander("Preview uploaded CSV"):
            st.dataframe(df.head(10), use_container_width=True)

        with st.expander("CSV column information"):
            column_info = pd.DataFrame({
                "Column": df.columns,
                "Data type": [str(dtype) for dtype in df.dtypes],
                "Missing values": df.isnull().sum().values
            })
            st.dataframe(column_info, use_container_width=True)

    st.divider()
    st.subheader("Step 3: Ask a question")

    question = st.text_input(
        "Ask something about your uploaded file:",
        placeholder="Example: What are the key points in this document?"
    )

    if st.button("Get Answer", type="primary", use_container_width=True):
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            try:
                with st.spinner("Finding relevant information and generating your answer..."):
                    if file_data["type"] == "pdf":
                        relevant_chunks = retrieve_context(
                            file_data["collection_name"],
                            question,
                            number_of_results=4
                        )

                        context = "\n\n".join(relevant_chunks)

                        if not context.strip():
                            st.session_state.answer = (
                                "I could not find relevant information in the PDF."
                            )
                        else:
                            st.session_state.answer = ask_groq(
                                question,
                                context,
                                "pdf"
                            )

                    elif file_data["type"] == "csv":
                        df = file_data["dataframe"]

                        context = build_csv_context(
                            df,
                            question,
                            file_data["collection_name"]
                        )

                        st.session_state.answer = ask_groq(
                            question,
                            context,
                            "csv"
                        )

            except Exception as error:
                st.error(f"Something went wrong: {error}")

    st.subheader("Step 4: Your answer")

    if st.session_state.answer:
        st.markdown(st.session_state.answer)

else:
    st.info(
        "Start by entering your details and uploading a PDF or CSV file "
        "from the sidebar. Then click **Process file**."
    )

st.sidebar.divider()
st.sidebar.caption("Supported formats: PDF and CSV")
st.sidebar.caption(f"LLM: {MODEL_NAME}")
