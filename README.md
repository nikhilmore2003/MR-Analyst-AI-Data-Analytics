# MR. Analyst 📊
### AI-Powered Data Analysis & Document Intelligence Assistant

MR. Analyst is an AI-powered application that enables users to interact with PDF documents and CSV datasets using natural language.

Built using Python, Streamlit, Groq API, ChromaDB, Pandas and DuckDB.

## Features

### PDF Document Intelligence (RAG)
- Upload PDF documents.
- Extract and split PDF text into chunks.
- Generate semantic embeddings using Sentence Transformers.
- Store embeddings in ChromaDB.
- Retrieve relevant information through semantic search.
- Generate context-aware answers using Groq LLM.
- Display relevant source pages.

### CSV Data Analytics
- Upload CSV datasets.
- Process data using Pandas.
- Automatically extract dataset schemas.
- Generate DuckDB SQL queries using Groq.
- Execute read-only SQL queries.
- Generate natural-language explanations and insights.

## Technology Stack

| Component | Technology |
|---|---|
| Frontend | Streamlit |
| Language | Python |
| LLM | Groq API |
| Vector Database | ChromaDB |
| Embeddings | Sentence Transformers |
| Text Processing | LangChain |
| PDF Processing | PyPDF |
| Data Processing | Pandas |
| SQL Engine | DuckDB |

## Architecture

**PDF Pipeline**

PDF Upload → Text Extraction → Chunking → Embeddings → ChromaDB → Semantic Retrieval → Groq LLM → Answer with Sources

**CSV Pipeline**

CSV Upload → Pandas → Schema Extraction → AI SQL Generation → DuckDB Execution → Groq LLM → Insights

## Installation

Clone the repository:

```bash
git clone https://github.com/nikhilmore2003/MR-Analyst-AI-Data-Analytics.git
```

Navigate into the folder:

```bash
cd MR-Analyst-AI-Data-Analytics
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create a `.env` file:

```env
GROQ_API_KEY=your_groq_api_key
```

Run the application:

```bash
streamlit run mr_analyst_app.py
```

## Future Enhancements

- Excel file support
- Advanced data visualizations
- Chat history persistence
- OCR support for scanned PDFs
- Downloadable analysis reports

## Author

**Nikhil More**

GitHub: [nikhilmore2003](https://github.com/nikhilmore2003)

LinkedIn: [Nikhil More](https://www.linkedin.com/in/nikhil-more-a43a2b259/)
