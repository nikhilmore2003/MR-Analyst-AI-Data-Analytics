
# Import the text splitter
from langchain_text_splitters import RecursiveCharacterTextSplitter

# STEP 1: Create sample text
text = """
Python is a popular programming language.

Pandas is a Python library used for data analysis.
It helps us clean, manipulate and analyse datasets.

SQL is used to communicate with relational databases.
We use SQL to retrieve, update and manage data.

Matplotlib is a Python library used for data visualization.
It helps us create charts, graphs and plots.
"""

# STEP 2: Create a text splitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20
)

# STEP 3: Split the text
chunks = splitter.split_text(text)

# STEP 4: Print the chunks
print("Total Chunks:", len(chunks))

for i, chunk in enumerate(chunks):
    print(f"\n--- Chunk {i+1} ---")
    print(chunk)
    print("Characters:", len(chunk))
