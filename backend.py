
import chromadb

# STEP 1: Create a local database
client = chromadb.PersistentClient(path="./my_database")

# STEP 2: Create a collection (similar to a SQL table)
collection = client.get_or_create_collection(
    name="python_learning"
)

# STEP 3: Insert documents
documents = [
    "Python is a programming language",
    "SQL is used to manage relational databases",
    "Pandas is used for data analysis",
    "Matplotlib is used for data visualization",
    "Machine learning uses algorithms to learn from data"
]

collection.add(
    ids=["1", "2", "3", "4", "5"],
    documents=documents
)

print("Documents added successfully!")

# STEP 4: Search using natural language
results = collection.query(
    query_texts=["Which library helps analyse data?"],
    n_results=2
)

# STEP 5: Display results
print("\nSearch Results:")

for result in results["documents"][0]:
    print(result)
