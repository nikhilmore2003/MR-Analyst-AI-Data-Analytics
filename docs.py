
# Import PdfReader from pypdf
from pypdf import PdfReader

# STEP 1: Open the PDF
reader = PdfReader("powerbi_dax_notes.pdf")

# STEP 2: Find total number of pages
print("Total Pages:", len(reader.pages))

# STEP 3: Extract text from all pages
full_text = ""

for page_number, page in enumerate(reader.pages):

    # Extract text from current page
    text = page.extract_text()

    # Add extracted text to full_text
    if text:
        full_text += text + "\n"

    print(f"Page {page_number + 1} extracted!")

# STEP 4: Display extracted text
print("\n========== EXTRACTED TEXT ==========\n")

print(full_text)

# STEP 5: Check total characters
print("\nTotal Characters:", len(full_text))
