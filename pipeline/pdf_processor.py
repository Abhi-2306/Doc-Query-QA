import fitz  # PyMuPDF

def extract_text(pdf_path):
    """
    Opens a PDF file and extracts all text from every page.
    Returns one clean string of text.
    """
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text.strip()


def chunk_text(text, chunk_size=400, overlap=50):
    """
    Splits long text into smaller overlapping chunks.
    - chunk_size: number of words per chunk
    - overlap: number of words shared between chunks
      so no sentence is cut off mid-meaning
    """
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk:
            chunks.append(chunk)
    return chunks


def process_pdf(pdf_path):
    """
    Full pipeline — extract text then chunk it.
    This is the main function api.py will call.
    Returns list of chunks ready for the model.
    """
    text = extract_text(pdf_path)
    chunks = chunk_text(text)
    return chunks
