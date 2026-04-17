from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load the retriever model once when file is imported
# all-MiniLM-L6-v2 is small, fast and good at finding
# semantically similar text
retriever = SentenceTransformer('all-MiniLM-L6-v2')


def find_best_chunk(question, chunks):
    """
    Finds the most relevant chunk for a given question.
    
    How it works:
    1. Converts question into a vector (numbers = meaning)
    2. Converts all chunks into vectors
    3. Compares question vector vs all chunk vectors
    4. Returns the chunk with highest similarity score
    """
    if not chunks:
        return ""
    
    # Convert question and chunks to vectors
    question_vec = retriever.encode([question])
    chunk_vecs = retriever.encode(chunks)
    
    # Compare vectors using cosine similarity
    scores = cosine_similarity(question_vec, chunk_vecs)
    
    # Return the most relevant chunk
    best_index = scores.argmax()
    return chunks[best_index]