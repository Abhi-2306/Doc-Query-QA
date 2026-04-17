# Stores all Q&A turns for the current session
chat_history = []


def build_context(question, history, chunk):
    """
    Builds the full context to send to the model.
    
    Combines:
    - Last 3 conversation turns (so follow-up questions work)
    - Most relevant document chunk (from retriever)
    
    Example:
    Q: What is the revenue? A: $5.2 million
    Q: Who reported it?
    → model sees previous Q&A + document chunk
    → understands "it" = revenue
    → answers correctly
    """
    history_text = ""
    for q, a in history[-3:]:  # only last 3 turns
        history_text += f"Q: {q} A: {a} "
    
    full_context = history_text + chunk
    return full_context


def save_to_history(question, answer):
    """
    Saves each Q&A turn to history.
    Called after every question is answered.
    """
    chat_history.append((question, answer))


def get_history():
    """
    Returns current chat history.
    """
    return chat_history


def clear_history():
    """
    Clears chat history.
    Called when a new PDF is uploaded
    so old conversation doesn't mix with new document.
    """
    chat_history.clear()