import os
import textwrap
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    MarkdownTextSplitter,
    PythonCodeTextSplitter,
    Language,
)
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import Ollama
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# --- Configuration ---
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
MODEL_NAME = "deepseek-r1:7b"  # 
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
EMBEDDING_MODEL = "all-mpnet-base-v2"  # 
DATA_FILE = "data.txt"
VECTORSTORE_PATH = "faiss_index"

# --- Chunking and Vectorstore Creation ---

def create_vectorstore(data_file, vectorstore_path):
    with open(data_file, "r", encoding="utf-8") as file:
        text = file.read()

    embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)

    all_chunks = []

    char_splitter = CharacterTextSplitter(chunk_size=CHUNK_SIZE * 2, chunk_overlap=CHUNK_OVERLAP)
    char_chunks = char_splitter.split_text(text)

    recursive_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    for chunk in char_chunks:
        all_chunks.extend(recursive_splitter.split_text(chunk))

    if "#" in text:
        md_splitter = MarkdownTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        md_chunks = md_splitter.split_text(text)
        all_chunks.extend(md_chunks)

    if "def " in text or "//" in text or "function" in text:
        try:
            py_splitter = PythonCodeTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
            py_chunks = py_splitter.split_text(text)
            all_chunks.extend(py_chunks)
        except:
            try:
                js_splitter = RecursiveCharacterTextSplitter.from_language(
                    language=Language.JS, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
                )
                js_chunks = js_splitter.split_text(text)
                all_chunks.extend(js_chunks)
            except Exception as e:
                print(f"Error splitting JavaScript: {e}")

    semantic_chunks = []
    current_semantic_chunk = ""
    for chunk in all_chunks:
        if not current_semantic_chunk:
            current_semantic_chunk = chunk
        else:
            embedding1 = embeddings.embed_query(chunk)
            embedding2 = embeddings.embed_query(current_semantic_chunk)

            if isinstance(embedding1, list):
                embedding1 = np.array(embedding1).reshape(1, -1)
            else:
                embedding1 = np.array([embedding1]).reshape(1, -1)

            if isinstance(embedding2, list):
                embedding2 = np.array(embedding2).reshape(1, -1)
            else:
                embedding2 = np.array([embedding2]).reshape(1, -1)

            similarity = cosine_similarity(embedding1, embedding2)[0][0]

            SIMILARITY_THRESHOLD = 0.5  # Adjust as needed
            if similarity >= SIMILARITY_THRESHOLD:
                current_semantic_chunk += " " + chunk
            else:
                semantic_chunks.append(current_semantic_chunk)
                current_semantic_chunk = chunk
    semantic_chunks.append(current_semantic_chunk)

    db = FAISS.from_texts(semantic_chunks, embeddings)
    db.save_local(vectorstore_path)
    return db


def load_vectorstore(vectorstore_path):
    embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)
    db = FAISS.load_local(vectorstore_path, embeddings, allow_dangerous_deserialization=True) 
    return db


# --- Chatbot Logic ---

def chatbot(query, db):
    llm = Ollama(base_url=OLLAMA_BASE_URL, model=MODEL_NAME)
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=db.as_retriever())

    result = qa_chain.run(query)

    wrapped_result = textwrap.fill(result, width=80)
    return wrapped_result


# --- Main Interaction ---

if __name__ == "__main__":
    data_file = DATA_FILE
    vectorstore_path = VECTORSTORE_PATH

    if not os.path.exists(vectorstore_path):
        print("Creating vectorstore (this may take a while)...")
        db = create_vectorstore(data_file, vectorstore_path)
    else:
        db = load_vectorstore(vectorstore_path)
        print("Vectorstore loaded.")

    while True:
        query = input("You: ")
        if query.lower() in ["exit", "quit"]:
            break

        response = chatbot(query, db)
        print("Chatbot:", response)