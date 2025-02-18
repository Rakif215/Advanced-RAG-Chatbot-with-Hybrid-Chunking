# Advanced-RAG-Chatbot-with-Hybrid-Chunking

This project demonstrates an advanced Retrieval Augmented Generation (RAG) chatbot that utilizes a hybrid chunking strategy to improve information retrieval and response quality.

## Description

This chatbot combines several advanced chunking techniques to handle diverse content types (regular text, Markdown, Python code, JavaScript code) and ensure semantic coherence in the retrieved information. It uses Ollama as the Large Language Model (LLM) and FAISS for efficient vector storage and similarity search.

## Key Features

*   **Hybrid Chunking:** Combines character and recursive character splitting for initial text breakdown.
*   **Specialized Splitters:** Includes Markdown, Python, and JavaScript code splitters to handle structured content effectively.
*   **Semantic Chunking:** Employs sentence embeddings and a similarity threshold to group semantically related chunks, improving context and relevance.
*   **Ollama Integration:** Uses Ollama for text generation.
*   **FAISS Vectorstore:** Leverages FAISS for fast and efficient similarity search.

## Requirements

To run this chatbot, you will need the following Python libraries:

langchain
langchain-huggingface  # For embeddings
sentence-transformers
faiss-cpu  # or faiss-gpu if you have a GPU
ollama-python
scikit-learn
numpy
requests

You can install these using pip:

```bash
pip install -r requirements.txt  # Or install each package individually
```
# How to Run
 * **Start Ollama:** Run the Ollama server with the specified model:

'''Bash

ollama run deepseek-r1:7b  # Or your chosen model
'''
* **Run the Chatbot:** Execute the Python script:

'''Bash
python chatbot.py
'''
