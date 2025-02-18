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

```Bash

ollama run deepseek-r1:7b  # Or your chosen model
```
* **Run the Chatbot:** Execute the Python script:

```Bash
python chatbot.py
```

## Data File (`data.txt`)

The `data.txt` file contains the data that the chatbot will use to answer questions. It should include the text you want the chatbot to "know." You can include a mix of regular text, Markdown, Python code, and JavaScript code.


## Chunking Strategies

The chatbot uses a combination of chunking methods:

### 1. **Character and Recursive Splitting**

These methods are used as a first pass to break down the text into smaller chunks:
- **Character Splitting** divides the text into fixed-size chunks.
- **Recursive Splitting** uses separators like newlines and spaces to keep related sentences and paragraphs together.

### 2. **Markdown/Python/JavaScript Splitters**

Specialized splitters are applied to handle structured content:
- **Markdown Splitter**: Parses Markdown syntax.
- **Python Splitter**: Splits Python code by functions and classes.
- **JavaScript Splitter**: Handles JavaScript code similarly.

### 3. **Semantic Chunking**

This is the most important step. It uses sentence embeddings and a similarity threshold to combine semantically related chunks. This ensures that chunks contain related information, even if they were initially separated by the other splitters. Semantic chunking significantly improves the context and relevance of the retrieved information.

## Vectorstore

The FAISS vectorstore is created from the chunks and saved to disk. This allows for efficient similarity search when answering questions. The vectorstore stores the embeddings of the chunks, enabling the chatbot to quickly find the most relevant chunks for a given query.

## Model

The chatbot uses the `deepseek-r1:7b Ollama` model (or a model you specify). You can change the `MODEL_NAME` variable in the `chatbot.py` file to use a different model.

