# PDF Analyzer by LAUVRE

An AI-powered PDF analysis application that allows users to ask questions about their documents and receive context-aware answers using Retrieval-Augmented Generation (RAG).

## Overview

PDF Analyzer by LAUVRE is a local AI application built to explore how Large Language Models (LLMs), text embeddings, and semantic search can work together to make document analysis more accessible.

Instead of manually searching through long PDF documents, users can upload a PDF and ask questions in natural language.

## Features

- **PDF text extraction** — Extracts text from PDF documents.
- **Text chunking with overlap** — Splits document text into overlapping chunks to preserve context.
- **Semantic search** — Retrieves relevant document sections using embeddings and cosine similarity.
- **AI-powered answers** — Generates answers using retrieved document context.
- **Source references** — Displays source text and page numbers to help users verify answers.
- **Local AI processing** — Uses locally hosted models through Ollama.
- **Interactive chat interface** — Built with Streamlit.

## Tech Stack

- Python
- Streamlit
- PyMuPDF
- Ollama
- Llama 3.2
- Nomic Embed Text
- Scikit-learn

## How It Works

1. Upload a PDF document.
2. Extract text from each page.
3. Split the extracted text into overlapping chunks.
4. Generate embeddings for each chunk.
5. Convert the user's question into an embedding.
6. Retrieve the most semantically similar chunks.
7. Send the retrieved context to the language model.
8. Display the generated answer with supporting source sections.

## Installation

### Prerequisites

- Python
- [Ollama](https://ollama.com/)
- Git

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd pdf-analyzer-by-lauvre
```

Replace `YOUR_GITHUB_REPOSITORY_URL` with the actual URL of this repository.

### 2. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 3. Download the required models

```bash
ollama pull llama3.2
ollama pull nomic-embed-text
```

### 4. Run the application

```bash
python -m streamlit run main.py
```

Open the local URL displayed in the terminal to access the application.

## Project Goals

This project was developed as a practical learning project in AI engineering, focusing on document processing, embeddings, semantic retrieval, and Retrieval-Augmented Generation.

## Current Limitations

- Answer quality depends on PDF text extraction and retrieval relevance.
- Scanned PDFs may require OCR before their text can be searched.
- AI model performance depends on the available local hardware.
- The application currently runs locally and requires Ollama and the specified models.

## Author

**LAUVRE**

Project: PDF Analyzer by LAUVRE