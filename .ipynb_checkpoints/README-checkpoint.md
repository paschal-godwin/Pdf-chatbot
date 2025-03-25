# PDF Chatbot (RAG-based)

A simple chatbot that allows you to upload multiple PDFs and ask questions based on their content. It uses Retrieval-Augmented Generation (RAG) to extract relevant information and generate accurate responses using OpenAI's GPT-4o-mini.

---

## Features

- Upload and query multiple PDFs
- Retrieves relevant text chunks using vector similarity
- Uses OpenAI's GPT-4o-mini  for responses
- Built with Streamlit for easy interaction

---

## How It Works

1. Upload one or more PDF files through the app.
2. The content is extracted and chunked for processing.
3. Chunks are embedded using OpenAI embeddings and stored in a FAISS vector store.
4. When you ask a question:
   - Relevant chunks are retrieved based on your query
   - The chatbot uses those chunks as context to generate a response
5. Chat history is saved across interactions

---

## Technologies Used

- Python
- Streamlit
- LangChain
- Chromadb
- OpenAI GPT-4o-mini
- PyMuPDF

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/paschal-godwin/Pdf-chatbot.git
cd Pdf-chatbot

2. Install dependencies
pip install -r requirements.txt

3. Add your API key
Create a .env file and add your OpenAI key:
OPENAI_API_KEY=your_key_here

4. Run the app
streamlit run app.py

Future Improvements:

 Display source documents alongside answers

 Add support for DOCX, TXT, and other file types



