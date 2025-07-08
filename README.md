# 🤖 Studybot.AI

**Studybot.AI** is a smart assistant that lets you **chat with your PDFs or URLs**.

Upload documents or paste links, and then ask questions about the content. Studybot.AI uses **AI-powered retrieval** and **language generation** to give you accurate, contextual answers in real time.

---

## 🧠 Features

- 📄 Upload and process multiple PDF documents
- 🌐 Enter a webpage URL to fetch article content
- ✂️ Auto-chunks and embeds text
- 📌 Stores and searches content using Pinecone vector DB
- 🤖 Generates answers using the Mistral LLM API
- 💬 Clean Streamlit UI with question answering + sources

---

## 🎯 Ideal Use Cases

- Study from notes, slides, or textbooks (PDF)
- Research and summarize online articles or blogs (URL)
- Build personal knowledge bases
- Build an open-source RAG chatbot starter

---

## 🛠️ Tech Stack

| Component       | Technology                               |
|----------------|-------------------------------------------|
| Frontend        | Streamlit                                |
| PDF Parsing     | PyPDF2                                   |
| URL Scraping    | Requests + BeautifulSoup                 |
| Text Chunking   | LangChain CharacterTextSplitter          |
| Embedding Model | HuggingFace Instruct (MiniLM)            |
| Vector DB       | Pinecone                                  |
| LLM             | Mistral (OpenAI-compatible API)          |

---

## 🖼️ System Architecture

> _This diagram shows how Studybot.AI works behind the scenes._

![System Architecture](assets/architecture.png)

---

## 📦 Installation

1. **Clone the repository**

```bash
git clone https://github.com/yourusername/Studybot.AI.git
cd Studybot.AI
