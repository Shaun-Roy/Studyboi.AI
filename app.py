import streamlit as st
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from pinecone import Pinecone
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
import requests
from bs4 import BeautifulSoup

# Load environment variables
key = st.secrets["PINECONE_API_KEY"]
env = st.secrets["PINECONE_ENV"]
mistral_key = st.secrets["MISTRAL_API_KEY"]

# Initialize Pinecone
index_name = "pdfchat-index"
pc = Pinecone(api_key=key)
index = pc.Index(index_name)

# Extract text from PDFs
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text

# Extract text from URL
def get_text_from_url(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        paragraphs = soup.find_all('p')
        text = "\n\n".join([p.get_text() for p in paragraphs if p.get_text(strip=True)])
        return text
    except Exception as e:
        st.error(f"Failed to fetch URL content: {e}")
        return ""

# Chunk text
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

# Embed and upsert vectors to Pinecone
def get_vectorstore(text_chunks):
    embeddings = HuggingFaceInstructEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    embedded_chunks = embeddings.embed_documents(text_chunks)

    vectors = [
        {"id": str(i), "values": embedded_chunks[i], "metadata": {"text": text_chunks[i]}}
        for i in range(len(text_chunks))
    ]
    index.upsert(vectors=vectors)

# Retrieve chunks for query
def retrieve_chunks(query, k=5):
    embeddings = HuggingFaceInstructEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    query_vector = embeddings.embed_query(query)
    results = index.query(vector=query_vector, top_k=k, include_metadata=True)
    return results

# Generate answer with Mistral
def generate_answer(question, retrieved_chunks):
    context = "\n\n".join([match["metadata"]["text"] for match in retrieved_chunks["matches"]])

    prompt = f"""You are a helpful assistant. Use the following context to answer the user's question.

Context:
{context}

Question:
{question}

Answer:"""

    llm = ChatOpenAI(
        model="mistral-small",
        base_url="https://api.mistral.ai/v1",
        openai_api_key=mistral_key,
        temperature=0.3
    )

    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content

def main():
    st.set_page_config(page_title="Chat with your PDFs or URLs", page_icon=":books:")
    st.header("Chat with your PDFs or URLs :books:")

    # Sidebar inputs & processing button
    with st.sidebar:
        st.subheader("Upload & Index")
        url = st.text_input("Enter a webpage URL", key="url_input")
        pdf_docs = st.file_uploader("Upload your PDFs", accept_multiple_files=True)

        if st.button("Process"):
            with st.spinner("Processing and indexing data..."):
                raw_text = ""
                if url:
                    raw_text = get_text_from_url(url)
                elif pdf_docs:
                    raw_text = get_pdf_text(pdf_docs)
                else:
                    st.error("Please upload a PDF or enter a URL.")
                    return

                if raw_text:
                    text_chunks = get_text_chunks(raw_text)
                    get_vectorstore(text_chunks)
                    st.success("Vector store created and data indexed!")

    # Main area: Query input and display
    query = st.text_input("Ask a question about your documents:", key="query_input")

    if query:
        with st.spinner("Retrieving and generating answer..."):
            results = retrieve_chunks(query)
            if results["matches"]:
                answer = generate_answer(query, results)
                st.subheader("Answer:")
                st.write(answer)

                with st.expander("Show Retrieved Chunks"):
                    for match in results["matches"]:
                        st.write(match["metadata"]["text"])
            else:
                st.info("No relevant chunks found. Please upload/index documents first.")

if __name__ == "__main__":
    main()

