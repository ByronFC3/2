import os
import json
import gradio as gr
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA

# === Config ===
VECTOR_DIR = "grant_rag_faiss_store"
UPLOAD_DIR = "uploads"
os.makedirs(VECTOR_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

# === Load Vectorstore if exists ===
vectorstore = None
retriever = None
rag_chain = None

def load_vectorstore():
    global vectorstore, retriever, rag_chain
    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.load_local(VECTOR_DIR, embeddings)
        retriever = vectorstore.as_retriever()
        rag_chain = RetrievalQA.from_chain_type(
            llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0),
            retriever=retriever,
            return_source_documents=True
        )
        return True
    except Exception as e:
        print("Could not load vectorstore:", e)
        return False

load_vectorstore()

# === Upload + Index Function ===
def process_pdf(file):
    global vectorstore, retriever, rag_chain
    filepath = os.path.join(UPLOAD_DIR, file.name)
    with open(filepath, "wb") as f:
        f.write(file.read())

    loader = PyPDFLoader(filepath)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    if vectorstore:
        vectorstore.add_documents(chunks)
    else:
        vectorstore = FAISS.from_documents(chunks, embeddings)

    vectorstore.save_local(VECTOR_DIR)
    retriever = vectorstore.as_retriever()
    rag_chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0),
        retriever=retriever,
        return_source_documents=True
    )
    return "✅ PDF processed and vectorstore updated."

# === Chat Function ===
def ask_question(query):
    if not rag_chain:
        return "⚠️ Please upload a PDF first.", ""
    result = rag_chain({"query": query})
    answer = result["result"]
    sources = result.get("source_documents", [])
    source_text = "\n\n📚 **Sources:**"
    for i, doc in enumerate(sources[:3]):
        snippet = doc.page_content[:300].strip().replace("\n", " ")
        source_text += f"\n{i+1}. {snippet}..."
    return answer, source_text

# === Gradio UI ===
with gr.Blocks() as demo:
    gr.Markdown("# 📝 GA-AIM Grant PDF Q&A Assistant")

    with gr.Row():
        file_input = gr.File(label="Upload Grant PDF")
        upload_btn = gr.Button("Index PDF")
        upload_status = gr.Textbox(label="Upload Status")

    with gr.Row():
        question = gr.Textbox(label="Ask a Question")
        answer = gr.Textbox(label="Answer", lines=4)
        sources = gr.Textbox(label="Source Chunks", lines=6)

    upload_btn.click(process_pdf, inputs=file_input, outputs=upload_status)
    question.submit(ask_question, inputs=question, outputs=[answer, sources])

demo.launch()