from datetime import datetime
from flask import Flask, request
from langchain_ollama import OllamaLLM
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_community.vectorstores import Chroma
import os


app = Flask(__name__)

cached_llm = OllamaLLM(model="llama3.2:3b")

embedding = FastEmbedEmbeddings()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1024, 
    chunk_overlap=128,
    separators=["\n\n", "\n", ". ", " ", ""],
    length_function=len,
    is_separator_regex=False
)



def showTime():
    return str("["+datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')+" UTC]")



@app.route("/ai", methods=["POST"])
def aiPost():
    print("{showTime()} aiPost called")
    json_content = request.json
    query = json_content.get("query")

    print(f"{showTime()} query: {query}")
    response_answer = cached_llm.invoke(query)
    return response_answer



@app.route("/pdf", methods=["POST"])
def pdfPost():
    file = request.files["file"]
    file_name = file.filename
    dir_name = "pdf/"
    if not os.path.exists(dir_name):
        os.makedirs(dir_name) 
    save_file = os.path.join(dir_name, file_name)
    file.save(save_file)
    print(f"{showTime()} filename: {file_name}")

    loader = PDFPlumberLoader(save_file)
    docs = loader.load_and_split()
    doc_len = len(docs)
    print(f"{showTime()} docs len: {doc_len}")

    chunks = text_splitter.split_documents(docs)
    chunks_len = len(chunks)
    print(f"{showTime()} chunks len: {chunks_len}")

    db_dir = "db/"
    if not os.path.exists(db_dir):
        os.makedirs(db_dir) 
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding,
        persist_directory=db_dir
    )
    # write vector store
    vector_store.persist()
    print(f"{showTime()} vector_store saved")

    response = {
        "status": "Upload successfull", 
        "filename": file_name, 
        "doc_len": doc_len, 
        "chunks":  chunks_len
    }
    return response



def start_app():
    app.run(host="0.0.0.0", port=8080, debug=True)



if __name__ == "__main__":
    start_app()