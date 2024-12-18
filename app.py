from datetime import datetime
from flask import Flask, request
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaLLM
import os



pdf_dir = "pdf/"
if not os.path.exists(pdf_dir):
    os.makedirs(pdf_dir) 
db_dir = "db/"
if not os.path.exists(db_dir):
    os.makedirs(db_dir) 


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

raw_prompt = PromptTemplate.from_template(
    """ 
    <s>[INST] You are a technical assistant good at searching docuemnts. If you do not have an answer from the provided information say so. [/INST] </s>
    [INST] {input}
           Context: {context}
           Answer:
    [/INST]
"""
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



@app.route("/ask_pdf", methods=["POST"])
def aiPDFPost():
    print("{showTime()} askPost called")
    json_content = request.json
    query = json_content.get("query")
    print(f"{showTime()} query: {query}")

    print(f"{showTime()} Loading vector store")
    vector_store = Chroma(persist_directory=db_dir, embedding_function=embedding)

    print(f"{showTime()} Creating chain")
    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k": 20,
            "score_threshold": 0.5
        },
    )

    document_chain = create_stuff_documents_chain(llm=cached_llm, prompt=raw_prompt)
    chain = create_retrieval_chain(retriever=retriever, combine_docs_chain=document_chain)
    result = chain.invoke({"input": query})
    print(f"{showTime()} {result}")

    sources = []
    for doc in result["context"]:
        sources.append(
            {"source": doc.metadata["source"], "page_content": doc.page_content}
        )

    response_answer = ({"answer": result["answer"],"sources": sources})
    return response_answer



@app.route("/pdf", methods=["POST"])
def pdfPost():
    file = request.files["file"]
    file_name = file.filename
    save_file = os.path.join(pdf_dir, file_name)
    file.save(save_file)
    print(f"{showTime()} filename: {file_name}")

    loader = PDFPlumberLoader(save_file)
    docs = loader.load_and_split()
    doc_len = len(docs)
    print(f"{showTime()} docs len: {doc_len}")

    chunks = text_splitter.split_documents(docs)
    chunks_len = len(chunks)
    print(f"{showTime()} chunks len: {chunks_len}")

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