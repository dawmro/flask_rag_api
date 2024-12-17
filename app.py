from flask import Flask, request
from langchain_ollama import OllamaLLM
import os


app = Flask(__name__)

cached_llm = OllamaLLM(model="llama3.2:3b")



@app.route("/ai", methods=["POST"])
def aiPost():
    print("aiPost called")
    json_content = request.json
    query = json_content.get("query")

    print(f"query: {query}")
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

    print(f"filename: {file_name}")
    response = {"status": "Upload successfull", "filename": file_name}
    return response



def start_app():
    app.run(host="0.0.0.0", port=8080, debug=True)



if __name__ == "__main__":
    start_app()