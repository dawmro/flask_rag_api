from flask import Flask, request
from langchain_ollama import OllamaLLM


app = Flask(__name__)

cached_llm = OllamaLLM(model="llama3.2:3b")

# 
# print(response)


@app.route("/ai", methods=["POST"])
def aiPost():
    print("aiPost called")
    json_content = request.json
    query = json_content.get("query")

    print(f"query: {query}")
    response_answer = cached_llm.invoke(query)
    return response_answer

def start_app():
    app.run(host="0.0.0.0", port=8080, debug=True)



if __name__ == "__main__":
    start_app()