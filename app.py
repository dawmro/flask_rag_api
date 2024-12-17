from langchain_ollama import OllamaLLM


llm = OllamaLLM(model="llama3.2:3b")

response = llm.invoke("Why is sky blue?")
print(response)