from langchain_ollama import ChatOllama
from langchain_openai.chat_models import ChatOpenAI

#model = ChatOpenAI(model="gpt-3.5-turbo")
model = ChatOllama(model="mistral")

response = model.invoke("The sky is")
print(response.content)
