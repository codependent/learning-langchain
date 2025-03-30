from langchain_openai.chat_models import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

#model = ChatOpenAI()
model = ChatOllama(model="mistral")
prompt = [HumanMessage("What is the capital of France?")]

response = model.invoke(prompt)
print(response.content)
