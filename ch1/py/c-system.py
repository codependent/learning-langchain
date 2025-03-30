from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langchain_openai.chat_models import ChatOpenAI

#model = ChatOpenAI()
model = ChatOllama(model="gemma:latest")
system_msg = SystemMessage(
    "You are a helpful assistant that responds to questions with two sentences"
)
human_msg = HumanMessage("What is the capital of France?")

response = model.invoke([system_msg, human_msg])
print(response.content)
