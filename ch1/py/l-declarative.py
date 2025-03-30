from langchain_ollama import ChatOllama
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# the building blocks

template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        ("human", "{question}"),
    ]
)

#model = ChatOpenAI()
model = ChatOllama(model="gemma3:4b")

# combine them with the | operator

chatbot = template | model

# use it

response = chatbot.invoke({"question": "Which model providers offer LLMs?"})
print(response.content)

# streaming

for part in chatbot.stream({"question": "Which model providers offer LLMs?"}):
    print(part)
