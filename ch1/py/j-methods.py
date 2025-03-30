from langchain_ollama import ChatOllama
from langchain_openai.chat_models import ChatOpenAI

#model = ChatOpenAI(model="gpt-3.5-turbo")
model = ChatOllama(model="gemma:latest")

completion = model.invoke("Hi there!")
# Hi!
print(completion)

completions = model.batch(["Hi there!", "Bye!"])
# ['Hi!', 'See you!']
print(completion)

for token in model.stream("Bye!"):
    print(token)
    # Good
    # bye
    # !
