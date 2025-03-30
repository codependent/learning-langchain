from langchain_ollama import ChatOllama
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.prompts import PromptTemplate

# both `template` and `model` can be reused many times

template = PromptTemplate.from_template("""Answer the question based on the context below. If the question cannot be answered using the information provided, answer with "I don't know".

Context: {context}

Question: {question}

Answer: """)

#model = ChatOpenAI(model="gpt-3.5-turbo")
model = ChatOllama(model="gemma3:4b")

# `prompt` and `completion` are the results of using template and model once

prompt = template.invoke(
    {
        "context": "The most recent advancements in NLP are being driven by Large Language Models (LLMs). These models outperform their smaller counterparts and have become invaluable for developers who are creating applications with NLP capabilities. Developers can tap into these models through Hugging Face's `transformers` library, or by utilizing OpenAI and Cohere's offerings through the `openai` and `cohere` libraries, respectively.",
        "question": "Which model providers offer LLMs?",
    }
)

response = model.invoke(prompt)
print(response)
