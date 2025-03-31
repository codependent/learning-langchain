from langchain_ollama import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

texts = [
    "Hi there!",
    "Oh, hello!",
    "What's your name?",
    "My friends call me World",
    "Hello World!"
]

embeddings = model.encode(texts)

print(embeddings)

#model = OpenAIEmbeddings(model="text-embedding-3-small")
embeddings_model = OllamaEmbeddings(model="nomic-embed-text")
embeddings = model.embed_documents([
    "Hi there!",
    "Oh, hello!",
    "What's your name?",
    "My friends call me World",
    "Hello World!"
])

print(embeddings)
