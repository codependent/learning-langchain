from langchain.chains.conversation.prompt import DEFAULT_TEMPLATE
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_postgres.vectorstores import PGVector
from langchain_core.documents import Document
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# PGVector DB connection
connection = "postgresql+psycopg://langchain:langchain@localhost:6024/langchain"



# Sample documents
docs = [
    Document(
        page_content="A bunch of scientists bring back dinosaurs and mayhem breaks loose",
        metadata={"year": 1993, "rating": 7.7, "genre": "science fiction"},
    ),
    Document(
        page_content="Leo DiCaprio gets lost in a dream within a dream within a dream within a ...",
        metadata={"year": 2010, "director": "Christopher Nolan", "rating": 8.2, "genre": "science fiction"},
    ),
    Document(
        page_content="A psychologist / detective gets lost in a series of dreams within dreams within dreams and Inception reused the idea",
        metadata={"year": 2006, "director": "Satoshi Kon", "rating": 8.6, "genre": "science fiction"},
    ),
    Document(
        page_content="A bunch of normal-sized women are supremely wholesome and some men pine after them",
        metadata={"year": 2019, "director": "Greta Gerwig", "rating": 8.3, "genre": "comedy"},
    ),
    Document(
        page_content="Some weird aliens fight for the freedom of the resistance against an evil galactic empire",
        metadata={"year": 1979, "director": "George Lucas", "rating": 9.2, "genre": "science fiction"},
    ),
    Document(
        page_content="Toys come alive and have a blast doing so",
        metadata={"year": 1995, "genre": "animated"},
    ),
    Document(
        page_content="Three men walk into the Zone, three men walk out of the Zone",
        metadata={
            "year": 1979,
            "director": "Andrei Tarkovsky",
            "genre": "thriller",
            "rating": 9.9,
        },
    ),
]

# Embeddings
embeddings_model = OllamaEmbeddings(model="nomic-embed-text")

# Vectorstore
vectorstore = PGVector.from_documents(docs, embeddings_model, connection=connection)

# Metadata schema
metadata_field_info = [
    AttributeInfo(name="genre", description="The genre of the movie", type="string or list[string]"),
    AttributeInfo(name="year", description="The year the movie was released", type="integer"),
    AttributeInfo(name="director", description="The name of the movie director", type="string"),
    AttributeInfo(name="rating", description="A 1-10 rating for the movie", type="float"),
]

# Description of contents
document_content_description = "Brief summary of a movie"

# Define Ollama LLM for querying metadata
llm = ChatOllama(model="llama3", temperature=0)

custom_prompt = PromptTemplate(
    template=DEFAULT_TEMPLATE
    .replace("Respond with a JSON object", "Only respond with a valid JSON object. Do NOT include explanations. Be careful to match all query conditions"),
    input_variables=["query", "schema", "examples"]
)


# Create the retriever
retriever = SelfQueryRetriever.from_llm(
    llm=llm,
    vectorstore=vectorstore,
    document_contents=document_content_description,
    metadata_field_info=metadata_field_info,
    prompt=custom_prompt,
    enable_limit=True,
)

query1 = "I want to watch a movie rated higher than 8.5"
query2 = "Find movies of science fiction genre with a rating greater than 8.5"
query3 = "I'd like to watch a comedy"

print("🔍 DEBUGGING: Inspecting structured query...")
structured_query1 = retriever.query_constructor.invoke(query1)
structured_query2 = retriever.query_constructor.invoke(query2)
structured_query3 = retriever.query_constructor.invoke(query3)
print("Structured Queries:")
print(structured_query1)
print(structured_query2)
print(structured_query3)

# Query examples
results_1 = retriever.invoke(query1)
results_2 = retriever.invoke(query2)
results_3 = retriever.invoke(query3)

print("Results 1:")
for r in results_1:
    print(r.page_content, "=>", r.metadata)

print("\nResults 2:")
for r in results_2:
    print(r.page_content, "=>", r.metadata)

print("\nResults 3:")
for r in results_3:
    print(r.page_content, "=>", r.metadata)
