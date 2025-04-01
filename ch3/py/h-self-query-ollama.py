from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_postgres.vectorstores import PGVector
from langchain_core.documents import Document
from langchain.prompts import PromptTemplate

# PGVector DB connection
connection = "postgresql+psycopg://langchain:langchain@localhost:6024/langchain"



# Sample documents
docs = [
    Document(
        page_content="A bunch of scientists bring back dinosaurs and mayhem breaks loose",
        metadata={"year": 1993, "rating": 8.7, "genre": "science fiction"},
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
        metadata={"year": 1995, "genre": "animated", "rating": 7.5},
    ),
    Document(
        page_content="Three men walk into the Zone, three men walk out of the Zone",
        metadata={"year": 1979, "director": "Andrei Tarkovsky", "genre": "thriller","rating": 9.9},
    ),
]

# Embeddings
embeddings_model = OllamaEmbeddings(model="nomic-embed-text")

# Vectorstore
vectorstore = PGVector.from_documents(docs, embeddings_model, connection=connection)

# Metadata schema
metadata_field_info = [
    AttributeInfo(
        name="genre",
        description="The genre of the movie. One of ['science fiction', 'comedy', 'drama', 'thriller', 'romance', 'action', 'animated']",
        type="string",
    ),
    AttributeInfo(
        name="year",
        description="The year the movie was released",
        type="integer",
    ),
    AttributeInfo(
        name="director",
        description="The name of the movie director",
        type="string",
    ),
    AttributeInfo(
        name="rating", description="A 1-10 rating for the movie", type="float"
    ),
]

# Description of contents
document_content_description = "Brief summary of a movie"

# Define Ollama LLM for querying metadata
llm = ChatOllama(model="gemma3:4b", temperature=0)

prompt_template_str = """
You are a text-to-structured-query converter. Convert natural language questions into structured queries.

Use the provided metadata schema and examples to guide your conversion.

Respond ONLY with a valid JSON object. Do not include any explanations.

Schema:
{schema}

Examples:
{examples}

User Query:
{query}

Structured Query:
"""

custom_prompt = PromptTemplate(
    template=prompt_template_str
    .replace("Respond with a JSON object", "Only respond with a valid JSON object. Do NOT include explanations. Be careful to match all query conditions"),
    input_variables=["query", "schema", "examples"]
)


# Create the retriever
retriever = SelfQueryRetriever.from_llm(
    llm=llm,
    vectorstore=vectorstore,
    document_contents=document_content_description,
    metadata_field_info=metadata_field_info,
    #prompt=custom_prompt,
    enable_limit=False,
    examples = [
        {
            "query": "I want to watch science fiction movies with a rating at least 5.5",
            "filter": 'and(gt("rating", 5.5), eq("genre", "science fiction"))'
        },
        {
            "query": "Find movies of science fiction genre with a rating greater than 8.5",
            "filter": 'and(gt("rating", 8.5), eq("genre", "science fiction"))'
        },
        {
            "query": "Find comedy movies",
            "filter": 'eq("genre", "comedy")'
        },
        {
            "query": "Show animated movies with a rating above 7",
            "filter": 'and(gt("rating", 7), eq("genre", "animated"))'
        },
        {
            "query": "Find science fiction movies with a rating greater than 8.5",
            "filter": 'and(gt("rating", 8.5), eq("genre", "science fiction"))'
        }
    ],
    search_kwargs={"k":4}
)

queries = [
    "I want to watch a movie rated higher than 8.5",
    "Find movies of science fiction genre with a rating greater than 8.5",
    "I'd like to watch a comedy",
    "Find science fiction movies with a rating greater than 8.5",
    "Show me animated movies above 7 rating",
    "I'd like to find movies released in 1980 or before"
]

# Structured query inspection
print("Structured Queries:")
structured_queries = [retriever.query_constructor.invoke(q) for q in queries]
for i, sq in enumerate(structured_queries, 1):
    print(f"Query {i}: {sq}")

# Query and results
for i, query in enumerate(queries, 1):
    print(f"\nResults {i} - Query {query}:")
    results = retriever.invoke(query)
    for r in results:
        print(r.page_content, "=>", r.metadata)