import json
import re
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_ollama import ChatOllama
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
    },
    # Ensure queries with 'science fiction' and 'rating' include the genre clause
    {
        "query": "Find science fiction movies above 8.0 rating",
        "filter": 'and(gt("rating", 8.0), eq("genre", "science fiction"))'
    },
    # Adjust for cases where the genre isn't clear or needs to be inferred
    {
        "query": "Find movies with a rating greater than 8.0",
        "filter": 'gt("rating", 8.0)'
    }
]


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
    AttributeInfo(name="genre", description="The genre of the movie", type="string or list[string]"),
    AttributeInfo(name="year", description="The year the movie was released", type="integer"),
    AttributeInfo(name="director", description="The name of the movie director", type="string"),
    AttributeInfo(name="rating", description="A 1-10 rating for the movie", type="float"),
]

# Prompt to instruct LLM to return clean JSON with simple key/value filters
prompt_template_str = """
You are a text-to-structured-query converter. Convert natural language questions into structured filters.

Use the metadata schema and examples to guide your conversion. Ensure that if a genre is mentioned or implied, it is included.

Respond ONLY with a valid JSON object using simple key:value or conditions. Avoid advanced logic (e.g., no function calls).

Schema:
{schema}

Examples:
{examples}

User Query:
{query}

Structured Query:
"""

prompt = PromptTemplate(
    template=prompt_template_str,
    input_variables=["query", "schema", "examples"]
)

llm = ChatOllama(model="gemma3:4b", temperature=0)
output_parser = StrOutputParser()

def clean_llm_response(response_text: str):
    """Remove code fences and parse the JSON."""
    response_text = response_text.strip()
    response_text = re.sub(r"^```json|```$", "", response_text, flags=re.MULTILINE).strip()
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        print(f"❌ Failed to parse LLM response: {response_text}")
        return {}

structured_query_chain = (
        prompt
        | llm
        | output_parser
        | RunnableLambda(clean_llm_response)
)

import re

def eval_filter_to_dict(filter_str: str) -> dict:
    """
    Converts a LangChain-style filter string into a dictionary suitable for PGVector's `filter` parameter.
    Example: and(gt("rating", 8.5), eq("genre", "science fiction"))
    """

    def parse_comparison(op: str, args: list) -> dict:
        field = args[0]
        value = args[1]
        if op == "eq":
            return {field: value}
        return {field: {f"${op}": value}}

    def tokenize(s: str):
        tokens = []
        current = ''
        depth = 0
        for c in s:
            if c == ',' and depth == 0:
                tokens.append(current.strip())
                current = ''
            else:
                if c == '(':
                    depth += 1
                elif c == ')':
                    depth -= 1
                current += c
        if current:
            tokens.append(current.strip())
        return tokens

    def parse(s: str):
        s = s.strip()
        if s.startswith("and(") or s.startswith("or("):
            op = "$and" if s.startswith("and") else "$or"
            inner = s[s.find('(') + 1:-1]
            sub_tokens = tokenize(inner)
            return {op: [parse(t) for t in sub_tokens]}
        else:
            match = re.match(r'(\w+)\("(.+?)",\s*(.+)\)', s)
            if not match:
                raise ValueError(f"Unsupported filter expression: {s}")
            op, field, value = match.groups()
            value = value.strip()
            if value.startswith('"') and value.endswith('"'):
                value = value[1:-1]
            else:
                try:
                    value = float(value) if '.' in value else int(value)
                except ValueError:
                    pass
            return parse_comparison(op, [field, value])

    return parse(filter_str)


# -------------------
# Apply to real queries
# -------------------

queries = [
    "Find science fiction movies with a rating greater than 8.5",
    "Show me animated movies above 7 rating"
]

for query in queries:
    print(f"\n🔍 Query: {query}")
    structured = structured_query_chain.invoke({
        "query": query,
        "schema": str(metadata_field_info),
        "examples": str(examples)
    })

    print(f"📦 Raw structured filter: {structured}")

    # If the filter is structured as expected, pass to PGVector
    if "filter" in structured:
        try:
            filter_dict = eval_filter_to_dict(structured["filter"])
            print(f"✅ Filter dict: {filter_dict}")
            results = vectorstore.similarity_search(query, k=4, filter=filter_dict)
            for doc in results:
                print(" -", doc.page_content, "=>", doc.metadata)
        except Exception as e:
            print("❌ Failed to apply filter:", e)
    else:
        print("⚠️ No filter returned.")
