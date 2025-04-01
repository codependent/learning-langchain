from typing import Literal
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain_core.runnables import RunnableLambda
import logging

logging.basicConfig(level=logging.DEBUG)

# Data model for function schema
class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""
    datasource: Literal["python_docs", "js_docs"] = Field(
        ...,
        description="Given a user question, choose which datasource would be most relevant for answering their question",
    )

# Convert the Pydantic model to OpenAI-style function
tools = [convert_to_openai_function(RouteQuery)]

# Use Llama 3.1 from Ollama with tools (function calling)
llm = ChatOllama(model="llama3.1", temperature=0).bind(tools=tools)

# Prompt template
system = """You are an expert at routing a user question to the appropriate data source. 
Based on the programming language the question is referring to, route it to the relevant data source."""
prompt = ChatPromptTemplate.from_messages(
    [("system", system), ("human", "{question}")]
)

# The user question
question = """Why doesn't the following code work: 
from langchain_core.prompts 
import ChatPromptTemplate 
prompt = ChatPromptTemplate.from_messages(["human", "speak in {language}"]) 
prompt.invoke("french") """

# Run prompt through the model
messages = prompt.invoke({"question": question})
response = llm.invoke(messages)

# DEBUG: print raw response
print("\n🔍 Raw response:")
print(response)

# Handle the tool call (manual step)
tool_call = response.tool_calls[0]
tool_args = tool_call['args']
route_result = RouteQuery(**tool_args)

print("\n📦 Routing to:", route_result)

# Example of chaining to another tool
def choose_route(result):
    if "python_docs" in result.datasource.lower():
        return "chain for python_docs"
    else:
        return "chain for js_docs"

# Run full logic chain
final_result = choose_route(route_result)
print("\n✅ Chosen route:", final_result)
