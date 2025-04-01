from typing import Literal
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain_core.runnables import RunnableLambda
import logging

logging.basicConfig(level=logging.DEBUG)

# Data model for routing tool
class RouteToDatasource(BaseModel):
    datasource: Literal["python_docs", "js_docs"] = Field(
        ...,
        description="Choose which datasource is most relevant based on the programming language in the user's question",
    )

pydantic_openai_function = convert_to_openai_function(RouteToDatasource)

# Convert tool schema
tools = [pydantic_openai_function]

# LLM with tool binding
llm = ChatOllama(model="llama3.1", temperature=0).bind(tools=tools)

# Clear system prompt instructing use of the routing tool
system = """
You are a routing assistant. 
You must always use the tool `RouteToDatasource` to select the best documentation source.
Given a programming-related question, respond by calling the tool and specifying either 'python_docs' or 'js_docs' as the most relevant source.
Do not explain your choice — just call the tool with the right parameter.
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system),
    ("human", "{question}")
])

question = """Why doesn't the following code work: 
from langchain_core.prompts 
import ChatPromptTemplate 
prompt = ChatPromptTemplate.from_messages(["human", "speak in {language}"]) 
prompt.invoke("french")"""

# Invoke LLM with prompt
messages = prompt.invoke({"question": question})
response = llm.invoke(messages)

# DEBUG: Check raw response
print("\n🔍 Raw response:\n", response)

# Extract tool call
if response.tool_calls:
    tool_call = response.tool_calls[0]
    tool_args = tool_call['args']

    print("\n🛠 Tool called:", tool_call['name'])

    # ✅ Parse the tool output
    route_result = RouteToDatasource(**tool_args)

    print("\n📦 Routing to:", route_result)

    # Optional chain logic
    def choose_route(result):
        if "python_docs" in result.datasource.lower():
            return "chain for python_docs"
        else:
            return "chain for js_docs"

    final_result = choose_route(route_result)
    print("\n✅ Chosen route:", final_result)

else:
    print("⚠️ No tool call detected. Model returned raw content:")
    print(response.content)
    # optional fallback: try to parse manually if content looks like JSON


