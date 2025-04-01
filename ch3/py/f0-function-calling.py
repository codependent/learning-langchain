from langchain_core.utils.function_calling import convert_to_openai_function
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from typing import Literal

# Define your tool
class RouteQuery(BaseModel):
    datasource: Literal["python_docs", "js_docs"]

# Convert to OpenAI-style tool
tools = [convert_to_openai_function(RouteQuery)]

# Create tool-calling enabled model
llm = ChatOllama(model="llama3.1", temperature=0).bind(tools=tools)

# Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "Decide which docs are relevant."),
    ("human", "How do I use list comprehensions in Python?")
])

# Run
response = llm.invoke(prompt.invoke({}))

print("\nRaw tool call response:")
print(response)

tool_call = response.tool_calls[0]
print("Tool called:", tool_call['name'])
print("Arguments:", tool_call['args'])

# Optional: Call the actual Python function
def list_comprehension(condition, input_list):
    return [x for x in input_list if eval(condition)]

output = list_comprehension(**tool_call['args'])
print("✅ Function output:", output)