"""
The below example will use a SQLite connection with the Chinook database, which is a sample database that represents a digital media store. Follow these installation steps to create Chinook.db in the same directory as this notebook. You can also download and build the database via the command line:

```bash
curl -s https://raw.githubusercontent.com/lerocha/chinook-database/master/ChinookDatabase/DataSources/Chinook_Sqlite.sql | sqlite3 Chinook.db

```

Afterwards, place `Chinook.db` in the same directory where this code is running.

"""

from langchain_community.tools import QuerySQLDatabaseTool
from langchain_community.utilities import SQLDatabase
from langchain.chains import create_sql_query_chain
from langchain_core.runnables import RunnableLambda
from langchain_ollama import ChatOllama
# replace this with the connection details of your db
from langchain_openai import ChatOpenAI

db = SQLDatabase.from_uri("sqlite:///Chinook.db")
print(db.get_usable_table_names())
#llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
llm = ChatOllama(model="gemma3:4b", temperature=0)

def clean_sql_output(sql: str) -> str:
    # Remove triple backticks and optional language hints (e.g., ```sql)
    return sql.strip().removeprefix("```sql").removesuffix("```").strip("` \n")

sanitize_sql = RunnableLambda(lambda x: {"query": clean_sql_output(x)})

# convert question to sql query
write_query = create_sql_query_chain(llm, db)

# Execute SQL query
execute_query = QuerySQLDatabaseTool(db=db)

# combined chain = write_query | execute_query
combined_chain = write_query | sanitize_sql | execute_query

# run the chain
result = combined_chain.invoke({"question": "How many employees are there?"})

print(result)

result = combined_chain.invoke({"question": "Look for all customers with id between 1 and 10, return all customer information. Use correct column names from the Chinook schema"})

print(result)
