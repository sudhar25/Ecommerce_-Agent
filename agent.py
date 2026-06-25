import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent

load_dotenv()

if not os.getenv("GROQ_API_KEY"):
    raise ValueError("API Key not found! Check your .env file.")

db = SQLDatabase.from_uri("sqlite:///olist.db")

llm = ChatGroq(
    model="llama3-70b-8bit",
    temperature=0
)

agent_executor = create_sql_agent(
    llm=llm,
    db=db,
    agent_type="openai-tools",
    verbose=True
)

if __name__ == "__main__":
    print("agent is initialized and ready to analyze.")
    
    question = "how many columns are in the orders table, and what are their names?"
    print(f"\nAsking Agent: {question}\n")
    response = agent_executor.invoke({"input": question})
    print(f"\nfinal Answer:\n{response['output']}")