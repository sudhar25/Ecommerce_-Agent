import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from visualizer import plot_monthly_sales, plot_hourly_peaks
from langchain.tools import tool
from ml_cluster import generate_rfm_cluster
load_dotenv()

@tool
def generate_monthly_sales_chart(year: int) -> str:
    """useful when the user ask to visualize or chat"""
    path = plot_monthly_sales(year)
    return f"chart generated successfully and saved to local file: {path}"

@tool
def generate_hourly_peak_chart() -> str:
    """useful when the user ask for hourly purchase"""
    path = plot_hourly_peaks()
    return f"hourly peak chart generated successfully and saved to local file: {path}"

@tool
def segment_customers_rfm() -> str:
    """Useful when the user asks to segment customers, find behavioral patterns, group high-value customers, or run RFM/clustering algorithms."""
    path = generate_rfm_cluster(n_clusters=4)
    return f"Machine learning clustering applied successfully. Scatter plot saved to local file: {path}"
    
if not os.getenv("GROQ_API_KEY"):
    raise ValueError("API Key not found! Check your .env file.")

db = SQLDatabase.from_uri("sqlite:///olist.db")

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0
)

extra_tools = [generate_monthly_sales_chart, generate_hourly_peak_chart, segment_customers_rfm ]
agent_executor = create_sql_agent(
    llm=llm,
    db=db,
    agent_type="openai-tools",
    extra_tools=extra_tools,
    verbose=True
)

if __name__ == "__main__":
    # Test the Agent's Machine Learning Routing
    question = "Can you segment our customers based on their behavior using machine learning and show me a plot of it?"
    response = agent_executor.invoke({"input": question})
    print(response['output'])