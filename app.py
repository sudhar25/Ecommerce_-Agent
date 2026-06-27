import streamlit as st
import os
import sqlite3
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain.tools import tool

# Import our custom engineering modules
from visualizer import plot_monthly_sales, plot_hourly_peaks
from ml_cluster import generate_rfm_cluster

load_dotenv()

# 1. Configure the Page Layout
st.set_page_config(page_title="E-Commerce AI Agent Platform", page_icon="", layout="wide", initial_sidebar_state="expanded")

# Inject Custom CSS for Modern UI
st.markdown("""
<style>
    /* Main Background & Fonts */
    .stApp {
        background-color: #0e1117;
        font-family: 'Inter', sans-serif;
    }
    
    /* Ensure all text is white and aligned */
    .stApp p, .stApp span, .stApp div, .stApp label, .stApp li {
        color: #ffffff !important;
    }
    
    /* Sleek Title Styling */
    h1 {
        font-weight: 800;
        background: -webkit-linear-gradient(45deg, #FF4B2B, #FF416C);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding-bottom: 10px;
    }
    
    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #1a1c23;
        border-radius: 8px 8px 0px 0px;
        padding: 10px 20px;
        color: #ffffff;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #1f2229 0%, #262932 100%);
        border-bottom: 2px solid #FF416C;
    }
    
    /* Chat Interface bubbles */
    [data-testid="stChatMessage"] {
        background-color: #1a1c23;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 10px;
        border: 1px solid #2d3748;
    }
    
    /* Button Aesthetics */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(0,0,0,0.2);
        width: 100%;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.3);
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    
    /* Subheaders */
    h2, h3 {
        color: #e2e8f0;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

st.title("AI-Powered E-Commerce Analytics & ML Platform")
st.markdown("*Interact with raw database tables using natural language, execute automated visualizations, and run machine learning models.*")

# 2. Secure Database Safe-Fail Mechanism for Cloud Deployment
@st.cache_resource
def initialize_database():
    db_path = "olist.db"
    # If running in the cloud without the massive local file, create a lightweight fallback schema
    if not os.path.exists(db_path):
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS orders (
                order_id TEXT, customer_id TEXT, order_status TEXT, order_purchase_timestamp TEXT
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS order_payments (
                order_id TEXT, payment_value REAL
            )
        """)
        # Insert a sample deployment record
        cursor.execute("INSERT INTO orders VALUES ('prod_1', 'cust_1', 'delivered', '2017-05-15 12:00:00')")
        cursor.execute("INSERT INTO order_payments VALUES ('prod_1', 150.50)")
        conn.commit()
        conn.close()
    return SQLDatabase.from_uri(f"sqlite:///{db_path}")

db = initialize_database()

# 3. Register the Custom Agent Tools
@tool
def generate_monthly_sales_chart(year: int) -> str:
    """Useful when the user asks to visualize or chart sales/revenue trends over months for a specific year."""
    path = plot_monthly_sales(year)
    return f"Chart generated successfully and saved to local file: {path}"

@tool
def generate_hourly_peak_chart() -> str:
    """Useful when the user asks to see hourly purchase distributions, peak shopping times, or buying trends by hour."""
    path = plot_hourly_peaks()
    return f"Hourly peak chart generated successfully and saved to local file: {path}"

@tool
def segment_customers_rfm() -> str:
    """Useful when the user asks to segment customers, find behavioral patterns, group high-value customers, or run RFM clustering algorithms."""
    path = generate_rfm_cluster(n_clusters=4)
    return f"Machine learning clustering applied successfully. Scatter plot saved to local file: {path}"

# 4. Initialize the AI Agent Engine
if os.getenv("GROQ_API_KEY"):
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    extra_tools = [generate_monthly_sales_chart, generate_hourly_peak_chart, segment_customers_rfm]
    agent_executor = create_sql_agent(
        llm=llm, db=db, agent_type="openai-tools", extra_tools=extra_tools, verbose=True
    )
else:
    st.error("Groq API Key missing. Please check your configuration.")
    agent_executor = None

# 5. Build the Dashboard Navigation Tabs
tab1, tab2, tab3 = st.tabs(["AI Data Scientist Chat", "Pre-Built Analytics", "Unsupervised Machine Learning"])

# --- TAB 1: AI CHAT INTERFACE ---
with tab1:
    st.header("Ask the Database Anything")
    st.markdown("*Example: 'What tables are available?' or 'Show me a chart of monthly sales for 2017'*")
    
    # Initialize session state for persistent chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display historical conversation
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Process new user input
    if user_query := st.chat_input("Enter your business analytics question:"):
        with st.chat_message("user"):
            st.markdown(user_query)
        st.session_state.messages.append({"role": "user", "content": user_query})

        if agent_executor:
            with st.chat_message("assistant"):
                with st.spinner("Analyzing database schemas and computing responses..."):
                    try:
                        response = agent_executor.invoke({"input": user_query})
                        output_text = response["output"]
                        st.markdown(output_text)
                        
                        # Contextual check: If the agent successfully generated a local image file, render it directly inside the chat interface
                        if "monthly_sales.png" in output_text and os.path.exists("monthly_sales.png"):
                            st.image("monthly_sales.png", caption="Dynamic Monthly Sales Chart")
                        if "hourly_peaks.png" in output_text and os.path.exists("hourly_peaks.png"):
                            st.image("hourly_peaks.png", caption="Dynamic Hourly Purchase Patterns")
                        if "rfm_clusters.png" in output_text and os.path.exists("rfm_clusters.png"):
                            st.image("rfm_clusters.png", caption="Machine Learning Customer Segmentation")
                            
                        st.session_state.messages.append({"role": "assistant", "content": output_text})
                    except Exception as e:
                        st.error(f"An execution pipeline error occurred: {str(e)}")

# --- TAB 2: PRE-BUILT ANALYTICS ---
with tab2:
    st.header("Standard Executive Reporting")
    st.markdown("Manually trigger automated pipeline visualizations without utilizing the chat interface.")
    
    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True):
            st.subheader("Annual Revenue Trend")
            st.markdown("View a detailed 12-month breakdown of sales performance.")
            selected_year = st.selectbox("Select Year:", [2016, 2017, 2018], index=1)
            if st.button("Generate 12-Month Revenue Report", key="btn_rev"):
                with st.spinner("Processing transactional records..."):
                    chart_img = plot_monthly_sales(year=selected_year)
                    st.image(chart_img, use_container_width=True)
    with col2:
        with st.container(border=True):
            st.subheader("Traffic Hotspots")
            st.markdown("Identify the time of day with the highest volume of purchases. **Note: This chart is an All-Time Aggregate**, meaning it calculates the busiest time of day across the entire history of the company to help you schedule support shifts effectively.")
            if st.button("Analyze Peak Traffic Hours", key="btn_traffic"):
                with st.spinner("Processing purchase timestamps..."):
                    chart_img = plot_hourly_peaks()
                    st.image(chart_img, use_container_width=True)

# --- TAB 3: MACHINE LEARNING SEGMENTATION ---
with tab3:
    st.header("Customer Behavioral Clustering")
    
    with st.container(border=True):
        st.markdown("### RFM Analysis Pipeline")
        st.markdown("Automatically segments your customer base based on their purchasing behavior (Recency, Frequency, Monetary value).")
        
        if st.button("Run Segmentation Pipeline", key="btn_kmeans"):
            with st.spinner("Analyzing customer behavior..."):
                ml_img = generate_rfm_cluster(n_clusters=4)
                st.image(ml_img, caption="Customer Segmentation Boundaries", use_container_width=True)
                
                # Business summary for the CEO
                st.info("""
                **Executive Summary of Segments:**
                - **VIP / Champions:** High spenders who buy frequently and recently. Offer them exclusive loyalty rewards.
                - **Loyal Customers:** Frequent buyers with consistent spend. Keep them engaged with targeted upsells.
                - **At-Risk / Needs Attention:** Past customers who haven't purchased recently. Re-engage them with win-back campaigns.
                - **New / Low-Value:** Recent or infrequent buyers with low spend. Focus on onboarding and nurturing.
                """)
                st.success("Customer profiles successfully grouped. Data segments are ready for tactical marketing exports.")