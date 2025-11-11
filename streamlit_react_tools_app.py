import streamlit as st
import os
import pandas as pd
import sqlite3
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import tool

st.set_page_config(page_title="SQL AI by Rasyid", page_icon="💬", layout="wide")
st.title("💬 Data AI Agent by Rasyid")
st.caption("A chatbot that can answer questions about your uploaded data using SQL")

with st.sidebar:
    st.subheader("⚙️ Settings")

    google_api_key = st.text_input("Google AI API Key", type="password")

    st.divider()
    st.markdown("### 📂 Upload Your Data File")
    uploaded_file = st.file_uploader(
        "Upload a CSV, Excel, JSON, or Parquet file",
        type=["csv", "xlsx", "json", "parquet"]
    )

    st.divider()
    st.markdown("### 🤖 Model & Parameters")

    selected_model = st.selectbox(
        "Select Gemini Model",
        ["gemini-2.5-flash", "gemini-2.5-pro"],
        index=0,
    )

    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05)
    top_p = st.slider("Top P", 0.0, 1.0, 0.9, 0.05)
    top_k = st.slider("Top K", 1, 100, 40, 1)
    max_tokens = st.slider("Max Output Tokens", 128, 4096, 1024, 128)

    st.divider()
    reset_button = st.button("🗑️ Reset Conversation", help="Clear all messages and start fresh")

if not google_api_key:
    st.info("Please add your Google AI API key in the sidebar to start chatting.", icon="🗝️")
    st.stop()

DB_PATH = "uploaded_data.db"

def load_file_to_sqlite(uploaded_file):
    """Reads uploaded file and saves it as a SQLite table."""
    if uploaded_file is None:
        return None, "No file uploaded yet."

    file_name = uploaded_file.name
    table_name = os.path.splitext(file_name)[0].replace(" ", "_")

    try:
        if file_name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        elif file_name.endswith(".xlsx"):
            df = pd.read_excel(uploaded_file)
        elif file_name.endswith(".json"):
            df = pd.read_json(uploaded_file)
        elif file_name.endswith(".parquet"):
            df = pd.read_parquet(uploaded_file)
        else:
            return None, "Unsupported file format."

        if df.empty:
            return None, "The uploaded file is empty."

        conn = sqlite3.connect(DB_PATH)
        df.to_sql(table_name, conn, if_exists="replace", index=False)
        conn.close()

        return table_name, f"✅ File successfully loaded as table: **{table_name}**"

    except Exception as e:
        return None, f"❌ Error loading file: {e}"

def execute_sql_query(sql_query: str):
    """Executes SQL query on the uploaded SQLite DB."""
    if not os.path.exists(DB_PATH):
        return "⚠️ Database not found. Please upload a data file first."

    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query(sql_query, conn)
        conn.close()
        if df.empty:
            return "No results."
        return df.head(20).to_markdown(index=False)
    except Exception as e:
        return f"Error executing SQL: {e}"

def get_schema_overview():
    """Returns schema info and sample data."""
    if not os.path.exists(DB_PATH):
        return "⚠️ No database found. Please upload a file first."

    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        tables = cursor.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()

        schema_info = ""
        for t in tables:
            tname = t[0]
            schema = cursor.execute(f"PRAGMA table_info({tname})").fetchall()
            sample = pd.read_sql_query(f"SELECT * FROM {tname} LIMIT 5", conn)
            schema_info += f"### Table: {tname}\n\nColumns:\n"
            for col in schema:
                schema_info += f"- {col[1]} ({col[2]})\n"
            schema_info += "\nSample data:\n"
            schema_info += sample.head(5).to_markdown(index=False)
            schema_info += "\n\n"
        conn.close()
        return schema_info
    except Exception as e:
        return f"Error getting schema info: {e}"

@tool
def execute_sql(sql_query: str):
    """Execute a SQL query against the uploaded database."""
    result = execute_sql_query(sql_query)
    return f"```sql\n{sql_query}\n```\n\nQuery Results:\n{result}"

@tool
def get_schema_info():
    """Get info about the uploaded database schema and sample data."""
    return get_schema_overview()

if uploaded_file:
    table_name, msg = load_file_to_sqlite(uploaded_file)
    st.sidebar.success(msg)
else:
    st.warning("Please upload a dataset file to start querying.", icon="📁")
    st.stop()

session_signature = f"{google_api_key}-{selected_model}-{temperature}-{top_p}-{top_k}-{max_tokens}-{uploaded_file.name}"

if ("agent" not in st.session_state) or (st.session_state.get("_session_sig") != session_signature):
    try:
        llm = ChatGoogleGenerativeAI(
            model=selected_model,
            google_api_key=google_api_key,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_output_tokens=max_tokens,
        )

        st.session_state.agent = create_react_agent(
            model=llm,
            tools=[get_schema_info, execute_sql],
            prompt="""You are a helpful assistant that can answer questions about uploaded tabular data using SQL.

            IMPORTANT:
            1. Always call get_schema_info() first to learn the database.
            2. Write a valid SQLite SQL query.
            3. Execute it via execute_sql().
            4. Explain the result clearly and in natural language.
            """,
        )

        st.session_state._session_sig = session_signature
        st.session_state.pop("messages", None)

    except Exception as e:
        st.error(f"Error initializing model: {e}")
        st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

if reset_button:
    st.session_state.pop("agent", None)
    st.session_state.pop("messages", None)
    st.rerun()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prompt = st.chat_input("Ask a question about your uploaded data...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    try:
        messages = []
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                messages.append(AIMessage(content=msg["content"]))

        with st.spinner("Thinking..."):
            response = st.session_state.agent.invoke({"messages": messages})

            if "messages" in response and len(response["messages"]) > 0:
                answer = response["messages"][-1].content
            else:
                answer = "I'm sorry, I couldn't generate a response."

    except Exception as e:
        answer = f"An error occurred: {e}"

    with st.chat_message("assistant"):
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
