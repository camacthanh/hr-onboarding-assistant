import streamlit as st
import os
import json
import time
from dotenv import load_dotenv
from typing import List, Dict, Set
from contextlib import redirect_stdout
from io import StringIO

# --- Core Imports ---
from pinecone import Pinecone, ServerlessSpec

# --- Langchain Core Imports
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# --- Langchain Specific Imports
from langchain.tools import tool
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import create_openai_tools_agent, AgentExecutor

# --- Langchain Partner Imports ---
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

# --- Load Environment Variables ---
load_dotenv()

# --- Pinecone Configuration ---
INDEX_NAME = "hr-onboarding-index"
EMBEDDING_DIMENSION = 1536  # text-embedding-3-small

# --- Agenda 1: Define Real-World Problem & Data Needs ---

# Mock data for the RAG knowledge base
mock_hr_docs_data = [
    {
        "content": "Our Work-From-Home (WFH) policy allows employees to work remotely up to 3 days per week. You must get approval from your direct manager for your WFH schedule.",
        "metadata": {"source": "Company Handbook, Page 12"}
    },
    {
        "content": "To request Paid Time Off (PTO), please use the 'TimeOff' portal. All requests must be submitted at least 2 weeks in advance, unless it is a sick day.",
        "metadata": {"source": "Company Handbook, Page 23"}
    },
    {
        "content": "Our company health insurance plan is provided by 'Global Health Inc.' You can enroll within your first 30 days. The plan covers medical, dental, and vision.",
        "metadata": {"source": "Benefits Guide, Page 5"}
    },
    {
        "content": "The 2024 company holidays are: New Year's Day, Memorial Day, Independence Day, Labor Day, Thanksgiving Day, and Christmas Day.",
        "metadata": {"source": "Company Handbook, Page 10"}
    }
]


# --- Separate Mock Database Functions ---

def _load_mock_task_db() -> Dict[str, List[str]]:
    """Loads the mock employee task database."""
    return {
        "E123": ["Complete compliance training", "Set up 1-on-1 with manager", "Enroll in health insurance"],
        "E456": ["Sign non-disclosure agreement", "Order company laptop"],
    }


def _load_valid_orientation_topics() -> Set[str]:
    """Loads the set of valid orientation topics."""
    return {"hr_benefits", "it_security"}


# --- Agenda 4: Implement Azure OpenAI Function Calling (Tools) ---

@tool
def get_onboarding_tasks(employee_id: str) -> str:
    """
    Checks the onboarding task list for a specific employee by their ID.
    """
    # The print statement will be captured and shown in the Streamlit UI
    print(f"\n[Tool Call: get_onboarding_tasks(employee_id='{employee_id}')]")
    tasks_db = _load_mock_task_db()
    tasks = tasks_db.get(employee_id, ["No tasks found for this employee."])
    return json.dumps({"employee_id": employee_id, "tasks": tasks})


@tool
def schedule_orientation(employee_id: str, topic: str) -> str:
    """
    Schedules a mandatory orientation session for an employee on a specific topic.
    Valid topics are 'hr_benefits' or 'it_security'.
    """
    print(f"\n[Tool Call: schedule_orientation(employee_id='{employee_id}', topic='{topic}')]")
    valid_topics = _load_valid_orientation_topics()

    if topic.lower() not in valid_topics:
        return json.dumps({"status": "error", "message": f"Invalid topic. Must be one of {list(valid_topics)}."})

    return json.dumps({"status": "success", "session_id": f"SESS_98765",
                       "message": f"{topic} orientation scheduled for {employee_id}."})


# --- Cached Agent Setup ---
@st.cache_resource
def load_agent():
    """
    Initializes all clients, vector stores, and the agent executor.
    This is cached by Streamlit to run only once.
    """
    print("--- Initializing AI Agent (This runs only once) ---")

    # 1. Initialize Clients
    try:
        # --- LLM Client (Account 1) ---
        llm = AzureChatOpenAI(
            azure_deployment=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME", "gpt-4o-mini"),
            api_version="2024-07-01-preview",
            azure_endpoint=os.getenv("AZURE_OPENAI_CHAT_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_CHAT_API_KEY"),
        )

        # --- Embeddings Client (Account 2) ---
        embeddings = AzureOpenAIEmbeddings(
            model=os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT_NAME", "text-embedding-3-small"),
            azure_endpoint=os.getenv("AZURE_OPENAI_EMBEDDINGS_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_EMBEDDINGS_API_KEY"),
        )

        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        print("✅ LLM, Embeddings, and Pinecone clients initialized.")

    except Exception as e:
        print(f"❌ Error initializing clients: {e}")
        raise e

    # 2. Build Document Store with Pinecone
    try:
        if INDEX_NAME not in [index["name"] for index in pc.list_indexes().indexes]:
            print(f"Creating new Pinecone index: {INDEX_NAME}")
            pc.create_index(
                name=INDEX_NAME,
                dimension=EMBEDDING_DIMENSION,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            while not pc.describe_index(INDEX_NAME).status['ready']:
                print("Waiting for index to be ready...")
                time.sleep(1)

            print("Upserting documents to Pinecone...")
            docs = [Document(page_content=d["content"], metadata=d["metadata"]) for d in mock_hr_docs_data]
            PineconeVectorStore.from_documents(
                docs,
                embedding=embeddings,
                index_name=INDEX_NAME
            )
            print("✅ Documents upserted.")
        else:
            print(f"✅ Connecting to existing Pinecone index: {INDEX_NAME}")

        vectorstore = PineconeVectorStore(
            index_name=INDEX_NAME,
            embedding=embeddings
        )
        retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
        print("✅ Pinecone vector store and retriever ready.")

    except Exception as e:
        print(f"❌ Error creating/connecting to Pinecone: {e}")
        raise e

    # 3. Create RAG Tool
    rag_tool = create_retriever_tool(
        retriever,
        "company_handbook_retriever",
        "Searches and returns information from the company handbook about policies like WFH, PTO, holidays, and health insurance."
    )

    # 4. Combine all tools
    tools = [rag_tool, get_onboarding_tasks, schedule_orientation]

    # 5. Define Agent System Prompt
    system_prompt = """
    You are a helpful and friendly HR Onboarding Assistant.
    Your name is "BaoBot".

    - **Answer general questions:** Use the 'company_handbook_retriever' tool to answer questions about company policies (WFH, PTO, holidays, insurance).
    - **Handle specific actions:** Use your other tools to help with personal tasks like checking onboarding status or scheduling meetings.
    - **Employee ID:** For personal tasks, you MUST know the employee's ID. If you don't, ask for it. **For this demo, assume the employee ID is 'E123' unless told otherwise.**
    - **Be conversational:** Keep a chat history.
    """

    # 6. Create Agent
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent = create_openai_tools_agent(llm, tools, prompt)

    # 7. Create and return Agent Executor
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    print("--- Agent is fully loaded and ready! ---")
    return agent_executor


# --- Streamlit UI Application ---

st.title("🤖 BaoBot (HR Onboarding Assistant)")
st.caption(
    "I can answer policy questions (WFH, PTO) or help with your tasks (e.g., 'What are my tasks?' or 'Schedule my IT orientation').")

# Load the agent
try:
    agent_executor = load_agent()
except Exception as e:
    st.error(f"Failed to load AI agent. Have you set your API keys? Error: {e}")
    st.stop()

# Initialize chat history in session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display past messages
for message in st.session_state.chat_history:
    role = "user" if isinstance(message, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(message.content)

# Get new user input
if prompt := st.chat_input("Ask about WFH, PTO, or your tasks..."):
    # Add user message to history and display
    st.session_state.chat_history.append(HumanMessage(content=prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get and display assistant response
    with st.chat_message("assistant"):
        response_container = st.empty()
        with st.spinner("BaoBot is thinking..."):
            # Capture stdout logs (from verbose=True and tool prints)
            f = StringIO()
            with redirect_stdout(f):
                response = agent_executor.invoke({
                    "input": prompt,
                    "chat_history": st.session_state.chat_history
                })
            stdout_logs = f.getvalue()

        # Display the final answer
        output = response['output']
        response_container.markdown(output)

        # Add assistant message to history
        st.session_state.chat_history.append(AIMessage(content=output))

        # Display the agent's thought process in an expander
        if stdout_logs:
            with st.expander("Show agent thoughts 🧠"):
                st.code(stdout_logs, language="log")