from agno.agent import Agent  # Importing the Agent class for creating agents
from openai import OpenAI  # Importing OpenAI client for API interactions
from agno.models.openai import (
    OpenAIChat,
)  # Importing OpenAIChat model for GPT-based interactions
from agno.playground import (
    Playground,
    serve_playground_app,
)  # For creating and serving a playground app
from agno.storage.agent.sqlite import (
    SqliteAgentStorage,
)  # For storing agent sessions in SQLite
from agno.tools.duckduckgo import DuckDuckGoTools  # For web search capabilities


import os
from dotenv import load_dotenv  # For loading environment variables from a .env file

# Load environment variables from a .env file
load_dotenv()

# Set API keys from environment variables
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["ASSISTANT_ID"] = os.getenv("ASSISTANT_ID")

# Path to the SQLite database for storing agent sessions
agent_storage: str = "tmp/agents.db"

# Initialize OpenAI client
client = OpenAI()
assistant_id = os.getenv("ASSISTANT_ID")

# Define a web agent with DuckDuckGo search capabilities
nifty50_agent = Agent(
    name="Tech Analysis - Nifty50",
    model=OpenAIChat(id="gpt-4o-mini"),
    tools=[DuckDuckGoTools()],
    instructions=["Always include sources", "Always use tables to display data"],
    storage=SqliteAgentStorage(table_name="web_agent", db_file=agent_storage),
    add_datetime_to_instructions=True,  # Include current date and time in instructions
    add_history_to_messages=True,  # Include conversation history in messages
    num_history_responses=5,  # Number of history responses to include
    markdown=True,  # Enable markdown formatting
)


app = Playground(agents=[nifty50_agent]).get_app()

# Serve the playground app with live reload enabled
if __name__ == "__main__":
    serve_playground_app("nifty:app", reload=True)
