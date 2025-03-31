from agno.agent import Agent, RunResponse  # Importing the Agent class and response type
from agno.playground import (
    Playground,
    serve_playground_app,
)  # For creating and serving a playground app
from agno.models.openai import (
    OpenAIChat,
)  # Importing OpenAIChat model for GPT-based interactions
from agno.models.groq import Groq  # Importing Groq model for Groq-based interactions
from agno.tools.duckduckgo import DuckDuckGoTools  # For web search capabilities
from agno.tools.yfinance import YFinanceTools  # For financial data retrieval

import os
from dotenv import load_dotenv  # For loading environment variables from a .env file

# Load environment variables from a .env file
load_dotenv()

# Set API keys from environment variables
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# Define a web agent with Groq model and DuckDuckGo tools
web_agent = Agent(
    name="Web Agent",
    role="search the web for information",
    model=Groq(id="qwen-2.5-32b"),
    tools=[DuckDuckGoTools()],
    instructions="Always include the sources",
    show_tool_calls=True,  # Display tool calls in responses
    markdown=True,  # Enable markdown formatting
)

# Define a finance agent with OpenAIChat model and YFinance tools
finance_agent = Agent(
    name="Finance Agent",
    role="Get financial data",
    model=OpenAIChat(id="gpt-4o-mini"),
    tools=[
        YFinanceTools(
            stock_price=True,
            analyst_recommendations=True,
            stock_fundamentals=True,
            company_info=True,
            technical_indicators=True,
        )
    ],
    instructions="Use tables to display data",
    show_tool_calls=True,
    markdown=True,
)

# Define a team of agents combining the web and finance agents
agent_team = Agent(
    team=[web_agent, finance_agent],
    model=Groq(id="qwen-2.5-32b"),
    instructions=["Always include sources", "Use tables to display data"],
    show_tool_calls=True,
    markdown=True,
)

# Generate and print a response using the agent team
agent_team.print_response("Analyze companies TCS, DR Reddy, Infosys and HCL Tech")

# Initialize the agent with OpenAI's GPT-4 model and enable markdown formatting
# agent = Agent(model=OpenAIChat(id="gpt-4o-mini"), markdown=True)

# agent = Agent(
#     model=Groq(id="qwen-2.5-32b"),
#     description="You are an assistant please reply based ont he question",
#     tools=[DuckDuckGoTools()],
#     markdown=True,
# )
# Generate and print a response to a prompt
# agent.print_response("Share a 2 sentence horror story.")
# agent.print_response("Who won the India vs Newzealand finals in CT 2025")
