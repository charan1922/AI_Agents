from agno.agent import Agent  # Importing the Agent class for creating agents
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
from agno.tools.googlesearch import GoogleSearchTools  # For web search capabilities
from agno.tools.yfinance import YFinanceTools  # For financial data retrieval
from agno.models.groq import Groq  # Importing Groq model for Groq-based interactions

# Path to the SQLite database for storing agent sessions
agent_storage: str = "tmp/agents.db"

# Define a web agent with DuckDuckGo search capabilities
web_agent_openai_duckduckgo = Agent(
    name="Open AI Web Agent (DuckDuckGo)",
    model=OpenAIChat(id="gpt-4o-mini"),
    tools=[DuckDuckGoTools()],
    instructions=["Always include sources"],
    storage=SqliteAgentStorage(table_name="web_agent", db_file=agent_storage),
    add_datetime_to_instructions=True,  # Include current date and time in instructions
    add_history_to_messages=True,  # Include conversation history in messages
    num_history_responses=5,  # Number of history responses to include
    markdown=True,  # Enable markdown formatting
)

# Define a finance agent with YFinance tools for financial data
finance_agent = Agent(
    name="Open AI Finance Agent",
    model=OpenAIChat(id="gpt-4o-mini"),
    tools=[
        YFinanceTools(
            stock_price=True,
            analyst_recommendations=True,
            company_info=True,
            company_news=True,
            stock_fundamentals=True,
            income_statements=True,
            key_financial_ratios=True,
            technical_indicators=True,
            historical_prices=True,
            enable_all=True,
            cache_results=True,
        )
    ],
    instructions=["Always use tables to display data"],
    storage=SqliteAgentStorage(table_name="finance_agent", db_file=agent_storage),
    add_datetime_to_instructions=True,
    add_history_to_messages=True,
    num_history_responses=5,
    markdown=True,
)

groq_web_agent_duckduckgo = Agent(
    name="Groq Web Agent (DuckDuckGo)",
    role="search the web for information",
    model=Groq(id="qwen-2.5-32b"),
    tools=[DuckDuckGoTools()],
    storage=SqliteAgentStorage(table_name="groq_web_agent", db_file=agent_storage),
    instructions=["Always include sources"],
    show_tool_calls=True,  # Display tool calls in responses
    add_datetime_to_instructions=True,  # Include current date and time in instructions
    add_history_to_messages=True,  # Include conversation history in messages
    num_history_responses=5,  # Number of history responses to include
    markdown=True,  # Enable markdown formatting
)

groq_web_agent_google = Agent(
    name="Groq Web Agent (Google)",
    role="search the web for information",
    model=Groq(id="qwen-2.5-32b"),
    tools=[GoogleSearchTools(fixed_max_results=3)],
    storage=SqliteAgentStorage(
        table_name="groq_web_agent_google", db_file=agent_storage
    ),
    instructions=["Always include sources"],
    show_tool_calls=True,  # Display tool calls in responses
    add_datetime_to_instructions=True,  # Include current date and time in instructions
    add_history_to_messages=True,  # Include conversation history in messages
    num_history_responses=5,  # Number of history responses to include
    markdown=True,  # Enable markdown formatting
)
# Create a playground app with the defined agents
app = Playground(
    agents=[
        finance_agent,
        groq_web_agent_duckduckgo,
        web_agent_openai_duckduckgo,
        groq_web_agent_google,
    ]
).get_app()

# Serve the playground app with live reload enabled
if __name__ == "__main__":
    serve_playground_app("playground:app", reload=True)
