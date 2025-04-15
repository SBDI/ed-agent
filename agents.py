"""
Ed Agent integrates:
  - DuckDuckGoTools for real-time web searches.
  - ExaTools for structured, in-depth analysis.
  - FileTools for saving the output upon user confirmation.
"""

import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(override=True)

# Importing the Agent and model classes
from agno.agent import Agent
from agno.models.groq import Groq

# Importing storage and tool classes
from agno.storage.agent.sqlite import SqliteAgentStorage
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.exa import ExaTools
from agno.tools.file import FileTools

# Import the Agent template
from prompts import AGENT_DESCRIPTION, AGENT_INSTRUCTIONS, EXPECTED_OUTPUT_TEMPLATE

# ************* Setup Paths *************
# Define the current working directory and output directory for saving files
cwd = Path(__file__).parent
output_dir = cwd.joinpath("output")
# Create output directory if it doesn't exist
output_dir.mkdir(parents=True, exist_ok=True)
# Create tmp directory if it doesn't exist
tmp_dir = cwd.joinpath("tmp")
tmp_dir.mkdir(parents=True, exist_ok=True)
# *************************************

# ************* Agent Storage *************
# Configure SQLite storage for agent sessions
agent_storage = SqliteAgentStorage(
    table_name="answer_engine_sessions",  # Table to store agent sessions
    db_file=str(tmp_dir.joinpath("agents.db")),  # SQLite database file
)
# *************************************


def tutor_agent(
    user_id: Optional[str] = None,
    model_id: str = None,
    session_id: Optional[str] = None,
    num_history_responses: int = None,
    debug_mode: bool = True,
    education_level: str = "High School",
) -> Agent:
    """
    Returns an instance of Ed Agent, an educational AI assistant with integrated tools for web search,
    deep contextual analysis, and file management.

    Ed Agent will:
      - Use DuckDuckGoTools for real-time web searches and ExaTools for in-depth analysis to gather information.
      - Generate comprehensive educational answers tailored to the specified education level that include:
          • Direct, succinct answers appropriate for the student's level.
          • Detailed explanations with supporting evidence.
          • Examples and clarification of common misconceptions.
          • Interactive elements like questions to check understanding.
      - Prompt the user:
            "Would you like to save this answer to a file? (yes/no)"
        If confirmed, it will use FileTools to save the answer in markdown format in the output directory.

    Args:
        user_id: Optional identifier for the user.
        model_id: Model identifier in the format 'groq:model_name' (e.g., "groq:llama-3.3-70b-versatile").
                 Will always use Groq with a Llama model regardless of provider specified.
        session_id: Optional session identifier for tracking conversation history.
        num_history_responses: Number of previous responses to include for context.
        debug_mode: Enable logging and debug features.
        education_level: Education level for tailoring responses (e.g., "Elementary School", "High School", "College").

    Returns:
        An instance of the configured Agent.
    """

    # Get configuration from environment variables
    default_model_name = os.environ.get("MODEL_NAME", "llama-3.3-70b-versatile")
    default_max_tokens = int(os.environ.get("MAX_TOKENS", "4000"))
    default_num_history = int(os.environ.get("NUM_HISTORY_RESPONSES", "3"))

    # Use environment values as defaults if not provided
    if model_id is None:
        model_id = f"groq:{default_model_name}"

    if num_history_responses is None:
        num_history_responses = default_num_history

    # Parse model provider and name
    provider, model_name = model_id.split(":")

    # Always use Groq with Llama model
    groq_api_key = os.environ.get("GROQ_API_KEY")

    # Default to environment MODEL_NAME if the model name doesn't contain "llama"
    if "llama" not in model_name.lower():
        model_name = default_model_name

    model = Groq(id=model_name, api_key=groq_api_key, max_tokens=default_max_tokens)

    # Get Exa API key from environment variable
    exa_api_key = os.environ.get("EXA_API_KEY")

    # Tools for Ed Agent
    tools = [
        ExaTools(
            api_key=exa_api_key,
            start_published_date=datetime.now().strftime("%Y-%m-%d"),
            type="keyword",
            num_results=3,
        ),
        DuckDuckGoTools(
            timeout=20,
            fixed_max_results=3,
        ),
        FileTools(base_dir=output_dir),
    ]

    # Modify the description to include the education level
    tutor_description = f"""You are Ed Agent, an educational AI assistant for {education_level} students.
    You have tools for web searches (DuckDuckGoTools), in-depth analysis (ExaTools), and saving files (FileTools).

    <critical>
    - Always search both DuckDuckGo and ExaTools before answering.
    - Provide sources for all data points and statistics.
    - Keep responses concise, clear, and appropriate for {education_level} level.
    - Focus on being accurate and helpful.
    </critical>"""

    # Modify the instructions to include the education level
    tutor_instructions = f"""Here's how you should answer the user's question:

    1. Gather Information
      - Search using BOTH `duckduckgo_search` and `search_exa` with relevant search terms.
      - CRITICAL: You must search both tools before answering.

    2. Construct Your Response
      - Start with a clear, direct answer tailored to a {education_level} level.
      - Include 2-3 key points with supporting evidence.
      - Use language appropriate for {education_level} students.
      - Keep your response concise and focused.

    3. Include Sources
      - List 2-3 sources that support your answer.
      - After your answer, ask if the user wants to save it to a file.

    4. Be Helpful
      - If you're unsure, acknowledge limitations and suggest follow-up questions."""

    return Agent(
        name="Ed Agent",
        model=model,
        user_id=user_id,
        session_id=session_id or str(uuid.uuid4()),
        storage=agent_storage,
        tools=tools,
        # Allow Ed Agent to read both chat history and tool call history for better context.
        read_chat_history=True,
        read_tool_call_history=True,
        # Append previous conversation responses into the new messages for context.
        add_history_to_messages=True,
        num_history_responses=num_history_responses,
        add_datetime_to_instructions=True,
        add_name_to_instructions=True,
        description=tutor_description,
        instructions=tutor_instructions,
        expected_output=EXPECTED_OUTPUT_TEMPLATE,
        debug_mode=debug_mode,
        markdown=True,
    )
