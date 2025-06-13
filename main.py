import os
import openai
from typing import TypedDict, List, Optional
from langchain_core.messages import HumanMessage, ToolMessage, BaseMessage
from langchain.graph import StateGraph, END
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from pyswip import Prolog
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv
from typing import TypedDict, List, Optional
from langchain_core.messages import BaseMessage


load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("OpenAI API key not found. Ensure OPENAI_API_KEY is set in the .env file.")

openai.api_key = api_key

from typing import TypedDict, List, Optional
from langchain_core.messages import BaseMessage

class ConversationState(TypedDict):
    input: str
    messages: List[BaseMessage]
    response: Optional[BaseMessage]

def add_user_message(state):
    from langchain_core.messages import HumanMessage
    state["messages"].append(HumanMessage(content=state["input"]))
    return state