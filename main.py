import os
import openai
from typing import TypedDict, List, Optional
from langchain_core.messages import HumanMessage, ToolMessage, BaseMessage
from langgraph.graph import StateGraph, END
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from pyswip import Prolog
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv
from typing import TypedDict, List, Optional
from langchain_core.messages import BaseMessage
from rag_9 import retrieve_context_rag

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OpenAI API key not found. Ensure OPENAI_API_KEY is set in the .env file.")
openai.api_key = api_key



kb = Prolog()
kb.consult("family_tree.kb")

@tool
def query_knowledge_base(query: str) -> str:
    try:
        results = list(kb.query(query))
        if results:
            return f"Query results: {results}"
        else:
            return "No results found for the query."
    except Exception as e:
        return f"Error executing query: {str(e)}"

tools = [query_knowledge_base, retrieve_context_rag]
model = init_chat_model("gpt-4o-mini", model_provider="openai").bind_tools(tools)


with open("family_tree.kb", "r") as file:
    kb_content = file.read()


class ConversationState(TypedDict):
    input: str
    messages: List[BaseMessage]
    response: Optional[BaseMessage]

def add_user_message(state: ConversationState):
    state["messages"].append(HumanMessage(content=state["input"]))
    return state

def ask_model(state: ConversationState):
    response = model.invoke(state["messages"])
    state["response"] = response
    state["messages"].append(response)
    return state

def handle_tool_call(state: ConversationState):
    if hasattr(state["response"], "tool_calls"):
        for call in state["response"].tool_calls:
            tool_name = call["name"]
            args = call["args"]
            if tool_name == "query_knowledge_base":
                result = query_knowledge_base(args["query"])
            elif tool_name == "retrieve_context_rag":
                result = retrieve_context_rag(args["question"])
            else:
                result = f"Unknown tool: {tool_name}"
            state["messages"].append(ToolMessage(
                content=result,
                tool_call_id=call["id"]
            ))
        followup = model.invoke(state["messages"])
        state["messages"].append(followup)
        state["response"] = followup
    return state

graph_builder = StateGraph(ConversationState)
graph_builder.add_node("add_user_message", add_user_message)
graph_builder.add_node("ask_model", ask_model)
graph_builder.add_node("handle_tool_call", handle_tool_call)

graph_builder.set_entry_point("add_user_message")
graph_builder.add_edge("add_user_message", "ask_model")

def check_tool_use(state: ConversationState):
    if hasattr(state["response"], "tool_calls") and state["response"].tool_calls:
        return "handle_tool_call"
    return END

graph_builder.add_conditional_edges("ask_model", check_tool_use, {
    "handle_tool_call": "handle_tool_call",
    END: END
})

app = graph_builder.compile()

if __name__ == "__main__":
    initial_state = {
        "input": "Who is the grandfather of joe?",
        "messages": [],
        "response": None
    }
    final_state = app.invoke(initial_state)
    print("Final Response:\n", final_state["messages"][-1].content)