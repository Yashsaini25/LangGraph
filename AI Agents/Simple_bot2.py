from typing import Dict, TypedDict, List, Union
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import json
import os

load_dotenv()
HISTORY_FILE = "chat_history.json"

class AgentState(TypedDict):
    messages : List[Union[HumanMessage, AIMessage]]

def load_history() -> List[BaseMessage]:
    if not os.path.exists(HISTORY_FILE) or os.path.getsize(HISTORY_FILE) == 0:
        return []
    
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            raw_messages = json.load(f)
            messages = []
            for msg in raw_messages:
                if msg["type"] == "human":
                    messages.append(HumanMessage(content=msg["content"]))
                elif msg["type"] == "ai":
                    messages.append(AIMessage(content=msg["content"]))
            return messages
    except json.JSONDecodeError:
        print("Warning: chat_history.json is corrupted. Starting fresh.")
        return []

def save_message(message: BaseMessage):
    try:
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = []
    except json.JSONDecodeError:
        data = []

    msg_type = "human" if isinstance(message, HumanMessage) else "ai"
    data.append({"type": msg_type, "content": message.content})

    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

def process(state : AgentState):
    response = llm.invoke(state['messages'])
    ai_msg = AIMessage(content = response.content)
    state["messages"].append(ai_msg)
    save_message(ai_msg)
    print(response.content)
    return state

graph = StateGraph(AgentState)
graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process", END)

agent = graph.compile()

conv_history = load_history()
print(conv_history)
user_input = input("Enter: ")

while user_input != "exit":
    human_msg = HumanMessage(content = user_input)
    conv_history.append(human_msg)
    print(conv_history)
    save_message(human_msg)
    result = agent.invoke({"messages" : conv_history})
    conv_history = result["messages"]
    user_input = input("Enter: ")