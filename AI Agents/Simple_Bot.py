# reddit me 10 pages , some tech newsletter (scrap) -> llm se article gen-> google drive me save kiya -> llm se image banwaya -> linkedin oar post kiya 

# insta page -> rag model (in some great data , new curious) -> create basic audio from tts like in voice of sinchan , or doremon , explaining concepts -> then in background use gameplay -> make automated captions using text and place on video using ffmpeg -> upload 5 times daily on insta and yt shorts -> $$$$$

from typing import Dict, TypedDict, List, Union
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

class AgentState(TypedDict):
    messages : List[HumanMessage]

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

def process(state : AgentState):
    response = llm.invoke(state['messages'])
    print(response.content)
    print(response)
    return state

graph = StateGraph(AgentState)
graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process", END)

agent = graph.compile()

user_input = input("Enter: ")
while user_input != "exit":
    agent.invoke({"messages" : [HumanMessage(content = user_input)]})
    user_input = input("Enter: ")