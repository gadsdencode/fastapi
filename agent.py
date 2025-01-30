# agent.py

from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing import Annotated
from typing_extensions import TypedDict


class State(TypedDict):
    messages: Annotated[list, add_messages]


def create_agent():
    """Creates a LangGraph agent using ChatAnthropic"""
    graph_builder = StateGraph(State)
    llm = ChatAnthropic(model="claude-3-5-sonnet-20240620")

    def chatbot(state: State):
        """Processes user input and returns an AI-generated response."""
        return {"messages": [llm.invoke(state["messages"])]}

    graph_builder.add_node("chatbot", chatbot)
    graph_builder.add_edge(START, "chatbot")
    graph_builder.add_edge("chatbot", END)

    return graph_builder.compile()


# Initialize the new LangGraph agent
the_langraph_graph = create_agent()
