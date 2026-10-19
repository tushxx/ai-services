#state.py
from typing import Annotated, Literal, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
 
class ContentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    topic: str
    draft: str
    edited_version: str
    status: Literal["drafting", "editing", "reviewing", "published", "rejected"]