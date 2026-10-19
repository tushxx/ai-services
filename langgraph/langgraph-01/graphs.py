from langgraph.graph import START, END, StateGraph
from state import ContentState
from nodes import write_draft, edit_content, publish_decision


def build_workflow():
    graph = StateGraph(ContentState)
    graph.add_node("write", write_draft)
    graph.add_node("edit", edit_content)
    graph.add_node("publish", publish_decision)

    graph.add_edge(START, "write")
    graph.add_edge("write", "edit")
    graph.add_edge("edit", "publish")
    graph.add_edge("publish", END)

    return graph.compile()

