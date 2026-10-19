from langchain_core.messages import AIMessage
from state import ContentState
from agents import write_blog_draft, edit_blog_content, make_publish_decision


def write_draft(state):
    draft = write_blog_draft(state["topic"])
    return {
        "draft" : draft,
        "messages" : [AIMessage(content = "Writer created the initial blog draft.")],
        "status" : "editing"
    }

def edit_content(state):
    edited_version = edit_blog_content(state["draft"])
    return {
        "edited_version": edited_version,
        "messages": [AIMessage(content="Editor improved the blog content.")],
        "status": "reviewing"
    }

def publish_decision(state):
    decision, reasoning = make_publish_decision(state["edited_version"])
    return {
        "status" : decision,
        "messages" : [AIMessage(content = f"Publisher: {reasoning}")]
    }

