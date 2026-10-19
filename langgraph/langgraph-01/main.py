from langchain_core.messages import HumanMessage
from graphs import build_workflow
from state import ContentState

def run_content_review_pipeline():
    topic = input("Enter Blog topic: ").strip()

    if not topic:
        print("A blog topic is required")
        return

    workflow = build_workflow()

    initial_state: ContentState = {
        "messages" : [HumanMessage(content = f"Write about: {topic}")],
        "topic" : topic,
        "draft" : "",
        "edited_version" : "",
        "status" : "drafting"
    }

    final_state = workflow.invoke(initial_state)

    print(final_state["topic"])
    print(final_state["draft"])
    print(final_state["edited_version"])
    print(final_state["status"].upper())
    for message in final_state["messages"]:
        print(message.content)


if __name__ == "__main__":
    main()

