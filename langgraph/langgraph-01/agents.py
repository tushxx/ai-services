import os
from google import genai
from dotenv import load_dotenv

load_dotenv()

client = genai.Client(api_key = os.getenv("GEMINI_API_KEY"))

def generate_content(prompt):
    response = client.models.generate_content(
        model = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite"),
        contents = prompt
    )
    return response.text.strip()


def write_blog_draft(topic):
    prompt = f"""You are a professional blog writer. write a 150-200 blog post about : {topic} 
    The blog post must include:
    - A clear title
    - A shorte introduction
    - Three key points
    - A concise conclusion

    Return only the blog post without explanations
    """
    return generate_content(prompt)


def edit_blog_content(draft):
    prompt = f""" You are a professional editor. Improve the following blog post by correcting grammar and enhancing clarity, flow, word choice, and tone while preserving its original structure and meaning.

    Return only the improved blog post without explanations.

    Blog post:
    {draft}
    """
    return generate_content(prompt)

def make_publish_decision(edited_content):
    prompt = f"""You are a professional publisher. Evaluate the following blog post on quality, clarity, structure, engagement, and completeness.

    Respond using exactly this format:
    DECISION: PUBLISH or REJECT
    REASONING: One sentence explanation

    Blog post:
    {edited_content}
    """

    response = generate_content(prompt)
    decision = "rejected"
    reasoning = "Content review complete"

    for line in response.splitlines():
        key, separator, value = line.partition(":")
        if not separator:
            continue
        if key.strip().upper() == "DECISION":
            decision = "published" if value.strip().upper() == "PUBLISH" else "rejected"
        if key.strip().upper() == "REASONING" and value.strip():
            reasoning = value.strip()

    return decision, reasoning