import os
from dotenv import load_dotenv
from google import genai

load_dotenv()


def extract_repair_tasks(report: str) -> list[str]:
    prompt = f"""
        You are a property maintenance specialist.

        Read the inspection report and identify every repair or follow-up
        job mentioned in it.

        Return one job per line using exactly this format:
        REPAIR: description of the repair

        Do not include any other text.

        Inspection report:
        {report}
    """

    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    response = client.models.generate_content(
        model=os.getenv("GEMINI_MODEL"),
        contents=prompt
    )

    tasks = []

    for line in (response.text or "").splitlines():
        line = line.strip()

        if line.startswith("REPAIR:"):
            task = line[len("REPAIR:"):].strip()
            if task:
                tasks.append(task)

    return tasks


def assign_technician_and_urgency(
    task: str, crew: list[str]
) -> tuple[str, str]:
    prompt = f"""
        You are a property maintenance work-order coordinator.

        Assign this repair to exactly one technician from the supplied crew.
        Copy the technician's name exactly as written in the crew list.
        Choose an urgency of High, Medium, or Low.

        Return exactly two lines:
        TECHNICIAN: technician name
        URGENCY: High, Medium, or Low

        Repair task:
        {task}

        Available crew:
        {chr(10).join(crew)}
    """

    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    response = client.models.generate_content(
        model=os.getenv("GEMINI_MODEL"),
        contents=prompt
    )

    technician = ""
    urgency = ""

    for line in (response.text or "").splitlines():
        line = line.strip()

        if line.startswith("TECHNICIAN:"):
            technician = line[len("TECHNICIAN:"):].strip()
        elif line.startswith("URGENCY:"):
            urgency = line[len("URGENCY:"):].strip()

    return technician, urgency
