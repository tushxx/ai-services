import os
import re
import sys

import chromadb
import fitz
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from google import genai


load_dotenv()


def load_and_chunk_document(document_path):
    """
    Load the PDF, split it into numbered sections, and create
    500-character chunks with a 420-character step.
    """

    doc = fitz.open(document_path)

    full_text = ""

    for page in doc:
        full_text += page.get_text("text") + "\n"

    doc.close()

    # Sections look like:
    # 1. Employment and Probation
    # 2. Working Hours and Attendance
    # ...
    # 10. Information Security and Device Use
    heading_pattern = re.compile(
        r"(?m)^(\d+)\.\s+([^\n]+)"
    )

    matches = list(heading_pattern.finditer(full_text))

    if not matches:
        return []

    chunks = []

    for i, match in enumerate(matches):

        section = f"{match.group(1)}. {match.group(2).strip()}"

        # Start at the heading so the heading belongs to the chunk
        start = match.start()

        # Stop at the next section heading
        if i + 1 < len(matches):
            end = matches[i + 1].start()
        else:
            end = len(full_text)

        section_text = full_text[start:end]

        # Collapse line breaks/whitespace into spaces
        section_text = re.sub(
            r"\s+",
            " ",
            section_text
        ).strip()

        # 500 character window
        # 420 character step
        window_size = 500
        step = 420

        position = 0

        while position < len(section_text):

            chunk_text = section_text[
                position:position + window_size
            ].strip()

            # Discard chunks shorter than 60 characters
            if len(chunk_text) >= 60:
                chunks.append({
                    "section": section,
                    "text": chunk_text
                })

            position += step

            # Don't create another unnecessary window
            if position >= len(section_text):
                break

    return chunks


def retrieve_sections(question, embedding_model, collection):
    """
    Retrieve the four most relevant chunks.
    """

    # Embed question using the supplied SentenceTransformer
    question_embedding = embedding_model.encode(
        question
    ).tolist()

    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=4,
        include=[
            "documents",
            "metadatas",
            "distances"
        ]
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    retrieved_chunks = []

    for document, metadata, distance in zip(
        documents,
        metadatas,
        distances
    ):
        similarity = round(
            1 - distance,
            2
        )

        retrieved_chunks.append({
            "chunk": document,
            "section": metadata["section"],
            "similarity": similarity
        })

    # Most relevant first
    retrieved_chunks.sort(
        key=lambda item: item["similarity"],
        reverse=True
    )

    return retrieved_chunks


def augment_prompt(question, retrieved_chunks):
    """
    Build the grounded prompt for Gemini.
    """

    context = []

    for i, item in enumerate(
        retrieved_chunks,
        start=1
    ):
        context.append(
            f"""Context {i}
Section: {item["section"]}
Excerpt: {item["chunk"]}"""
        )

    context_block = "\n\n".join(context)

    prompt = f"""
You are an employee policy assistant for Halcyon Logistics Pvt Ltd.

Answer the employee's question using ONLY the policy excerpts
provided below.

Do not use outside knowledge.
Do not make assumptions.
Do not invent policy rules.

Name the supporting policy section in your answer.

If the supplied excerpts do not contain enough information to
answer the question, state that the matter is not covered in
the provided policy excerpts.

POLICY EXCERPTS:

{context_block}

EMPLOYEE QUESTION:

{question}

Provide a concise, grounded answer and name the supporting
section.
""".strip()

    return prompt


def generate_answer(prompt, client):
    """
    Generate the final answer using Gemini.
    """

    model_name = os.getenv(
        "GEMINI_MODEL"
    )

    response = client.models.generate_content(
        model=model_name,
        contents=prompt
    )

    return response.text.strip()


def policy_qa_pipeline(question, document_path):
    """
    Run the complete employee policy RAG pipeline.
    """

    # ---------------------------------------------------------
    # 1. Load and chunk document
    # ---------------------------------------------------------

    chunks = load_and_chunk_document(
        document_path
    )

    # ---------------------------------------------------------
    # 2. Load embedding model
    # ---------------------------------------------------------

    embedding_model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    # ---------------------------------------------------------
    # 3. Create embeddings in a single batch
    # ---------------------------------------------------------

    documents = [
        chunk["text"]
        for chunk in chunks
    ]

    embeddings = embedding_model.encode(
        documents
    ).tolist()

    # ---------------------------------------------------------
    # 4. Create temporary ChromaDB collection
    # ---------------------------------------------------------

    chroma_client = chromadb.Client()

    collection_name = "employee_policy"

    # Remove collection from previous invocation
    try:
        chroma_client.delete_collection(
            name=collection_name
        )
    except Exception:
        pass

    collection = chroma_client.create_collection(
        name=collection_name,
        configuration={
            "hnsw": {
                "space": "cosine"
            }
        }
    )

    # ---------------------------------------------------------
    # 5. Store chunks
    # ---------------------------------------------------------

    ids = [
        f"chunk_{i}"
        for i in range(len(chunks))
    ]

    metadatas = [
        {
            "section": chunk["section"]
        }
        for chunk in chunks
    ]

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas
    )

    # ---------------------------------------------------------
    # 6. Retrieve relevant chunks
    # ---------------------------------------------------------

    retrieved_chunks = retrieve_sections(
        question,
        embedding_model,
        collection
    )

    # ---------------------------------------------------------
    # 7. Augment prompt
    # ---------------------------------------------------------

    prompt = augment_prompt(
        question,
        retrieved_chunks
    )

    # ---------------------------------------------------------
    # 8. Gemini client
    # ---------------------------------------------------------

    api_key = os.getenv(
        "GEMINI_API_KEY"
    )

    client = genai.Client(
        api_key=api_key
    )

    # ---------------------------------------------------------
    # 9. Generate answer
    # ---------------------------------------------------------

    answer = generate_answer(
        prompt,
        client
    )

    # ---------------------------------------------------------
    # 10. Create unique source list
    # ---------------------------------------------------------

    sources = []

    for item in retrieved_chunks:

        section = item["section"]

        if section not in sources:
            sources.append(section)

    # ---------------------------------------------------------
    # 11. Return required structure
    # ---------------------------------------------------------

    return {
        "question": question,
        "retrieved_chunks": retrieved_chunks,
        "sources": sources,
        "answer": answer
    }


HANDBOOK = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "data",
    "employee_policy_handbook.pdf",
)


def test_load_and_chunk_document_splits_every_numbered_section():
    chunks = load_and_chunk_document(HANDBOOK)

    sections = []
    for chunk in chunks:
        assert set(chunk) == {"section", "text"}
        assert 60 <= len(chunk["text"]) <= 500
        if chunk["section"] not in sections:
            sections.append(chunk["section"])

    assert sections == [
        "1. Employment and Probation",
        "2. Working Hours and Attendance",
        "3. Leave Entitlement",
        "4. Remote Work and Hybrid Policy",
        "5. Performance Appraisal and Increments",
        "6. Compensation and Payroll",
        "7. Travel and Expense Reimbursement",
        "8. Training and Certification Support",
        "9. Information Security and Device Use",
        "10. Grievance and Disciplinary Procedure",
    ]


def test_a_document_with_no_numbered_sections_returns_no_chunks(tmp_path):
    path = tmp_path / "plain.pdf"
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), "This page has no numbered heading.")
    document.save(path)
    document.close()

    assert load_and_chunk_document(str(path)) == []


def test_retrieve_sections_puts_the_matching_section_first():
    chunks = load_and_chunk_document(HANDBOOK)
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    chroma_client = chromadb.Client()
    collection = chroma_client.create_collection(
        name="employee_policy_test",
        configuration={"hnsw": {"space": "cosine"}},
    )
    collection.add(
        ids=[f"chunk_{i}" for i in range(len(chunks))],
        embeddings=embedding_model.encode(
            [chunk["text"] for chunk in chunks]
        ).tolist(),
        documents=[chunk["text"] for chunk in chunks],
        metadatas=[{"section": chunk["section"]} for chunk in chunks],
    )

    retrieved = retrieve_sections(
        "How long is the probation period for a new employee?",
        embedding_model,
        collection,
    )

    assert len(retrieved) == 4
    assert [item["similarity"] for item in retrieved] == sorted(
        (item["similarity"] for item in retrieved),
        reverse=True,
    )
    assert retrieved[0]["section"] == "1. Employment and Probation"
    assert "6 months" in retrieved[0]["chunk"]
    assert all(set(item) == {"chunk", "section", "similarity"} for item in retrieved)


def test_augment_prompt_carries_the_question_and_the_excerpts():
    retrieved = [{
        "chunk": "New employees serve a probation period of 6 months.",
        "section": "1. Employment and Probation",
        "similarity": 0.76,
    }]

    prompt = augment_prompt(
        "How long is the probation period?",
        retrieved,
    )

    assert "Halcyon Logistics Pvt Ltd" in prompt
    assert "How long is the probation period?" in prompt
    assert "1. Employment and Probation" in prompt
    assert "New employees serve a probation period of 6 months." in prompt
    assert "POLICY EXCERPTS:" in prompt


def test_generate_answer_returns_the_stripped_model_text(monkeypatch):
    monkeypatch.setenv("GEMINI_MODEL", "gemini-2.5-flash-lite")

    class Models:
        def generate_content(self, model, contents):
            assert model == "gemini-2.5-flash-lite"
            assert "probation" in contents

            class Response:
                text = "  six months  "

            return Response()

    class Client:
        models = Models()

    assert generate_answer("probation?", Client()) == "six months"


def test_policy_qa_pipeline_returns_the_required_fields(monkeypatch):
    monkeypatch.setattr(
        sys.modules[__name__],
        "generate_answer",
        lambda prompt, client: "New employees serve a probation period of 6 months.",
    )

    result = policy_qa_pipeline(
        "How long is the probation period for a new employee?",
        HANDBOOK,
    )

    assert result["question"] == "How long is the probation period for a new employee?"
    assert result["answer"] == "New employees serve a probation period of 6 months."
    assert result["retrieved_chunks"][0]["section"] == "1. Employment and Probation"
    assert result["sources"][0] == "1. Employment and Probation"
    assert result["sources"] == list(dict.fromkeys(
        item["section"] for item in result["retrieved_chunks"]
    ))


if __name__ == "__main__":

    question = input(
        "Enter your question: "
    ).strip()

    if not question:

        print(
            "Question cannot be empty."
        )

    else:

        result = policy_qa_pipeline(
            question,
            "data/employee_policy_handbook.pdf"
        )

        print(
            f"\nQuestion: {result['question']}"
        )

        print(
            "\nRetrieved Sections:"
        )

        for i, item in enumerate(
            result["retrieved_chunks"],
            start=1
        ):
            print(
                f"{i}. {item['section']} "
                f"(similarity: "
                f"{item['similarity']:.2f})"
            )

        print(
            f"\nSources: "
            f"{', '.join(result['sources'])}"
        )

        print("\nAnswer:")

        print(
            result["answer"]
        )