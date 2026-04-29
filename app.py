import streamlit as st
import time
from datetime import datetime
from dotenv import load_dotenv
import json
import re

from langchain.document_loaders import CSVLoader
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain

from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

# --------------------------------------------------
# ENV
# --------------------------------------------------
load_dotenv()

# --------------------------------------------------
# LOGIN
# --------------------------------------------------
EXAMINER_USERNAME = "examiner"
EXAMINER_PASSWORD = "1234"

# --------------------------------------------------
# VECTOR DB
# --------------------------------------------------
@st.cache_resource
def initialize_vector_db():
    loader = CSVLoader(file_path="ts_response.csv")
    docs = loader.load()

    embeddings = OpenAIEmbeddings()
    return FAISS.from_documents(docs, embeddings)

db = initialize_vector_db()

def retrieve_info(query, k=3):
    results = db.similarity_search(query, k=k)
    return [r.page_content for r in results]

# --------------------------------------------------
# LLM
# --------------------------------------------------
llm = ChatOpenAI(model="gpt-4-turbo", temperature=0.7)
eval_llm = ChatOpenAI(model="gpt-4-turbo", temperature=0)

# --------------------------------------------------
# CATEGORIES
# --------------------------------------------------
CATEGORIES = [
    "General Understanding",
    "Technical Understanding",
    "Methodology & Testing",
    "Problem-Solving",
    "System Thinking",
    "Reflection & Limitations",
    "Real-World Application"
]

# --------------------------------------------------
# PROMPT
# --------------------------------------------------
PROMPT_TEMPLATE = """
You are an experienced academic professor conducting a formal undergraduate viva.

CURRENT CATEGORY:
{current_category}

Student Answer:
{message}

Retrieved Knowledge:
{best_practice}

TASK:
Ask ONE clear viva question for the category.
"""

prompt = PromptTemplate(
    input_variables=["message", "best_practice", "current_category"],
    template=PROMPT_TEMPLATE
)

chain = LLMChain(llm=llm, prompt=prompt)

# --------------------------------------------------
# RESPONSE
# --------------------------------------------------
def generate_response(message, category):
    best_practice = retrieve_info(message)

    return chain.run(
        message=message,
        best_practice=best_practice,
        current_category=category
    )

# --------------------------------------------------
# EVALUATION
# --------------------------------------------------
def clean_json_output(text):
    text = re.sub(r"```json|```", "", text)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    return match.group(0) if match else text

def evaluate_answer(question, answer):
    prompt = f"""
Return JSON:
{{"overall_score":0-100,"feedback":"text"}}

Q:{question}
A:{answer}
"""
    response = eval_llm.predict(prompt)
    cleaned = clean_json_output(response)

    try:
        return json.loads(cleaned)
    except:
        return {"error": response}

# --------------------------------------------------
# STATE MACHINE
# --------------------------------------------------
def init_state():
    if "viva" not in st.session_state:
        st.session_state.viva = {
            "phase": "INIT",
            "current_question": None,
            "pending_answer": None,
            "qa_pairs": [],
            "evaluations": [],
            "question_count": 0,
            "category_index": 0,
            "skip_first_eval": True,
            "research_title": None
        }

def run_state_machine(user_input=None):
    state = st.session_state.viva

    # INIT → store title
    if state["phase"] == "INIT":
        if user_input:
            state["research_title"] = user_input
            state["phase"] = "ASK"

    # ASK QUESTION
    elif state["phase"] == "ASK":
        base = state.get("pending_answer") or state["research_title"]

        q = generate_response(
            base,
            CATEGORIES[state["category_index"]]
        )

        state["current_question"] = q
        state["phase"] = "WAIT"

    # WAIT ANSWER
    elif state["phase"] == "WAIT":
        if user_input:
            state["pending_answer"] = user_input
            state["phase"] = "EVAL"

    # EVALUATE
    elif state["phase"] == "EVAL":
        qa = {
            "question": state["current_question"],
            "answer": state["pending_answer"]
        }

        state["qa_pairs"].append(qa)

        if state["skip_first_eval"]:
            state["skip_first_eval"] = False
        else:
            state["evaluations"].append(qa)

        state["question_count"] += 1
        state["phase"] = "NEXT"

    # NEXT
    elif state["phase"] == "NEXT":
        if state["question_count"] % 2 == 0:
            state["category_index"] = min(
                state["category_index"] + 1,
                len(CATEGORIES) - 1
            )

        state["phase"] = "ASK"

# --------------------------------------------------
# PDF
# --------------------------------------------------
def generate_pdf(chat, evaluations):
    styles = getSampleStyleSheet()
    report = []

    report.append(Paragraph("AI Viva Report", styles["Title"]))
    report.append(Spacer(1, 12))

    for m in chat:
        role = "Student" if m["role"] == "user" else "Examiner"
        report.append(Paragraph(f"<b>{role}:</b> {m['content']}", styles["Normal"]))
        report.append(Spacer(1, 6))

    total, count = 0, 0

    for e in evaluations:
        ev = evaluate_answer(e["question"], e["answer"])

        report.append(Paragraph(f"<b>Q:</b> {e['question']}", styles["Normal"]))
        report.append(Paragraph(f"<b>A:</b> {e['answer']}", styles["Normal"]))

        if "overall_score" in ev:
            total += ev["overall_score"]
            count += 1
            report.append(Paragraph(f"Score: {ev['overall_score']}", styles["Normal"]))

        report.append(Paragraph(f"{ev.get('feedback','')}", styles["Normal"]))
        report.append(Spacer(1, 10))

    if count:
        report.append(Paragraph(f"Final Avg: {total/count:.2f}", styles["Heading2"]))

    file = "viva_report.pdf"
    SimpleDocTemplate(file, pagesize=A4).build(report)
    return file

# --------------------------------------------------
# MAIN
# --------------------------------------------------
def main():
    st.set_page_config(page_title="AIVivaXaminer", page_icon="🤖")
    st.title("AIVivaXaminer")

    init_state()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    state = st.session_state.viva

    # show chat
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    # show question
    if state["phase"] == "WAIT" and state["current_question"]:
        with st.chat_message("assistant"):
            st.markdown(state["current_question"])

    user_input = st.chat_input("Enter title or answer")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})

        run_state_machine(user_input)
        run_state_machine()  # advance system

        st.rerun()

if __name__ == "__main__":
    main()
