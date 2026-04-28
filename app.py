import streamlit as st
import time
from datetime import datetime
from dotenv import load_dotenv
import json

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

# --------------------------------------------------
# CATEGORY CONTROL
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

def get_current_category():
    idx = st.session_state.viva_state["current_category_index"] - 1
    return CATEGORIES[idx]

def advance_category():
    st.session_state.viva_state["current_category_index"] += 1
    st.session_state.viva_state["follow_up_allowed"] = False

# --------------------------------------------------
# VIVA PROMPT
# --------------------------------------------------
PROMPT_TEMPLATE = """
You are an experienced academic professor conducting a formal undergraduate viva assessment.

CURRENT CATEGORY: {current_category}

Student Answer:
{message}

Retrieved Knowledge:
{best_practice}

TASK:
Ask ONE question only.
"""

prompt = PromptTemplate(
    input_variables=["message", "best_practice", "current_category"],
    template=PROMPT_TEMPLATE
)

chain = LLMChain(llm=llm, prompt=prompt)

# --------------------------------------------------
# 🔥 EVALUATION PROMPT
# --------------------------------------------------
EVALUATION_PROMPT = """
You are an expert academic examiner.

Evaluate the student's viva using academic rigor.

For EACH category and EACH criterion:
- Score from 0 to 10
- Give short justification

Return ONLY JSON.

Transcript:
{transcript}
"""

def evaluate_viva(chat_history):
    transcript = ""
    for msg in chat_history:
        role = "Student" if msg["role"] == "user" else "Examiner"
        transcript += f"{role}: {msg['content']}\n"

    eval_prompt = PromptTemplate(
        input_variables=["transcript"],
        template=EVALUATION_PROMPT
    )

    eval_chain = LLMChain(
        llm=ChatOpenAI(model="gpt-4-turbo", temperature=0),
        prompt=eval_prompt
    )

    result = eval_chain.run(transcript=transcript)

    try:
        return json.loads(result)
    except:
        st.error("Evaluation parsing failed")
        return None

# --------------------------------------------------
# SCORE CALCULATION
# --------------------------------------------------
def compute_scores(evaluation):
    category_scores = {}

    for category, criteria in evaluation.items():
        scores = [c["score"] for c in criteria.values()]
        category_scores[category] = round(sum(scores) / len(scores), 2)

    final_score = round(sum(category_scores.values()) / len(category_scores), 2)

    return category_scores, final_score

# --------------------------------------------------
# STATE ENGINE
# --------------------------------------------------
def init_state():
    if "viva_state" not in st.session_state:
        st.session_state.viva_state = {
            "current_category_index": 1,
            "questions_asked_total": 0,
            "questions_per_category": [0,0,0,0,0,0,0],
            "follow_up_allowed": False
        }

def update_state(user_input):
    state = st.session_state.viva_state

    state["questions_asked_total"] += 1

    idx = state["current_category_index"] - 1
    state["questions_per_category"][idx] += 1

    if state["questions_per_category"][idx] >= 2:
        if state["current_category_index"] < 7:
            advance_category()

# --------------------------------------------------
# RESPONSE GENERATION
# --------------------------------------------------
def generate_response(message):
    best_practice = retrieve_info(message)

    response = chain.run(
        message=message,
        best_practice=best_practice,
        current_category=get_current_category()
    )

    return response

# --------------------------------------------------
# 🔥 PDF WITH EVALUATION
# --------------------------------------------------
def generate_viva_pdf(chat_history, filename="viva.pdf"):
    styles = getSampleStyleSheet()
    report = []
    
    report.append(Paragraph("AI Viva Report", styles["Title"]))
    report.append(Spacer(1, 12))
    
    date_str = datetime.now().strftime("%d %B %Y, %H:%M")
    report.append(Paragraph(f"<b>Date:</b> {date_str}", styles["Normal"]))
    report.append(Spacer(1, 20))

    # Transcript
    report.append(Paragraph("<b>Transcript</b>", styles["Heading2"]))
    report.append(Spacer(1, 10))

    for msg in chat_history:
        role = "Student" if msg["role"] == "user" else "Examiner"
        text = msg["content"].replace("\n", "<br/>")

        report.append(Paragraph(f"<b>{role}:</b><br/>{text}", styles["Normal"]))
        report.append(Spacer(1, 8))

    report.append(Spacer(1, 20))

    # Evaluation
    evaluation = evaluate_viva(chat_history)

    if evaluation:
        category_scores, final_score = compute_scores(evaluation)

        report.append(Paragraph("<b>Evaluation</b>", styles["Heading2"]))
        report.append(Spacer(1, 10))

        for category, criteria in evaluation.items():
            report.append(Paragraph(f"<b>{category}</b>", styles["Heading3"]))

            for crit, data in criteria.items():
                line = f"{crit}: {data['score']}/10 — {data['reason']}"
                report.append(Paragraph(line, styles["Normal"]))

            report.append(Paragraph(f"Category Score: {category_scores[category]}", styles["Normal"]))
            report.append(Spacer(1, 10))

        report.append(Spacer(1, 20))
        report.append(Paragraph(f"<b>Final Score: {final_score}/10</b>", styles["Title"]))

    doc = SimpleDocTemplate(filename, pagesize=A4)
    doc.build(report)

    return filename

# --------------------------------------------------
# APP
# --------------------------------------------------
def main():
    st.set_page_config(page_title="AIVivaXaminer", page_icon="🤖")
    st.title("AIVivaXaminer")

    init_state()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "viva_active" not in st.session_state:
        st.session_state.viva_active = True

    # Sidebar
    with st.sidebar:
        st.header("Examiner Panel")

        if st.session_state.messages and not st.session_state.viva_active:
            if st.button("Generate PDF"):
                pdf_file = generate_viva_pdf(st.session_state.messages)

                with open(pdf_file, "rb") as f:
                    st.download_button(
                        "Download PDF",
                        f,
                        file_name="Viva_Report.pdf"
                    )

        if st.button("Stop Viva"):
            st.session_state.viva_active = False

        if st.button("Reset"):
            st.session_state.messages = []
            st.session_state.viva_active = True
            st.session_state.viva_state = {
                "current_category_index": 1,
                "questions_asked_total": 0,
                "questions_per_category": [0,0,0,0,0,0,0],
                "follow_up_allowed": False
            }
            st.rerun()

    # Chat
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    user_input = st.chat_input("Enter your research title...") if st.session_state.viva_active else None

    if user_input:
        st.chat_message("user").markdown(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})

        with st.chat_message("assistant"):
            response = generate_response(user_input)
            st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})

        update_state(user_input)

if __name__ == "__main__":
    main()
