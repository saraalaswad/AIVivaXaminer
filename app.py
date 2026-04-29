import streamlit as st
import time
from datetime import datetime
from dotenv import load_dotenv
import json
import re
import os

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
# VECTOR DB
# --------------------------------------------------
@st.cache_resource
def initialize_vector_db():
    if not os.path.exists("ts_response.csv"):
        st.error("CSV file not found: ts_response.csv")
        st.stop()

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
# RUBRIC
# --------------------------------------------------
EVALUATION_FRAMEWORK = {
    "Problem Definition": ["Coherence","Relevance","Completeness","Engagement","Accuracy","Creativity","Fluency","Clarity","Descriptiveness","Informativeness"],
    "Literature Search": ["Coherence","Relevance","Completeness","Accuracy","Creativity","Clarity","Descriptiveness","Informativeness"],
    "Solution Design": ["Coherence","Relevance","Completeness","Accuracy","Creativity","Fluency","Clarity","Descriptiveness","Informativeness"],
    "Result & Analysis": ["Coherence","Relevance","Completeness","Accuracy","Creativity","Fluency","Clarity","Descriptiveness","Informativeness"],
    "Implementation / Product": ["Functionality","Real-world impact","User engagement","Performance"],
    "References & Citation": ["Structure","Completeness","Accuracy","Clarity","Formatting"],
    "Teamwork": ["Participation","Roles","Collaboration","Communication","Transparency"],
    "Documentation and Format": ["Structure","Coverage","Accuracy","Clarity","Presentation"],
    "Organization & Delivery": ["Delivery","Clarity","Engagement","Communication","Creativity"]
}

# --------------------------------------------------
# PROMPT
# --------------------------------------------------
PROMPT_TEMPLATE = """
You are an academic viva examiner.

Category: {current_category}

Student Answer:
{message}

Context:
{best_practice}

Ask ONE clear follow-up question only.
"""

prompt = PromptTemplate(
    input_variables=["message", "best_practice", "current_category"],
    template=PROMPT_TEMPLATE
)

chain = LLMChain(llm=llm, prompt=prompt)

# --------------------------------------------------
# CLEAN JSON
# --------------------------------------------------
def clean_json_output(text):
    text = re.sub(r"```json|```", "", text)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    return match.group(0) if match else text

# --------------------------------------------------
# EVALUATION
# --------------------------------------------------
def evaluate_answer(question, answer):
    prompt = f"""
Return ONLY valid JSON:

{{
  "overall_score": 0,
  "feedback": "",
  "scores": {{}}
}}

Rubric:
{json.dumps(EVALUATION_FRAMEWORK, indent=2)}

Question:
{question}

Answer:
{answer}
"""

    response = eval_llm.predict(prompt)
    cleaned = clean_json_output(response)

    try:
        return json.loads(cleaned)
    except:
        return {"error": "Parsing failed", "raw": response}

# --------------------------------------------------
# STATE (FIXED)
# --------------------------------------------------
def init_state():
    if "viva_state" not in st.session_state:
        st.session_state.viva_state = {}

    if "current_category_index" not in st.session_state.viva_state:
        st.session_state.viva_state["current_category_index"] = 1

    if "evaluations" not in st.session_state.viva_state:
        st.session_state.viva_state["evaluations"] = []

    if "skip_first_evaluation" not in st.session_state.viva_state:
        st.session_state.viva_state["skip_first_evaluation"] = True

# --------------------------------------------------
# CATEGORY
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
    return CATEGORIES[st.session_state.viva_state["current_category_index"] - 1]

# --------------------------------------------------
# PDF GENERATION (FIXED + FULL EVAL INCLUDED)
# --------------------------------------------------
def generate_pdf(chat, evaluations):
    styles = getSampleStyleSheet()
    report = []

    report.append(Paragraph("AI Viva Report", styles["Title"]))
    report.append(Spacer(1, 12))

    # --------------------------
    # TRANSCRIPT
    # --------------------------
    report.append(Paragraph("Transcript", styles["Heading2"]))
    report.append(Spacer(1, 10))

    for m in chat:
        role = "Student" if m["role"] == "user" else "Examiner"
        report.append(Paragraph(f"<b>{role}:</b> {m['content']}", styles["Normal"]))
        report.append(Spacer(1, 6))

    # --------------------------
    # EVALUATION
    # --------------------------
    report.append(Spacer(1, 20))
    report.append(Paragraph("Evaluation", styles["Heading2"]))
    report.append(Spacer(1, 10))

    for idx, e in enumerate(evaluations, 1):
        report.append(Paragraph(f"<b>Q{idx}:</b> {e.get('question','')}", styles["Normal"]))
        report.append(Paragraph(f"<b>A{idx}:</b> {e.get('answer','')}", styles["Normal"]))
        report.append(Spacer(1, 5))

        ev = e.get("evaluation", {}) or {}

        if ev.get("skipped"):
            report.append(Paragraph("Evaluation: Skipped (first input)", styles["Normal"]))

        elif "error" in ev:
            report.append(Paragraph("Evaluation: Failed to parse model output", styles["Normal"]))
            report.append(Paragraph(f"Raw: {ev.get('raw','')}", styles["Normal"]))

        else:
            if "overall_score" in ev:
                report.append(Paragraph(f"<b>Score:</b> {ev['overall_score']}/100", styles["Normal"]))

            if "feedback" in ev:
                report.append(Paragraph(f"<b>Feedback:</b> {ev['feedback']}", styles["Normal"]))

            if "scores" in ev and isinstance(ev["scores"], dict):
                report.append(Paragraph("<b>Rubric Breakdown:</b>", styles["Normal"]))
                for k, v in ev["scores"].items():
                    report.append(Paragraph(f"• {k}: {v}", styles["Normal"]))

        report.append(Spacer(1, 12))

    file = "viva_report.pdf"
    doc = SimpleDocTemplate(file, pagesize=A4)
    doc.build(report)

    return file

# --------------------------------------------------
# APP
# --------------------------------------------------
def main():
    st.title("🎓 AI Viva Examiner")

    init_state()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    user_input = st.chat_input("Enter your answer...")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})

        best = retrieve_info(user_input)

        response = chain.invoke({
            "message": user_input,
            "best_practice": best,
            "current_category": get_current_category()
        })

        response_text = response["text"] if isinstance(response, dict) else response

        st.session_state.messages.append({"role": "assistant", "content": response_text})

        # --------------------------
        # SKIP FIRST EVALUATION
        # --------------------------
        if st.session_state.viva_state.get("skip_first_evaluation", True):
            evaluation = {"skipped": True}
            st.session_state.viva_state["skip_first_evaluation"] = False
        else:
            evaluation = evaluate_answer(user_input, response_text)

        st.session_state.viva_state["evaluations"].append({
            "question": user_input,
            "answer": response_text,
            "evaluation": evaluation
        })

        st.rerun()

    if st.button("Generate PDF"):
        file = generate_pdf(
            st.session_state.messages,
            st.session_state.viva_state["evaluations"]
        )

        with open(file, "rb") as f:
            st.download_button("Download PDF", f, file_name="viva_report.pdf")

if __name__ == "__main__":
    main()
