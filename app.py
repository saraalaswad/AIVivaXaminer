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
# RUBRIC
# --------------------------------------------------
EVALUATION_FRAMEWORK = {
    "Problem Definition": ["Coherence","Relevance","Completeness","Engagement","Accuracy","Creativity","Fluency","Clarity","Descriptiveness","Informativeness"],
    "Literature Search": ["Coherence","Relevance","Completeness","Accuracy","Creativity","Clarity","Descriptiveness","Informativeness"],
    "Solution Design": ["Coherence","Relevance","Completeness","Accuracy","Creativity","Fluency","Clarity","Descriptiveness","Informativeness"],
    "Result & Analysis": ["Coherence","Relevance","Completeness","Accuracy","Creativity","Fluency","Clarity","Descriptiveness","Informativeness"],
    "Implementation / Product": ["Seamless functionality","Real-world significance","User engagement","Functionality"],
    "References & Citation": ["Organized structure","Complete citation","Accurate referencing","Clarity in citation format","Descriptive citations","Informative references"],
    "Teamwork": ["Active participation","Clear roles assigned","Professional collaboration","Transparent roles","Keeps team informed"],
    "Documentation and Format": ["Organized structure","Full coverage of content","Accurate documentation","Unique presentation style","Smooth report structure","Clear writing","Rich content","Informs the audience"],
    "Organization & Delivery": ["Logical and engaging delivery","Clear connection to objectives","Fully developed presentation","Audience interaction","Accurate communication","Creative visuals","Smooth delivery","Clear communication","Descriptive visuals","Clear and valuable content"]
}

# --------------------------------------------------
# PROMPT (VIVA QUESTION)
# --------------------------------------------------
PROMPT_TEMPLATE = """
You are an experienced academic professor conducting a viva.

CURRENT CATEGORY: {current_category}

Student Answer:
{message}

Retrieved Knowledge:
{best_practice}

Ask ONE question only.
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
# FINAL EVALUATION (USED ONLY IN PDF)
# --------------------------------------------------
def evaluate_answer(question, answer):
    prompt = f"""
You are a PhD viva examiner.

Return ONLY valid JSON:

{{
  "overall_score": 0-100,
  "feedback": "string"
}}

RUBRIC:
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
        return {"error": "parse_failed", "raw": response}

# --------------------------------------------------
# STATE
# --------------------------------------------------
def init_state():
    if "viva_state" not in st.session_state:
        st.session_state.viva_state = {
            "current_category_index": 1,
            "evaluations": []
        }

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
# BATCH EVALUATION (KEY UPGRADE)
# --------------------------------------------------
def batch_evaluate(evaluations):
    results = []

    for item in evaluations:
        q = item["question"]
        a = item["answer"]

        ev = evaluate_answer(q, a)

        results.append({
            **item,
            "evaluation": ev
        })

    return results

# --------------------------------------------------
# PDF GENERATION
# --------------------------------------------------
def generate_pdf(chat, evaluations):

    evaluated = batch_evaluate(evaluations)

    styles = getSampleStyleSheet()
    report = []

    report.append(Paragraph("AI Viva Report", styles["Title"]))
    report.append(Spacer(1, 12))

    # -----------------------
    # Transcript
    # -----------------------
    report.append(Paragraph("Transcript", styles["Heading2"]))
    report.append(Spacer(1, 10))

    for m in chat:
        role = "Student" if m["role"] == "user" else "Examiner"
        report.append(Paragraph(f"<b>{role}:</b> {m['content']}", styles["Normal"]))
        report.append(Spacer(1, 6))

    # -----------------------
    # Evaluation
    # -----------------------
    report.append(Spacer(1, 20))
    report.append(Paragraph("Evaluation", styles["Heading2"]))

    total = 0
    count = 0

    for e in evaluated:
        ev = e["evaluation"]

        report.append(Paragraph(f"<b>Q:</b> {e['question']}", styles["Normal"]))
        report.append(Paragraph(f"<b>A:</b> {e['answer']}", styles["Normal"]))

        if "overall_score" in ev:
            score = ev["overall_score"]
            total += score
            count += 1
            report.append(Paragraph(f"Score: {score}", styles["Normal"]))

        if "feedback" in ev:
            report.append(Paragraph(f"Feedback: {ev['feedback']}", styles["Normal"]))

        report.append(Spacer(1, 10))

    # -----------------------
    # FINAL SCORE (PhD style summary)
    # -----------------------
    if count > 0:
        avg = total / count
        report.append(Paragraph(f"<b>Final Average Score: {avg:.2f}</b>", styles["Heading2"]))

    file = "viva_report.pdf"
    doc = SimpleDocTemplate(file, pagesize=A4)
    doc.build(report)

    return file

# --------------------------------------------------
# APP
# --------------------------------------------------
def main():
    st.title("🎓 AI Viva Examiner (Batch Evaluation Mode)")

    init_state()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # -----------------------
    # CHAT DISPLAY
    # -----------------------
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    user_input = st.chat_input("Enter your answer...")

    # -----------------------
    # VIVA FLOW (NO EVALUATION HERE)
    # -----------------------
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})

        best = retrieve_info(user_input)

        response = chain.invoke({
            "message": user_input,
            "best_practice": best,
            "current_category": get_current_category()
        })["text"]

        st.session_state.messages.append({"role": "assistant", "content": response})

        # ONLY STORE (NO EVALUATION)
        st.session_state.viva_state["evaluations"].append({
            "question": user_input,
            "answer": response
        })

        st.rerun()

    # -----------------------
    # PDF TRIGGERS EVALUATION
    # -----------------------
    if st.button("Generate PDF"):
        file = generate_pdf(
            st.session_state.messages,
            st.session_state.viva_state["evaluations"]
        )

        with open(file, "rb") as f:
            st.download_button("Download PDF", f, file_name="viva_report.pdf")

if __name__ == "__main__":
    main()
