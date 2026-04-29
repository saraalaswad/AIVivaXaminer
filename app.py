import streamlit as st
import json
import re
from dotenv import load_dotenv

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
# PROMPT
# --------------------------------------------------
prompt = PromptTemplate(
    input_variables=["message", "best_practice", "current_category"],
    template="""
You are an academic viva examiner.

Category: {current_category}

Student Answer:
{message}

Context:
{best_practice}

Ask ONE viva question.
"""
)

chain = LLMChain(llm=llm, prompt=prompt)

# --------------------------------------------------
# STATE
# --------------------------------------------------
def init_state():
    if "viva_state" not in st.session_state:
        st.session_state.viva_state = {
            "started": False,
            "evaluations": []
        }

# --------------------------------------------------
# EVALUATION
# --------------------------------------------------
def evaluate_answer(question, answer):
    prompt = f"""
Return ONLY JSON:
{{
  "overall_score": 0-100,
  "feedback": "string"
}}

Q: {question}
A: {answer}
"""

    res = eval_llm.predict(prompt)

    try:
        return json.loads(res)
    except:
        return {"error": "parse_failed", "raw": res}

# --------------------------------------------------
# PDF
# --------------------------------------------------
def generate_pdf(chat, evaluations):

    styles = getSampleStyleSheet()
    report = []

    report.append(Paragraph("AI Viva Report", styles["Title"]))
    report.append(Spacer(1, 12))

    report.append(Paragraph("Transcript", styles["Heading2"]))

    for m in chat:
        role = "Student" if m["role"] == "user" else "Examiner"
        report.append(Paragraph(f"{role}: {m['content']}", styles["Normal"]))
        report.append(Spacer(1, 6))

    report.append(Spacer(1, 20))
    report.append(Paragraph("Evaluation", styles["Heading2"]))

    total = 0
    count = 0

    for e in evaluations:
        ev = e["evaluation"]

        report.append(Paragraph(f"Q: {e['question']}", styles["Normal"]))
        report.append(Paragraph(f"A: {e['answer']}", styles["Normal"]))

        if "overall_score" in ev:
            total += ev["overall_score"]
            count += 1
            report.append(Paragraph(f"Score: {ev['overall_score']}", styles["Normal"]))

        if "feedback" in ev:
            report.append(Paragraph(f"Feedback: {ev['feedback']}", styles["Normal"]))

        report.append(Spacer(1, 10))

    if count > 0:
        avg = total / count
        report.append(Paragraph(f"Final Score: {avg:.2f}", styles["Heading2"]))

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
            "current_category": "General Understanding"
        })["text"]

        st.session_state.messages.append({"role": "assistant", "content": response})

        state = st.session_state.viva_state

        # --------------------------------------------------
        # ✅ KEY FIX: NEVER STORE FIRST INPUT FOR EVALUATION
        # --------------------------------------------------
        if not state["started"]:
            state["started"] = True
        else:
            state["evaluations"].append({
                "question": response,
                "answer": user_input,
                "evaluation": evaluate_answer(response, user_input)
            })

        st.rerun()

    # --------------------------------------------------
    # PDF
    # --------------------------------------------------
    if st.button("Generate PDF"):
        file = generate_pdf(
            st.session_state.messages,
            st.session_state.viva_state["evaluations"]
        )

        with open(file, "rb") as f:
            st.download_button("Download PDF", f, file_name="viva_report.pdf")

if __name__ == "__main__":
    main()
