import streamlit as st
import time
import json
import tempfile
import pandas as pd
import altair as alt
from dotenv import load_dotenv
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4

load_dotenv()

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DEFAULT_MAX_QUESTIONS = 10
MIN_COVERAGE = {"General": 2, "Technical": 2, "Critical": 1, "Future": 1}

# ─────────────────────────────────────────────
# VECTOR STORE
# ─────────────────────────────────────────────
loader = CSVLoader(file_path="ts_response.csv")
documents = loader.load()
embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(documents, embeddings)

def retrieve_info(query):
    docs = db.similarity_search(query, k=3)
    return "\n".join([d.page_content for d in docs])

# ─────────────────────────────────────────────
# LLM
# ─────────────────────────────────────────────
llm = ChatOpenAI(model="gpt-4-turbo", temperature=0.4)

# ─────────────────────────────────────────────
# PROMPTS
# ─────────────────────────────────────────────
VIVA_PROMPT = """
You are AIVivaXaminer, an AI-based undergraduate viva examiner.

STRICT RULES:
- Ask ONE new question only.
- NEVER repeat any previous question.
- Output ONLY the question.
- No explanations.



PREVIOUS QUESTIONS:
{question_history}

STUDENT INPUT:
{message}

BEST PRACTICE CONTEXT:
{best_practice}

If no new meaningful questions remain, output:
FINAL_EVALUATION_READY
"""

CATEGORY_PROMPT = """
Classify the following question into EXACTLY ONE category:
General, Technical, Critical, Domain, Future

Question:
{question}

Return ONLY one word.
"""

SCORING_PROMPT = """
Evaluate the student's response using the rubric.
Return ONLY valid JSON.

Rubric dimensions:
Conceptual, Methodological, Technical, Critical, Communication
Scale: 0–4

Student Response:
{student_response}

JSON:
{
 "Conceptual": X,
 "Methodological": X,
 "Technical": X,
 "Critical": X,
 "Communication": X
}
"""

viva_chain = LLMChain(
    llm=llm,
    prompt=PromptTemplate(
        input_variables=["message", "best_practice", "question_history"],
        template=VIVA_PROMPT
    )
)

category_chain = LLMChain(
    llm=llm,
    prompt=PromptTemplate(
        input_variables=["question"],
        template=CATEGORY_PROMPT
    )
)

scoring_chain = LLMChain(
    llm=llm,
    prompt=PromptTemplate(
        input_variables=["student_response"],
        template=SCORING_PROMPT
    )
)

# ─────────────────────────────────────────────
# PDF GENERATION
# ─────────────────────────────────────────────
def generate_viva_pdf(questions, responses, averages, overall, recommendation):
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph("AIVivaXaminer – Final Viva Report", styles["Title"]))
    elements.append(Spacer(1, 12))
    elements.append(Paragraph(f"<b>Overall Score:</b> {overall}", styles["Normal"]))
    elements.append(Paragraph(f"<b>Final Recommendation:</b> {recommendation}", styles["Normal"]))
    elements.append(Spacer(1, 12))

    elements.append(Paragraph("<b>Dimension Averages</b>", styles["Heading2"]))
    for k, v in averages.items():
        elements.append(Paragraph(f"{k}: {v}", styles["Normal"]))

    elements.append(Spacer(1, 12))
    elements.append(Paragraph("<b>Viva Questions & Responses</b>", styles["Heading2"]))

    for i, (q, r) in enumerate(zip(questions, responses), 1):
        elements.append(Spacer(1, 8))
        elements.append(Paragraph(f"<b>Q{i}:</b> {q}", styles["Normal"]))
        elements.append(Paragraph(f"<b>Response:</b> {r}", styles["Normal"]))

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    doc = SimpleDocTemplate(tmp.name, pagesize=A4)
    doc.build(elements)
    return tmp.name

# ─────────────────────────────────────────────
# FINAL SCORING
# ─────────────────────────────────────────────
def compute_final_result(scores):
    averages = {}
    for dim, vals in scores.items():
        if vals:
            averages[dim] = round(sum(vals)/len(vals), 2)
    overall = round(sum(averages.values()) / len(averages), 2) if averages else 0

    if overall >= 3.5:
        rec = "Pass"
    elif overall >= 3.0:
        rec = "Pass with Minor Revisions"
    elif overall >= 2.5:
        rec = "Borderline"
    else:
        rec = "Fail"

    return averages, overall, rec

# ─────────────────────────────────────────────
# STREAMLIT APP
# ─────────────────────────────────────────────
def main():
    st.set_page_config("AIVivaXaminer", "🤖")
    st.title("🤖 AIVivaXaminer")

    # Examiner panel
    with st.sidebar:
        st.header("🎛 Examiner Control Panel")
        max_q = st.slider("Maximum Questions", 5, 15, DEFAULT_MAX_QUESTIONS)
        force_stop = st.button("🛑 Force Stop Viva")

    # Session state initialization
    defaults = {
        "messages": [], "question_history": [], "student_responses": [], "question_count": 0,
        "viva_completed": False,
        "category_counter": {"General":0,"Technical":0,"Critical":0,"Domain":0,"Future":0},
        "scores": {"Conceptual":[], "Methodological":[], "Technical":[], "Critical":[], "Communication":[]}
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

    # Force stop
    if force_stop:
        st.session_state.viva_completed = True

    

    # Display analytics dashboard (examiner only)
    with st.sidebar:
        st.header("📊 Viva Analytics Dashboard")
        total_q = st.session_state.question_count
        st.write(f"Questions Asked: {total_q} / {max_q}")

        # Category coverage
        st.subheader("Category Coverage")
        for cat, count in st.session_state.category_counter.items():
            st.progress(min(count / MIN_COVERAGE.get(cat, max_q), 1.0))
            st.write(f"{cat}: {count}")

        # Average scores
        st.subheader("Rubric Dimension Averages")
        averages = {}
        for dim, vals in st.session_state.scores.items():
            avg = round(sum(vals) / len(vals), 2) if vals else 0
            averages[dim] = avg
            st.write(f"{dim}: {avg}")

        # Viva status
        st.subheader("Viva Status")
        status = "Completed ✅" if st.session_state.viva_completed else "Ongoing 🟢"
        st.write(status)

        # Score chart
        df_scores = pd.DataFrame([{"Dimension": k, "Average": v} for k, v in averages.items()])
        chart = alt.Chart(df_scores).mark_bar().encode(x='Dimension', y='Average', color='Dimension')
        st.altair_chart(chart, use_container_width=True)

    # If viva completed, generate PDF
    if st.session_state.viva_completed:
        averages, overall, rec = compute_final_result(st.session_state.scores)
        pdf = generate_viva_pdf(
            st.session_state.question_history,
            st.session_state.student_responses,
            averages, overall, rec
        )
        st.markdown(f"### 🧾 Final Recommendation: **{rec}**")
        with open(pdf, "rb") as f:
            st.download_button("📄 Download Viva Report (PDF)", f, "AIViva_Report.pdf")
        st.stop()

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    
    # Student input
    if user_input := st.chat_input("Enter your research title"):
        st.session_state.messages.append({"role":"user","content":user_input})
        st.session_state.student_responses.append(user_input)

        # Score response
        try:
            s = json.loads(scoring_chain.run(student_response=user_input))
            for k in st.session_state.scores:
                st.session_state.scores[k].append(s.get(k,0))
        except:
            pass

        # Retrieve best practice context
        best = retrieve_info(user_input)
        history = "\n".join(st.session_state.question_history)

        # Generate next question
        question = viva_chain.run(
            message=user_input,
            best_practice=best,
            question_history=history
        ).strip()

        if question == "FINAL_EVALUATION_READY":
            st.session_state.viva_completed = True
            st.stop()

        # Deduplication
        if question in st.session_state.question_history:
            st.stop()

        # Category detection
        cat = category_chain.run(question=question).strip()
        if cat in st.session_state.category_counter:
            st.session_state.category_counter[cat] += 1

        # Update question history and counter
        st.session_state.question_history.append(question)
        st.session_state.question_count += 1

        # Stopping rules: max questions OR category coverage
        if st.session_state.question_count >= max_q or all(
            st.session_state.category_counter[k] >= v for k,v in MIN_COVERAGE.items()
        ):
            st.session_state.viva_completed = True

        # Display question
        with st.chat_message("assistant"):
            st.markdown(question)
        st.session_state.messages.append({"role":"assistant","content":question})

if __name__ == "__main__":
    main()

