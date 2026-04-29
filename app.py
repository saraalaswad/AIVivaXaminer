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
# RUBRIC
# --------------------------------------------------
EVALUATION_FRAMEWORK = {
    "Problem Definition": ["logical structure of the problem","pertinence to objectives","thoroughness of definition","N/A","correct identification of issues","innovative problem framing","N/A","clear articulation of problem","detailed background","provides context"],
    "Literature Search": ["logical flow of methods","clear connection to problem","comprehensive search","N/A","precision of sources","unique and novel sources","N/A","clear explanation of research", "detailed context and analysis", "explains literature thoroughly"],
    "Solution Design": ["logical flow of methods","applicability of methods","comprehensive methodology","N/A","precision of methods","novel approaches","N/A","clear description of steps","detailed steps", "explains methods thoroughly"],
    "Result & Analysis": ["logical interpretation of results","results aligned with objectives","depth of analysis","N/A","correct results and analysis","innovative insights","smooth presentation of analysis","clear explanation of results","detailed analysis", "insightful conclusions"],
    "Implementation / Product": ["Seamless functionality","Real-world significance","User engagement","Functionality", "original solution", "N/A", "intuitive design", "visual and functional detail", "educates the user"],
    "References & Citation": ["Organized structure", "N/A", "Complete citation", "N/A", "Accurate referencing", "N/A", "N/A", "Clarity in citation format","Descriptive citations","Informative references"],
    "Teamwork": ["N/A", "N/A", "N/A", "Active participation","Clear roles assigned", "innovative team strategies", "Professional collaboration","Transparent roles", "explicit team documentation", "Keeps team informed"],
    "Documentation and Format": ["Organized structure", "N/A", "Full coverage of content", "reader engagement", "Accurate documentation","Unique presentation style","Smooth report structure","Clear writing","Rich content","Informs the audience"],
    "Organization & Delivery": ["Logical and engaging delivery","Clear connection to objectives","Fully developed presentation","Audience interaction","Accurate communication","Creative visuals","Smooth delivery","Clear communication","Descriptive visuals","Clear and valuable content"]
}

# --------------------------------------------------
# CATEGORY CONTROL (PYTHON IS NOW THE BOSS)
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
# PROMPT (LLM HAS NO CONTROL OVER FLOW)
# --------------------------------------------------
PROMPT_TEMPLATE = """
You are an experienced academic professor conducting a formal undergraduate viva assessment. Your role is to evaluate the student’s understanding of their research project through a structured, interactive oral examination.

The student will first share their research title. Based on this title, the student’s message, and established academic best practices, you will generate appropriate, rigorous, and supportive viva-style questions.
Maintain a professional, supportive yet challenging tone, similar to that used by experienced viva examiners.


----------------------------------------
📊 CURRENT CATEGORY (DO NOT CHANGE)
----------------------------------------
{current_category}

----------------------------------------
RULES
----------------------------------------
• Ask ONLY ONE question
• After each question, pause and wait for the student’s full response.
• Then provide brief, constructive academic feedback or discussion before moving to the next question.
• Your goal is to assess depth of understanding, critical thinking, and ability to justify decisions, while guiding the student to refine and articulate their ideas clearly.
• Do NOT repeat previous topics
• Do NOT ask overview/problem unless category is General Understanding
• Keep tone formal and academic
• Do NOT add explanations

----------------------------------------
CONTEXT
----------------------------------------
Student Answer:
{message}

Retrieved Knowledge:
{best_practice}

----------------------------------------
TASK
----------------------------------------
Generate ONE viva question strictly for the current category.
"""

prompt = PromptTemplate(
    input_variables=[
        "message",
        "best_practice",
        "current_category"
    ],
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
You are an undergraduate viva examiner.

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
# STATE ENGINE (CRITICAL FIX)
# --------------------------------------------------
def init_state():
    if "viva_state" not in st.session_state:
        st.session_state.viva_state = {
            "current_category_index": 1,
            "questions_asked_total": 0,
            "questions_per_category": [0,0,0,0,0,0,0],
            "follow_up_allowed": False,
            "evaluations": [],
            "skip_first": True   # ✅ IMPORTANT FIX
        }

def update_state(user_input):
    state = st.session_state.viva_state

    state["questions_asked_total"] += 1

    idx = state["current_category_index"] - 1
    state["questions_per_category"][idx] += 1

    # simple follow-up trigger
    weak_words = ["not sure", "don't know", "unclear", "maybe"]

    state["follow_up_allowed"] = any(w in user_input.lower() for w in weak_words)

    # FORCE CATEGORY SHIFT (max 2 questions per category)
    if state["questions_per_category"][idx] >= 2:
        if state["current_category_index"] < 7:
            advance_category()

def ensure_state():
    if "viva_state" not in st.session_state:
        st.session_state.viva_state = {}

    if "evaluations" not in st.session_state.viva_state:
        st.session_state.viva_state["evaluations"] = []

    if "skip_first" not in st.session_state.viva_state:
        st.session_state.viva_state["skip_first"] = True

# --------------------------------------------------
# RESPONSE GENERATION
# --------------------------------------------------
def generate_response(message):
    best_practice = retrieve_info(message)
    state = st.session_state.viva_state

    response = chain.run(
        message=message,
        best_practice=best_practice,
        current_category=get_current_category()
    )

    return response

# --------------------------------------------------
# BATCH EVALUATION
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

    # Transcript
    report.append(Paragraph("Transcript", styles["Heading2"]))
    report.append(Spacer(1, 10))

    for m in chat:
        role = "Student" if m["role"] == "user" else "Examiner"
        report.append(Paragraph(f"<b>{role}:</b> {m['content']}", styles["Normal"]))
        report.append(Spacer(1, 6))

    # Evaluation
    report.append(Spacer(1, 20))
    report.append(Paragraph("Evaluation", styles["Heading2"]))

    total = 0
    count = 0

    for e in evaluated:
        ev = e["evaluation"]

        report.append(Paragraph(f"<b>Q:</b> {e['question']}", styles["Normal"]))
        report.append(Paragraph(f"<b>A:</b> {e['answer']}", styles["Normal"]))
        report.append(Spacer(1, 5))

        if "overall_score" in ev:
            score = ev["overall_score"]
            total += score
            count += 1
            report.append(Paragraph(f"Score: {score}", styles["Normal"]))

        if "feedback" in ev:
            report.append(Paragraph(f"Feedback: {ev['feedback']}", styles["Normal"]))

        report.append(Spacer(1, 10))

    if count > 0:
        avg = total / count
        report.append(Paragraph(f"<b>Final Average Score: {avg:.2f}</b>", styles["Heading2"]))

    pdf_file = "viva_report.pdf"
    doc = SimpleDocTemplate(pdf_file, pagesize=A4)
    doc.build(report)

    return pdf_file

# --------------------------------------------------
# APP
# --------------------------------------------------
def main():
    st.set_page_config(page_title="AIVivaXaminer", page_icon="🤖")

    st.title("AIVivaXaminer")

    init_state()
    ensure_state()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "viva_active" not in st.session_state:
        st.session_state.viva_active = True

    if "examiner_logged_in" not in st.session_state:
        st.session_state.examiner_logged_in = False

    # ---------------- SIDEBAR ----------------
    with st.sidebar:
        st.header("🎛️ Examiner Panel")

        # ---------------- LOGIN ----------------
        if not st.session_state.examiner_logged_in:
            st.subheader("🔐 Login")

            username = st.text_input("Username", key="exam_user")
            password = st.text_input("Password", type="password", key="exam_pass")

            if st.button("Login"):
                if username == EXAMINER_USERNAME and password == EXAMINER_PASSWORD:
                    st.session_state.examiner_logged_in = True
                    st.success("Logged in successfully")
                    st.rerun()
                else:
                    st.error("Invalid credentials")

            st.info("Login required for examiner controls.")

        else:
            st.success("Examiner Mode Active")

            # ---------------- Category ----------------
            st.subheader("📊 Current Category")
            st.write(get_current_category())
            st.divider()
            
            # ---------------- VIVA CONTROL ----------------
            st.subheader("🎛️ Viva Control")

            if st.session_state.viva_active:
                if st.button("🛑 Stop Viva"):
                    st.session_state.viva_active = False
                    st.success("Viva stopped")
                    st.rerun()
            else:
                st.info("Viva session ended")

            st.divider()

            # ---------------- EXPORT ----------------
            st.subheader("📄 Export")

            if st.session_state.messages and not st.session_state.viva_active:
                if st.button("Generate PDF"):
                    pdf_file = generate_pdf(
                        st.session_state.messages,
                        st.session_state.viva_state["evaluations"]
                    )

                    with open(pdf_file, "rb") as f:
                        st.download_button(
                            "⬇️ Download PDF",
                            f,
                            file_name="AIViva_Transcript.pdf",
                            mime="application/pdf"
                        )
            else:
                st.caption("Stop viva first to enable PDF")

            st.divider()
            # ---------------- SESSION ----------------
            st.subheader("⚙️ Session")

            if st.button("🔄 Reset Session"):
                st.session_state.viva_state = {
                    "current_category_index": 1,
                    "questions_asked_total": 0,
                    "questions_per_category": [0,0,0,0,0,0,0],
                    "follow_up_allowed": False
                }
                st.session_state.messages = []
                st.session_state.viva_active = True
                st.success("Session reset")
                st.rerun()

            if st.button("🚪 Logout"):
                st.session_state.examiner_logged_in = False
                st.rerun()


            

    # ---------------- CHAT HISTORY ----------------
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    user_input = st.chat_input("Enter your research title ...") if st.session_state.viva_active else None

    if user_input:
        st.chat_message("user").markdown(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})

        with st.chat_message("assistant"):
            placeholder = st.empty()

            response = generate_response(user_input)

            text = ""
            for w in response.split():
                text += w + " "
                time.sleep(0.01)
                placeholder.markdown(text + "▌")

            placeholder.markdown(text)

        # --------------------------------------------------
        # STORE Q/A FIRST (ALWAYS SAFE)
        # --------------------------------------------------
        qa_pair = {
            "question": response,
            "answer": user_input
        }
        
        st.session_state.messages.append({"role": "assistant", "content": response})
        
        # --------------------------------------------------
        # UPDATE STATE FIRST (BEFORE RERUN)
        # --------------------------------------------------
        update_state(user_input)
        
        # --------------------------------------------------
        # SKIP FIRST EVALUATION ONLY
        # --------------------------------------------------
        if st.session_state.viva_state.get("skip_first", True):
            st.session_state.viva_state["skip_first"] = False
        else:
            st.session_state.viva_state.setdefault("evaluations", []).append(qa_pair)
        
        # --------------------------------------------------
        # NOW SAFE TO RERUN
        # --------------------------------------------------
        st.rerun()


if __name__ == "__main__":
    main()
