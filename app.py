import streamlit as st
import time
from datetime import datetime
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
# STATE ENGINE (CRITICAL FIX)
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

    # simple follow-up trigger
    weak_words = ["not sure", "don't know", "unclear", "maybe"]

    state["follow_up_allowed"] = any(w in user_input.lower() for w in weak_words)

    # FORCE CATEGORY SHIFT (max 2 questions per category)
    if state["questions_per_category"][idx] >= 2:
        if state["current_category_index"] < 7:
            advance_category()

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
# PDF
# --------------------------------------------------
def generate_viva_pdf(chat_history, filename="viva.pdf"):
    styles = getSampleStyleSheet()
    report = []
    
    date_str = datetime.now().strftime("%d %B %Y, %H:%M")
    report.append(Paragraph("AI Viva Transcript", styles["Title"]))
    report.append(Spacer(1, 12))

    for msg in chat_history:
        role = "Student" if msg["role"] == "user" else "Examiner"
        text = msg["content"].replace("\n", "<br/>")

        report.append(Paragraph(f"<b>{role}:</b><br/>{text}", styles["Normal"]))
        report.append(Spacer(1, 10))

    doc = SimpleDocTemplate(filename, pagesize=A4)
    doc.build(report)

    return filename

# --------------------------------------------------
# APP
# --------------------------------------------------
def main():
    st.set_page_config(page_title="AIVivaXaminer", page_icon="🎓")

    st.title("🎓 AIVivaXaminer")

    init_state()

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
                if st.button("Generate PDF Transcript"):
                    pdf_file = generate_viva_pdf(
                        st.session_state.messages,
                        filename="AIViva_Transcript.pdf"
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

        st.session_state.messages.append({"role": "assistant", "content": response})

        # 🔥 CRITICAL FIX: UPDATE STATE HERE
        update_state(user_input)


if __name__ == "__main__":
    main()
