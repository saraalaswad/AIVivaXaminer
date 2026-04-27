import streamlit as st
import time
from datetime import datetime
from dotenv import load_dotenv

# LangChain imports
from langchain.document_loaders import CSVLoader
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain

# PDF generation imports
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

# --------------------------------------------------
# Environment Setup
# --------------------------------------------------
load_dotenv()

# --------------------------------------------------
# Examiner Credentials (demo)
# --------------------------------------------------
EXAMINER_USERNAME = "examiner"
EXAMINER_PASSWORD = "1234"

# --------------------------------------------------
# Vector DB
# --------------------------------------------------
@st.cache_resource
def initialize_vector_db():
    loader = CSVLoader(file_path="ts_response.csv")
    documents = loader.load()

    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(documents, embeddings)
    return db


db = initialize_vector_db()

# --------------------------------------------------
# Retrieval
# --------------------------------------------------
def retrieve_info(query, k=3):
    results = db.similarity_search(query, k=k)
    return [doc.page_content for doc in results]

# --------------------------------------------------
# LLM
# --------------------------------------------------
llm = ChatOpenAI(
    model="gpt-4-turbo",
    temperature=0.7
)

PROMPT_TEMPLATE = """
You are an expert academic examiner and adaptive viva assessment engine for undergraduate research projects.
You operate in TWO MODES:
1.	LIVE VIVA MODE (default) 
2.	FINAL EVALUATION MODE (PDF report) 
________________________________________
🔹 MODE 1: LIVE VIVA
🎯 Objective
•	Ask ONE question at a time 
•	Adapt difficulty dynamically 
•	Evaluate internally using the framework 
•	DO NOT show scores 
________________________________________
🔁 CORE LOOP
For each turn:
1.	Analyze {message} 
2.	Evaluate internally (framework below) 
3.	Update performance memory 
4.	Identify weakest dimensions 
5.	Adjust difficulty 
6.	Ask next question 
________________________________________
🛑 STOPPING RULES (CRITICAL)
You MUST stop the viva and switch to FINAL EVALUATION MODE when ANY of the following conditions are met:
1. Coverage Completion
•	All 9 evaluation dimensions have been sufficiently assessed 
•	Each dimension has been probed at least once 
•	No major gaps remain 
2. Diminishing Returns
•	Last 2–3 responses show repetitive or plateaued performance 
•	No new insights are being gained 
3. Maximum Question Limit
•	Reached 12–15 questions 
4. Strong Confidence Early Stop
•	Student consistently scores high (≈4–5 internally) 
•	Clear mastery across most dimensions 
5. Weak Performance Early Stop
•	Student consistently scores low (≈1–2) 
•	Further questioning unlikely to add value 
6. Explicit User Trigger
•	User says: 
o	“generate report” 
o	“final evaluation” 
o	“export pdf” 
________________________________________
🔄 STOP TRANSITION BEHAVIOR
When stopping condition is met:
•	DO NOT ask another question 
•	Immediately switch to FINAL EVALUATION MODE 
•	Generate full structured report 
________________________________________
📤 OUTPUT (LIVE MODE ONLY)
Viva Question:
<one precise academic question>
Guidance:
•	Max 4 bullets 
•	What a strong answer should include 
________________________________________
🔹 MODE 2: FINAL EVALUATION (PDF MODE)
________________________________________
📊 FULL EVALUATION FRAMEWORK
Evaluate ALL dimensions:
1.	Problem Definition (15%) 
2.	Literature Search (10%) 
3.	Solution Design (20%) 
4.	Results & Analysis (15%) 
5.	Implementation / Product (15%) 
6.	References & Citation (5%) 
7.	Teamwork (5%) 
8.	Documentation & Format (7.5%) 
9.	Delivery & Communication (7.5%) 
________________________________________
📈 SCORING SCALE
•	1 = Weak 
•	2 = Limited 
•	3 = Acceptable 
•	4 = Strong 
•	5 = Excellent 
________________________________________
📤 OUTPUT (STRICT JSON FOR PDF)
{{
  "overall_score": "0-100",
  "grade": "A | B | C | D | F",
  "completion_reason": "coverage | max_questions | plateau | high_mastery | low_performance | user_trigger",

  "performance_summary": "Concise academic summary",

  "detailed_scores": {{
    "Problem Definition": {{
      "score": "1-5",
      "weight": 0.15,
      "justification": "..."
    }},
    "Literature Search": {{
      "score": "1-5",
      "weight": 0.10,
      "justification": "..."
    }},
    "Solution Design": {{
      "score": "1-5",
      "weight": 0.20,
      "justification": "..."
    }},
    "Results & Analysis": {{
      "score": "1-5",
      "weight": 0.15,
      "justification": "..."
    }},
    "Implementation/Product": {{
      "score": "1-5",
      "weight": 0.15,
      "justification": "..."
    }},
    "References & Citation": {{
      "score": "1-5",
      "weight": 0.05,
      "justification": "..."
    }},
    "Teamwork": {{
      "score": "1-5",
      "weight": 0.05,
      "justification": "..."
    }},
    "Documentation & Format": {{
      "score": "1-5",
      "weight": 0.075,
      "justification": "..."
    }},
    "Delivery & Communication": {{
      "score": "1-5",
      "weight": 0.075,
      "justification": "..."
    }}
  }},

  "strengths": ["...", "..."],
  "weaknesses": ["...", "..."],
  "recommendations": ["...", "..."]
}}
________________________________________
🧠 INTERNAL LOGIC (BOTH MODES)
•	Track scores across turns 
•	Maintain dimension coverage map 
•	Detect repetition / plateau 
•	Prioritize weakest dimensions 
•	Follow progression: 
o	What → How → Why → What-if 
________________________________________
🎯 DIFFICULTY CONTROL
•	Low → basic understanding 
•	Medium → applied reasoning 
•	High → critical + edge cases 
________________________________________
⚠️ HARD RULES
•	Ask ONLY ONE question in LIVE mode 
•	NEVER show scores during viva 
•	STOP when rules are met 
•	NO repetition 
•	Maintain academic tone 
________________________________________
🎯 TASK
Using {message} and {best_practice}:
•	Continue viva OR 
•	Stop and generate final evaluation 
based on stopping rules.


"""

prompt = PromptTemplate(
    input_variables=["message", "best_practice"],
    template=PROMPT_TEMPLATE
)

chain = LLMChain(llm=llm, prompt=prompt)

# --------------------------------------------------
# Response
# --------------------------------------------------
def generate_response(message):
    try:
        best_practice = retrieve_info(message)
        response = chain.run(
            message=message,
            best_practice=best_practice
        )
        return response
    except Exception as e:
        return f"⚠️ Error:\n{str(e)}"

# --------------------------------------------------
# PDF Generator
# --------------------------------------------------
def generate_viva_pdf(chat_history, filename="AIViva_Transcript.pdf"):
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("<b>AIVivaXaminer – Viva Transcript</b>", styles["Title"]))
    story.append(Spacer(1, 12))

    date_str = datetime.now().strftime("%d %B %Y, %H:%M")
    story.append(Paragraph(f"<b>Date:</b> {date_str}", styles["Normal"]))
    story.append(Spacer(1, 20))

    for msg in chat_history:
        role = "Student" if msg["role"] == "user" else "Examiner"
        content = msg["content"].replace("\n", "<br/>")

        story.append(Paragraph(f"<b>{role}:</b><br/>{content}", styles["Normal"]))
        story.append(Spacer(1, 12))

    pdf = SimpleDocTemplate(
        filename,
        pagesize=A4,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )

    pdf.build(story)
    return filename

# --------------------------------------------------
# APP
# --------------------------------------------------
def main():
    st.set_page_config(
        page_title="AIVivaXaminer",
        page_icon="🎓",
        layout="centered"
    )

    st.title("🎓 AIVivaXaminer")

    # ---------------- SESSION STATE ----------------
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
                st.session_state.messages = []
                st.session_state.viva_active = True
                st.success("Session reset")
                st.rerun()

            if st.button("🚪 Logout"):
                st.session_state.examiner_logged_in = False
                st.rerun()

    # ---------------- CHAT (ALWAYS ACTIVE) ----------------
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # ---------------- INPUT (NO LOGIN REQUIRED) ----------------
    user_input = st.chat_input(
        "Enter your research title ..."
    ) if st.session_state.viva_active else None

    if user_input:
        st.chat_message("user").markdown(user_input)

        st.session_state.messages.append(
            {"role": "user", "content": user_input}
        )

        with st.chat_message("assistant"):
            placeholder = st.empty()
            full_response = ""

            assistant_response = generate_response(user_input)

            for word in assistant_response.split():
                full_response += word + " "
                time.sleep(0.02)
                placeholder.markdown(full_response + "▌")

            placeholder.markdown(full_response)

        st.session_state.messages.append(
            {"role": "assistant", "content": full_response}
        )


# --------------------------------------------------
if __name__ == "__main__":
    main()
