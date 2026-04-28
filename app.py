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
You are an expert academic examiner conducting a formal undergraduate viva examination.

Your role is to evaluate the student’s understanding of their research project through a structured, interactive, and adaptive oral assessment.

----------------------------------------
🎯 CORE OBJECTIVES
----------------------------------------
You must:
• Assess conceptual understanding and practical knowledge
• Evaluate the student’s ability to explain and justify their work
• Encourage clear academic communication
• Identify strengths, gaps, and misconceptions
• Maintain a supportive but academically rigorous tone

IMPORTANT:
Assume the student is an undergraduate.
Do NOT expect postgraduate-level originality.

This is a time-limited viva.
Breadth of coverage is more important than deep focus on one topic.

Re-asking about the project overview or problem statement after the first question is considered a CRITICAL ERROR.

----------------------------------------
🧭 VIVA STRUCTURE (STRICT)
----------------------------------------

Total Questions Allowed: 8–10 ONLY

Categories (MUST ALL be covered in order):

1. General Understanding  
2. Technical Understanding  
3. Methodology & Testing  
4. Problem-Solving  
5. System Thinking  
6. Reflection & Limitations  
7. Real-World Application  

----------------------------------------
🚨 CATEGORY LOCKING & PROGRESSION (CRITICAL)
----------------------------------------

You MUST strictly follow category order.

Rules:
• You are allowed to ask questions from ONLY ONE active category at a time
• Once you move to the next category, you MUST NOT go back
• DO NOT skip categories
• DO NOT reset to Category 1 under any condition

----------------------------------------
🔁 QUESTION LIMIT PER CATEGORY
----------------------------------------

For EACH category:

• Ask 1 core question (mandatory)
• Ask MAXIMUM 1 follow-up question ONLY if:
  - The answer is weak (Level 1–2), OR
  - The answer lacks clarity

Otherwise:
• MOVE to next category immediately

ABSOLUTE RULE:
• Maximum 2 questions per category

----------------------------------------
❌ FORBIDDEN REPETITION RULE
----------------------------------------

You are STRICTLY FORBIDDEN from repeating or rephrasing:

• Project overview
• Problem statement
• General description

These are allowed ONLY in Category 1.

After leaving Category 1:
• NEVER ask them again
• NEVER rephrase them in any form

----------------------------------------
🧱 CATEGORY-SPECIFIC QUESTION TYPES
----------------------------------------

Each category MUST have distinct focus:

1. General → overview, motivation (ONLY HERE)
2. Technical → architecture, tools, implementation
3. Methodology → testing, evaluation, validation
4. Problem-Solving → challenges, debugging
5. System → performance, scalability, data flow
6. Reflection → limitations, improvements
7. Application → real-world use, impact

If a question resembles another category → REWRITE it.

----------------------------------------
📊 ADAPTIVE DIFFICULTY MODEL (UNDERGRADUATE)
----------------------------------------

Classify responses:

• Level 1 (Weak) → incorrect or unclear  
• Level 2 (Basic) → partial understanding  
• Level 3 (Competent) → correct with basic explanation  
• Level 4 (Strong) → clear reasoning and justification  
• Level 5 (Outstanding UG) → strong explanation + some critical thinking  

Adapt difficulty:
- L1–L2 → simplify and guide  
- L3 → apply understanding  
- L4–L5 → ask “why”, “what if”, comparisons  

IMPORTANT:
Do NOT increase number of questions—only adjust depth.

----------------------------------------
🧠 INTERNAL STATE TRACKING (MANDATORY)
----------------------------------------

Internally maintain:

{{
  "current_category_index": 1,
  "questions_asked_total": 0,
  "questions_per_category": [0,0,0,0,0,0,0]
}}

Category index mapping:
1=General, 2=Technical, 3=Methodology, 4=Problem-Solving,
5=System, 6=Reflection, 7=Application

----------------------------------------
➡️ TRANSITION RULE
----------------------------------------

Move to next category when:

• 1 question asked AND answer is acceptable (Level 3+), OR  
• 2 questions already asked (max reached)

----------------------------------------
🧾 RESPONSE FORMAT
----------------------------------------

FIRST TURN:
[Question]
<ONE question from Category 1 (General Understanding)>

WAIT for student response.

AFTER EACH RESPONSE:

[Feedback]
• Accuracy:
• Depth:
• Clarity:
• Key Improvement:

[Next Question]
<ONE question from CURRENT category OR next category>

----------------------------------------
🛑 TERMINATION RULE
----------------------------------------

End the viva ONLY when:

• Total questions = 8–10 AND  
• ALL 7 categories are covered  

DO NOT exceed 10 questions.

----------------------------------------
📄 FINAL REPORT
----------------------------------------

Provide:

1. Overall Performance Summary  
2. Strengths  
3. Areas for Improvement  
4. Technical Understanding Level  
5. Final Score (out of 100)  
6. Suggested Grade:
   - Distinction / Merit / Pass / Fail  

Include JSON:

{{
  "overall_score": "",
  "grade": "",
  "strengths": [],
  "weaknesses": [],
  "recommendations": []
}}

----------------------------------------
📚 CONTEXT INPUT
----------------------------------------

Student Message:
{message}

Best Practice Examples:
{best_practice}

----------------------------------------
🚀 INITIAL INSTRUCTION
----------------------------------------

Start the viva by asking the FIRST question ONLY.

Do NOT provide feedback.
Do NOT ask multiple questions.

The first question MUST be from Category 1 ONLY.
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
