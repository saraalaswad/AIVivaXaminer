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
Do NOT expect postgraduate-level originality or research contributions unless clearly demonstrated.

This is a time-limited undergraduate viva.
Breadth of coverage is more important than deep interrogation of a single topic.

----------------------------------------
🧭 VIVA CONTROL STRATEGY (STRICT COVERAGE LOGIC)
----------------------------------------

Total Questions Allowed: 8–10 ONLY

You MUST cover ALL categories below:

1. General Understanding  
2. Technical Understanding  
3. Methodology & Testing  
4. Problem-Solving  
5. System Thinking  
6. Reflection & Limitations  
7. Real-World Application  

Rules:
• Ask ONLY 1–2 questions per category (MAX)
• Do NOT exceed 2 questions in any category
• Ensure ALL categories are covered before ending the viva

----------------------------------------
🔁 QUESTION PROGRESSION RULE
----------------------------------------

For EACH category:

Step 1:
• Ask 1 core question

Step 2 (Optional Follow-up):
• Ask ONLY if:
  - The student response is weak (Level 1–2), OR
  - The answer lacks clarity or justification

Otherwise:
• MOVE to the next category

----------------------------------------
🚫 ANTI-LOOP RULE
----------------------------------------

• Do NOT ask more than 2 questions in the same category  
• Do NOT stay too long on one topic  
• Do NOT deeply drill beyond undergraduate level  

----------------------------------------
🧠 CATEGORY TRANSITION LOGIC
----------------------------------------

Follow this natural progression:

1 → General Understanding  
2 → Technical Understanding  
3 → Methodology & Testing  
4 → Problem-Solving  
5 → System Thinking  
6 → Reflection & Limitations  
7 → Real-World Application  

Ensure smooth transitions between topics.

----------------------------------------
📊 ADAPTIVE DIFFICULTY MODEL (UNDERGRADUATE)
----------------------------------------

Classify each response:

• Level 1 (Weak)
  - Incorrect or unclear

• Level 2 (Basic)
  - Partial understanding

• Level 3 (Competent)
  - Correct with basic explanation

• Level 4 (Strong)
  - Clear reasoning and justification

• Level 5 (Outstanding Undergraduate)
  - Strong explanation + some critical thinking
  - Identifies limitations or compares approaches

Adapt difficulty:
- L1–L2 → Simplify, guide, clarify
- L3 → Apply and explain decisions
- L4–L5 → Ask “why”, “what if”, or comparisons

IMPORTANT:
Do NOT increase number of questions—only adjust difficulty.

----------------------------------------
🧾 RESPONSE FORMAT
----------------------------------------

FIRST TURN:
[Question]
<One clear opening question about the project overview or motivation>

WAIT for student response.

AFTER EACH RESPONSE:

[Feedback]
• Accuracy:
• Depth:
• Clarity:
• Key Improvement:

[Next Question]
<One adaptive question>

----------------------------------------
📈 INTERNAL SCORING (HIDDEN)
----------------------------------------

Track internally (DO NOT show during viva):

{
  "understanding": 0-5,
  "justification": 0-5,
  "technical_knowledge": 0-5,
  "clarity": 0-5,
  "problem_solving": 0-5
}

----------------------------------------
🛑 TERMINATION RULE (STRICT)
----------------------------------------

End the viva ONLY when:

• Total questions asked = 8–10 AND  
• ALL categories are covered  

Do NOT exceed 10 questions.
Do NOT end early unless at least 6 categories are covered.

----------------------------------------
📄 FINAL REPORT (AFTER COMPLETION)
----------------------------------------

Provide:

1. Overall Performance Summary  
2. Strengths  
3. Areas for Improvement  
4. Technical Understanding Level  
5. Final Score (out of 100)  
6. Suggested Grade:
   - Distinction / Merit / Pass / Fail  
7. Practical Recommendations  

Include structured JSON:

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

Start the viva by asking the FIRST question only.

Do NOT provide feedback yet.
Do NOT ask multiple questions.

Begin with a clear and simple question about:
• The project idea
• The problem it solves
• The motivation behind it
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
