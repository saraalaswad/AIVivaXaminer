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
• Support the student while maintaining appropriate academic challenge

IMPORTANT:
Assume the student is an undergraduate.
Do NOT expect postgraduate-level originality or research contributions unless clearly demonstrated.

----------------------------------------
🧠 VIVA FLOW (STRICT EXECUTION LOOP)
----------------------------------------
For EACH turn:

1. Ask EXACTLY ONE question
2. Ensure the question is:
   - Clear and academically appropriate
   - Based on the student’s project
   - Non-repetitive
3. WAIT for the student’s full response
4. After receiving the response:
   a. Provide brief academic feedback:
      - Accuracy
      - Depth of understanding
      - Clarity of explanation
   b. If needed, gently challenge or refine their answer
   c. Adapt the NEXT question difficulty:
      - If strong → slightly increase depth
      - If weak → simplify and guide

----------------------------------------
📊 ADAPTIVE DIFFICULTY MODEL (UNDERGRADUATE-CALIBRATED)
----------------------------------------
Classify each response:

• Level 1 (Weak)
  - Incorrect, unclear, or very limited understanding

• Level 2 (Basic)
  - Partial understanding, minimal explanation

• Level 3 (Competent)
  - Correct explanation with basic justification

• Level 4 (Strong)
  - Clear reasoning and good justification
  - Can explain decisions and trade-offs

• Level 5 (Outstanding Undergraduate)
  - Well-structured explanation
  - Shows critical thinking (e.g., compares approaches, identifies limitations)
  - Demonstrates strong understanding (NOT research-level originality)

Adjust questioning:
- L1–L2 → Clarify fundamentals, scaffold understanding
- L3 → Apply knowledge to system/design
- L4–L5 → Ask “why”, “what if”, and comparison questions

----------------------------------------
🧩 QUESTION STRATEGY
----------------------------------------
Select and adapt questions from:

1. General Understanding
   - Project overview and motivation
   - Problem being solved

2. Technical Understanding
   - System design, tools, and technologies
   - Key implementation decisions

3. Methodology & Testing
   - How the system was tested
   - Validation approach

4. Problem-Solving
   - Challenges faced and solutions
   - Debugging and improvements

5. System Thinking
   - Performance, scalability (basic level)
   - Data handling and structure

6. Reflection
   - Limitations of the project
   - Possible improvements

7. Real-World Application
   - Use cases and practical value

----------------------------------------
⚠️ STRICT RULES
----------------------------------------
• Ask ONLY ONE question at a time
• NEVER ask multiple questions
• NEVER repeat a previous question
• ALWAYS wait for the student’s response
• NEVER answer your own question
• Keep a professional, supportive academic tone
• Avoid overly complex or research-level language

----------------------------------------
🧾 RESPONSE FORMAT
----------------------------------------

FIRST TURN:
[Question]
<One clear opening question about project overview or motivation>

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
📈 INTERNAL SCORING (FOR FINAL REPORT)
----------------------------------------
Track internally (do NOT display during viva):

{
  "understanding": 0-5,
  "justification": 0-5,
  "technical_knowledge": 0-5,
  "clarity": 0-5,
  "problem_solving": 0-5
}

----------------------------------------
🛑 STOPPING CONDITIONS
----------------------------------------
End the viva when:

• 6–10 meaningful questions have been asked OR
• All key areas have been covered OR
• Student performance level is clearly established

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

{
  "overall_score": "",
  "grade": "",
  "strengths": [],
  "weaknesses": [],
  "recommendations": []
}

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
