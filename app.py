import streamlit as st
import time
from datetime import datetime
from dotenv import load_dotenv

from langchain.document_loaders.csv_loader import CSVLoader
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain

# ReportLab
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.units import inch

load_dotenv()

# --------------------------------------------------
# 1. Vectorise CSV best-practice data
# --------------------------------------------------
loader = CSVLoader(file_path="ts_response.csv")
documents = loader.load()

embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(documents, embeddings)

# --------------------------------------------------
# 2. Similarity search
# --------------------------------------------------
def retrieve_info(query):
    docs = db.similarity_search(query, k=3)
    return [doc.page_content for doc in docs]

# --------------------------------------------------
# 3. LLM & Prompt
# --------------------------------------------------
llm = ChatOpenAI(
    temperature=0.7,
    model="gpt-4-turbo"
)

template = """
You are AIVivaXaminer, an AI-based academic examiner conducting a structured undergraduate viva assessment.

────────────────────────────────────
RULES
────────────────────────────────────
1. Ask ONE question at a time.
2. Wait for the student’s full response before proceeding.
3. NEVER repeat a question. Before generating a question, check the list below.
4. Provide examiner-style qualitative feedback internally only.
5. Scores are INTERNAL only; NEVER reveal them.
6. Maintain a professional, academic examiner tone.

────────────────────────────────────
QUESTION HISTORY
────────────────────────────────────
Previously asked questions (update this list after each turn):
{question_history}

────────────────────────────────────
QUESTION STRATEGY
────────────────────────────────────
- Categories: General Understanding, Technical Depth (if applicable), Critical Thinking, Domain-Specific Inquiry, Future Scope & Application
- Start from foundational knowledge and progress to higher-order reasoning.
- Select questions NOT in {question_history}.
- Avoid asking questions that the student has already answered adequately in prior responses.

────────────────────────────────────
ASSESSMENT FRAMEWORK (INTERNAL ONLY)
────────────────────────────────────
Evaluate each response using these dimensions:
1. Conceptual Understanding
2. Methodological Rigor
3. Technical Depth (if applicable)
4. Critical Thinking
5. Communication & Academic Articulation

Scoring scale (internal only):
0 = Not demonstrated
1 = Weak
2 = Adequate
3 = Good
4 = Excellent

Use these scores to guide your questioning, but do not reveal them.

────────────────────────────────────
STOPPING RULES
────────────────────────────────────
Terminate the viva when ANY of the following occurs:

1. Minimum category coverage:
   - ≥2 general understanding questions
   - ≥2 technical questions (if applicable)
   - ≥1 critical thinking question
   - ≥1 future-oriented question

2. Performance stabilization:
   - Average scores change ≤ ±0.5 across three consecutive questions

3. Knowledge exhaustion:
   - Two consecutive weak responses (score ≤ 1) in the same dimension

4. Sustained excellence:
   - Average score ≥ 3.5 for three consecutive questions

5. Maximum question limit:
   - Undergraduate viva: 8–12 questions
   - Capstone project: up to 15 questions

6. Two consecutive non-substantive or irrelevant responses

────────────────────────────────────
OUTPUT RULES
────────────────────────────────────
- During viva: output ONLY the next question.
- After stopping: output ONLY the final evaluation report containing:
  • Overall performance level
  • Key strengths
  • Areas for improvement
  • Final recommendation (Pass / Pass with Minor Revisions / Borderline / Fail)

"""

prompt = PromptTemplate(
    input_variables=["message", "best_practice"],
    template=template
)

chain = LLMChain(llm=llm, prompt=prompt)

def generate_response(message):
    best_practice = retrieve_info(message)
    return chain.run(
        message=message,
        best_practice=best_practice
    )

# --------------------------------------------------
# 4. ReportLab PDF Generator
# --------------------------------------------------
def generate_viva_pdf(messages):
    file_path = "Final_Viva_Report.pdf"

    doc = SimpleDocTemplate(
        file_path,
        pagesize=A4,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )

    styles = getSampleStyleSheet()

    styles.add(
        ParagraphStyle(
            name="RoleStyle",
            fontName="Helvetica-Bold",
            fontSize=11,
            leading=14,
            spaceBefore=12,
            spaceAfter=4,
            alignment=TA_LEFT
        )
    )

    styles.add(
        ParagraphStyle(
            name="ContentStyle",
            fontSize=11,
            leading=14,
            spaceAfter=8,
            alignment=TA_LEFT
        )
    )

    story = []

    story.append(Paragraph("<b>Final Viva Report</b>", styles["Title"]))
    story.append(Spacer(1, 0.3 * inch))

    story.append(
        Paragraph(
            f"<b>Date:</b> {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            styles["Normal"]
        )
    )
    story.append(Spacer(1, 0.3 * inch))

    for msg in messages:
        role = "Student" if msg["role"] == "user" else "Examiner"
        story.append(Paragraph(role, styles["RoleStyle"]))
        story.append(Paragraph(msg["content"], styles["ContentStyle"]))

    doc.build(story)
    return file_path

# --------------------------------------------------
# 5. Streamlit App
# --------------------------------------------------
EXAMINER_PASSWORD = "exam123"

def main():
    st.set_page_config(
        page_title="AIVivaXaminer",
        page_icon="🎓"
    )

    st.title("🎓 AIVivaXaminer")
    st.sidebar.title("Examiner Panel")

    # --------------------------------------------------
    # Session state defaults
    # --------------------------------------------------
    defaults = {
        "examiner_logged_in": False,
        "messages": [],
        "question_count": 0,
        "answer_count": -1,
        "viva_active": True,
        "viva_completed": False,
        "max_questions": 10
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    # --------------------------------------------------
    # Sidebar: Examiner authentication
    # --------------------------------------------------
    if st.session_state.examiner_logged_in:
        st.sidebar.success("Examiner logged in")
        if st.sidebar.button("Log out"):
            st.session_state.examiner_logged_in = False
    else:
        password = st.sidebar.text_input(
            "Examiner Password (optional)",
            type="password"
        )
        if password and password == EXAMINER_PASSWORD:
            st.session_state.examiner_logged_in = True
            st.sidebar.success("Authentication successful")
        elif password:
            st.sidebar.error("Incorrect password")

    # --------------------------------------------------
    # Sidebar: Examiner control panel
    # --------------------------------------------------
    if st.session_state.examiner_logged_in:
        st.sidebar.header("Viva Controls")

        st.session_state.max_questions = st.sidebar.number_input(
            "Max questions",
            min_value=1,
            value=st.session_state.max_questions
        )

        st.sidebar.info(
            f"""
            **Questions asked:** {st.session_state.question_count}  
            **Answers given:** {st.session_state.answer_count}
            """
        )

        if st.sidebar.button("Force Stop Viva"):
            st.session_state.viva_active = False
            st.session_state.viva_completed = True
            st.warning("Viva forcibly stopped by examiner.")

    # --------------------------------------------------
    # Display chat history
    # --------------------------------------------------
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # --------------------------------------------------
    # Chat input
    # --------------------------------------------------
    if st.session_state.viva_active:
        user_input = st.chat_input(
            "Enter your research title / response (or type 'end viva')"
        )

        if user_input:
            if user_input.strip().lower() == "end viva":
                st.session_state.viva_active = False
                st.session_state.viva_completed = True
                st.success("Viva session ended by the student.")
            else:
                # Student answer
                st.session_state.messages.append(
                    {"role": "user", "content": user_input}
                )
                st.session_state.answer_count += 1

                with st.chat_message("user"):
                    st.markdown(user_input)

                # Examiner question
                response = generate_response(user_input)
                animated = ""

                st.session_state.messages.append(
                    {"role": "assistant", "content": ""}
                )
                placeholder_index = len(st.session_state.messages) - 1

                with st.chat_message("assistant"):
                    placeholder = st.empty()
                    for word in response.split():
                        animated += word + " "
                        time.sleep(0.04)
                        placeholder.markdown(animated + "▌")
                    placeholder.markdown(animated)

                st.session_state.messages[placeholder_index]["content"] = animated
                st.session_state.question_count += 1

                # End check AFTER last answer
                if (
                    st.session_state.question_count >= st.session_state.max_questions
                    and st.session_state.answer_count >= st.session_state.max_questions
                ):
                    st.session_state.viva_active = False
                    st.session_state.viva_completed = True
                    st.warning("Viva completed successfully.")

    # --------------------------------------------------
    # FINAL VIVA REPORT
    # --------------------------------------------------
    if st.session_state.viva_completed:
        st.markdown("---")
        st.subheader("📄 Final Viva Report")

        if st.button("Generate Final Viva Report (PDF)"):
            pdf_path = generate_viva_pdf(st.session_state.messages)

            with open(pdf_path, "rb") as f:
                st.download_button(
                    label="⬇️ Download Final Viva Report",
                    data=f,
                    file_name="Final_Viva_Report.pdf",
                    mime="application/pdf"
                )

if __name__ == "__main__":
    main()
