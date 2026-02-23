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
    docs = db.similarity_search(query, k=8)

    target_category = st.session_state.category_order[
        st.session_state.current_category_index
    ]

    filtered = []
    for doc in docs:
        question = doc.metadata.get("examiner_question", "")
        category = doc.metadata.get("category", "")

        if (
            question not in st.session_state.asked_questions
            and category == target_category
        ):
            filtered.append(doc.page_content)

    # Fallback: if no question found in target category
    if not filtered:
        for doc in docs:
            question = doc.metadata.get("examiner_question", "")
            if question not in st.session_state.asked_questions:
                filtered.append(doc.page_content)

    return filtered[:3]



# --------------------------------------------------
# 3. LLM & Prompt
# --------------------------------------------------
llm = ChatOpenAI(
    temperature=0.7,
    model="gpt-4-turbo"
)

template = """
You are an experienced academic professor conducting a viva for an undergraduate student. 
Your goal is to evaluate the student’s understanding of their research project by asking questions one at a time, then discussing their answer with constructive feedback.
Ask questions designed to probe the student’s knowledge of concepts, methodology, findings, problem-solving, and critical thinking.

You have been provided:
- Student input: {message}
- Retrieved Q&A examples (none of which have been previously asked): {retrieved_qa}
- Categories already covered: {asked_categories}

Question Categories (choose as appropriate for the student’s project):
•	General: project overview, motivation, challenges, validation, tools/technologies
•	Technical: system architecture, data security, algorithms, database design, data flow
•	Problem-Solving/Critical Thinking: lessons learned, scalability, comparison with other solutions, performance optimization
•	Domain-Specific: web/AI/ML/network considerations
•	Future Scope: enhancements, real-world application, deployment challenges, tech evolution

Instructions:
1. Select ONE question from the retrieved Q&A that has not been asked before.
2. Prefer a category that has not yet been covered in this viva.
3. Ask exactly ONE question.
4. Do NOT repeat or paraphrase any previously asked question.
5. Maintain a supportive but academically rigorous tone.
6. Occasionally acknowledge the student’s answer briefly before asking the next question.
7. If this is the final question, frame it as a reflective or future-oriented question.
8. Ensure questions follow a logical viva examination order.

Question strategy based on mode:
- CLARIFY: ask a simpler or guiding question
- PROBE: ask a standard category question
- DEEPEN: ask a deeper "why / how / justify" question

Output format:
**Category:**
Question:
"""

prompt = PromptTemplate(
    input_variables=["message", "retrieved_qa", "question_mode"],
    template=template
)

chain = LLMChain(llm=llm, prompt=prompt)

# --------------------------------------------------
# Viva flow helper (question depth control)
# --------------------------------------------------
def decide_question_mode(student_answer: str) -> str:
    if not student_answer:
        return "PROBE"

    word_count = len(student_answer.split())

    if word_count < 20:
        return "CLARIFY"      # weak / short answer
    elif word_count < 60:
        return "PROBE"        # normal answer
    else:
        return "DEEPEN"       # strong answer


def advance_category():
    st.session_state.current_category_index += 1
    if st.session_state.current_category_index >= len(st.session_state.category_order):
        st.session_state.current_category_index = 0


def generate_response(message):
    retrieved_qa = retrieve_info(message)

    response = chain.run(
        message=message,
        retrieved_qa=retrieved_qa,
        asked_categories=list(st.session_state.asked_categories)
    )

    # ----------------------------------------------
    # HARD MEMORY UPDATE (prevents repetition)
    # ----------------------------------------------
    st.session_state.asked_questions.add(response)

    return response


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
        "max_questions": 10,
        "asked_questions": set(),
        "asked_categories": set(),
        "viva_phase": "OPENING",
        "last_question_type": None,
        "followup_depth": 0

    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    # --------------------------------------------------
    # Viva memory (PREVENT REPEATED QUESTIONS)
    # --------------------------------------------------
    if "asked_questions" not in st.session_state:
        st.session_state.asked_questions = set()

    if "asked_categories" not in st.session_state:
        st.session_state.asked_categories = set()

    # --------------------------------------------------
    # Category rotation state
    # --------------------------------------------------
    if "category_order" not in st.session_state:
        st.session_state.category_order = [
            "General",
            "Technical",
            "Problem-Solving / Critical Thinking",
            "Domain-Specific",
            "Ethics / Professionalism",
            "Future Scope"
        ]

    if "current_category_index" not in st.session_state:
        st.session_state.current_category_index = 0




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

                question_mode = decide_question_mode(user_input)
                # Examiner question
                if (st.session_state.question_count < st.session_state.max_questions):
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
                    # Advance category rotation
                    st.session_state.current_category_index += 1
                    if st.session_state.current_category_index >= len(st.session_state.category_order):
                        st.session_state.current_category_index = 0

                    # Track category coverage (lightweight)
                    st.session_state.asked_categories.add("AUTO")


                    if question_mode == "DEEPEN":
                        st.session_state.followup_depth += 1
                    else:
                        st.session_state.followup_depth = 0
                
                    if st.session_state.followup_depth >= 2:
                        st.session_state.viva_phase = "REDIRECTING"
                
                    if st.session_state.viva_phase in ["REDIRECTING", "CLOSING"]:
                        advance_category()

                # End check AFTER last answer
                if (
                    st.session_state.question_count >= st.session_state.max_questions
                    and st.session_state.answer_count >= st.session_state.max_questions
                    or st.session_state.viva_phase == "CLOSING"
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
























