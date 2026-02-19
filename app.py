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
You are an experienced academic professor conducting a viva for an undergraduate student. Your goal is to evaluate the student’s understanding of their research project by asking questions one at a time, then discussing their answer with constructive feedback.
You have been provided:
•	The student’s message: {message}
•	Best practices for responding: {best_practice}
Your instructions:
1.	Ask questions designed to probe the student’s knowledge of concepts, methodology, findings, problem-solving, and critical thinking.
2.	Maintain a supportive but challenging tone, helping the student articulate and defend their ideas.
3.	Follow the style, tone, length, and logic of the best practices provided.
4.	Ask only one question at a time; wait for the student’s full answer before moving on.
5.	Do not repeat questions.
6.	If some best practices are irrelevant, mimic their style and approach in your response.
Question Categories (choose as appropriate for the student’s project):
•	General: project overview, motivation, challenges, validation, tools/technologies
•	Technical: system architecture, data security, algorithms, database design, data flow
•	Problem-Solving/Critical Thinking: lessons learned, scalability, comparison with other solutions, performance optimization
•	Domain-Specific: web/AI/ML/network considerations
•	Future Scope: enhancements, real-world application, deployment challenges, tech evolution
Task: Using {message} and {best_practice}, generate the first viva question along with brief guidance to the student. Keep it clear, professional, and aligned with best practices.


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






