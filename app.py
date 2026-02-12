import streamlit as st
import time
from io import BytesIO
from dotenv import load_dotenv

from langchain.document_loaders.csv_loader import CSVLoader
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain

# ✅ PDF generation
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

load_dotenv()

# -------------------------------
# 1. Vectorise the student-teacher response csv data
# -------------------------------
loader = CSVLoader(file_path="ts_response.csv")
documents = loader.load()
embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(documents, embeddings)

# -------------------------------
# 2. Function for similarity search
# -------------------------------
def retrieve_info(query):
    similar_response = db.similarity_search(query, k=3)
    return [doc.page_content for doc in similar_response]

# -------------------------------
# 3. Setup LLMChain & prompts
# -------------------------------
llm = ChatOpenAI(temperature=0.7, model="gpt-4-turbo")

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
    return chain.run(message=message, best_practice=best_practice)

# -------------------------------
# ✅ PDF generation helper
# -------------------------------
def generate_pdf(messages):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("<b>AIVivaXaminer – Viva Transcript</b>", styles["Title"]))
    story.append(Spacer(1, 12))

    for msg in messages:
        role = "Student" if msg["role"] == "user" else "Examiner"
        content = msg["content"].replace("\n", "<br/>")
        story.append(
            Paragraph(f"<b>{role}:</b> {content}", styles["Normal"])
        )
        story.append(Spacer(1, 8))

    doc.build(story)
    buffer.seek(0)
    return buffer

# -------------------------------
# 4. Streamlit app
# -------------------------------
EXAMINER_PASSWORD = "exam123"

def main():
    st.set_page_config(page_title="AIVivaXaminer", page_icon=":computer:")
    st.title(":computer: AIVivaXaminer")

    # Initialize session state
    defaults = {
        "examiner_logged_in": False,
        "messages": [],
        "question_count": 0,
        "viva_active": True,
        "max_questions": 10
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    # -------------------------------
    # Examiner Authentication
    # -------------------------------
    if st.session_state.examiner_logged_in:
        st.sidebar.success("Examiner logged in")
        if st.sidebar.button("Log out"):
            st.session_state.examiner_logged_in = False
    else:
        password = st.sidebar.text_input("Examiner Password", type="password")
        if password and password == EXAMINER_PASSWORD:
            st.session_state.examiner_logged_in = True
            st.sidebar.success("Authenticated")
        elif password:
            st.sidebar.error("Incorrect password")

    # -------------------------------
    # Examiner Control Panel
    # -------------------------------
    if st.session_state.examiner_logged_in:
        st.sidebar.header("Examiner Control Panel")
        st.session_state.max_questions = st.sidebar.number_input(
            "Max questions", min_value=1, value=st.session_state.max_questions
        )
        if st.sidebar.button("Force Stop Viva"):
            st.session_state.viva_active = False
            st.warning("Viva forcibly stopped by examiner.")

    # -------------------------------
    # Display chat messages
    # -------------------------------
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # -------------------------------
    # ✅ If viva ended, show PDF export
    # -------------------------------
    if not st.session_state.viva_active:
        st.info("Viva session has ended. You can download the transcript below.")

        pdf_file = generate_pdf(st.session_state.messages)

        st.download_button(
            label="📄 Download Viva Transcript (PDF)",
            data=pdf_file,
            file_name="AIVivaXaminer_Transcript.pdf",
            mime="application/pdf"
        )

        # Stop the rest of the app (no chat input)
        return

    # -------------------------------
    # Chat input
    # -------------------------------
    if user_input := st.chat_input(
        "Enter your research title to start (or type 'end viva' to finish):"
    ):
        if user_input.strip().lower() == "end viva":
            st.session_state.viva_active = False
            st.success("Viva session ended by the student.")
            st.experimental_rerun()  # force re-render to show PDF
            return

        with st.chat_message("user"):
            st.markdown(user_input)
        st.session_state.messages.append(
            {"role": "user", "content": user_input}
        )

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            assistant_response = generate_response(user_input)

            for chunk in assistant_response.split():
                full_response += chunk + " "
                time.sleep(0.05)
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)

        st.session_state.messages.append(
            {"role": "assistant", "content": full_response}
        )

        st.session_state.question_count += 1
        if st.session_state.question_count >= st.session_state.max_questions:
            st.session_state.viva_active = False
            st.warning("Maximum number of questions reached. Viva session ended.")
            st.experimental_rerun()  # force re-render to show PDF


if __name__ == "__main__":
    main()

