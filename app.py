import streamlit as st
import time
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from dotenv import load_dotenv
from fpdf import FPDF
from datetime import datetime

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
• The student’s message: {message}
• Best practices for responding: {best_practice}

Your instructions:
1. Ask questions designed to probe the student’s knowledge of concepts, methodology, findings, problem-solving, and critical thinking.
2. Maintain a supportive but challenging tone.
3. Follow the style and logic of the best practices.
4. Ask only one question at a time.
5. Do not repeat questions.

Task: Generate the next viva question with brief guidance.
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
# 4. Final Viva PDF Generator
# -------------------------------
def generate_viva_pdf(messages):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Final Viva Report", ln=True)

    pdf.set_font("Arial", size=11)
    pdf.cell(0, 8, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True)
    pdf.ln(5)

    for msg in messages:
        role = "Student" if msg["role"] == "user" else "Examiner"
        pdf.set_font("Arial", "B", 11)
        pdf.cell(0, 8, f"{role}:", ln=True)

        pdf.set_font("Arial", size=11)
        pdf.multi_cell(0, 8, msg["content"])
        pdf.ln(2)

    file_path = "Final_Viva_Report.pdf"
    pdf.output(file_path)
    return file_path

# -------------------------------
# 5. Streamlit App
# -------------------------------
EXAMINER_PASSWORD = "exam123"

def main():
    st.set_page_config(page_title="AIVivaXaminer", page_icon=":computer:")
    st.title(":computer: AIVivaXaminer")

    # -------------------------------
    # Session State Defaults
    # -------------------------------
    defaults = {
        "examiner_logged_in": False,
        "messages": [],
        "question_count": 0,
        "viva_active": True,
        "max_questions": 10,
        "viva_completed": False
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
            st.sidebar.success("Examiner authenticated.")
        elif password:
            st.sidebar.error("Incorrect password!")

    # -------------------------------
    # Examiner Control Panel
    # -------------------------------
    if st.session_state.examiner_logged_in:
        st.sidebar.header("Examiner Control Panel")

        st.session_state.max_questions = st.sidebar.number_input(
            "Max questions",
            min_value=1,
            value=st.session_state.max_questions
        )

        if st.sidebar.button("Force Stop Viva"):
            st.session_state.viva_active = False
            st.session_state.viva_completed = True
            st.warning("Viva forcibly stopped by examiner.")

    # -------------------------------
    # Display Chat History
    # -------------------------------
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if not st.session_state.viva_active:
        st.info("Viva session has ended.")

    # -------------------------------
    # Chat Input
    # -------------------------------
    if st.session_state.viva_active:
        if user_input := st.chat_input("Enter your research title to start (or type 'end viva'):"):
            if user_input.strip().lower() == "end viva":
                st.session_state.viva_active = False
                st.session_state.viva_completed = True
                st.success("Viva session ended by the student.")
                return

            with st.chat_message("user"):
                st.markdown(user_input)
            st.session_state.messages.append({"role": "user", "content": user_input})

            with st.chat_message("assistant"):
                placeholder = st.empty()
                response = generate_response(user_input)
                animated = ""
                for word in response.split():
                    animated += word + " "
                    time.sleep(0.04)
                    placeholder.markdown(animated + "▌")
                placeholder.markdown(animated)

            st.session_state.messages.append({"role": "assistant", "content": animated})

            st.session_state.question_count += 1
            if st.session_state.question_count >= st.session_state.max_questions:
                st.session_state.viva_active = False
                st.session_state.viva_completed = True
                st.warning("Maximum number of questions reached. Viva ended.")

    # -------------------------------
    # Final Viva PDF Export (POST-COMPLETION ONLY)
    # -------------------------------
    if (
        st.session_state.examiner_logged_in
        and st.session_state.viva_completed
    ):
        st.sidebar.markdown("### 📄 Final Viva Report")

        if st.sidebar.button("Generate Viva PDF Report"):
            pdf_path = generate_viva_pdf(st.session_state.messages)
            with open(pdf_path, "rb") as f:
                st.sidebar.download_button(
                    label="⬇️ Download Final Viva Report (PDF)",
                    data=f,
                    file_name="Final_Viva_Report.pdf",
                    mime="application/pdf"
                )

if __name__ == "__main__":
    main()
