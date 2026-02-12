import streamlit as st
import time
from datetime import datetime
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

from langchain.document_loaders.csv_loader import CSVLoader
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from dotenv import load_dotenv

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
# Viva Report Generators
# -------------------------------
def generate_viva_report():
    lines = []
    lines.append("AIVivaXaminer – Viva Examination Report")
    lines.append("=" * 45)
    lines.append(f"Start Time: {st.session_state.viva_start_time}")
    lines.append(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Total Questions Asked: {st.session_state.question_count}")
    lines.append(f"Viva End Reason: {st.session_state.viva_end_reason}")
    lines.append("\n--- Viva Transcript ---\n")

    for msg in st.session_state.messages:
        role = "Student" if msg["role"] == "user" else "Examiner"
        lines.append(f"{role}: {msg['content']}\n")

    return "\n".join(lines)

def generate_viva_pdf():
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    x_margin = 40
    y = height - 40

    def draw(text):
        nonlocal y
        if y < 50:
            c.showPage()
            y = height - 40
        c.drawString(x_margin, y, text)
        y -= 14

    c.setFont("Helvetica-Bold", 14)
    draw("AIVivaXaminer – Viva Examination Report")
    y -= 10

    c.setFont("Helvetica", 10)
    draw(f"Start Time: {st.session_state.viva_start_time}")
    draw(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    draw(f"Total Questions Asked: {st.session_state.question_count}")
    draw(f"Viva End Reason: {st.session_state.viva_end_reason}")
    y -= 20

    c.setFont("Helvetica-Bold", 11)
    draw("Viva Transcript")
    y -= 10
    c.setFont("Helvetica", 10)

    for msg in st.session_state.messages:
        role = "Student" if msg["role"] == "user" else "Examiner"
        for line in f"{role}: {msg['content']}".split("\n"):
            draw(line)

    c.save()
    buffer.seek(0)
    return buffer

# -------------------------------
# 4. Streamlit App
# -------------------------------
EXAMINER_PASSWORD = "exam123"

def main():
    st.set_page_config(page_title="AIVivaXaminer", page_icon=":computer:")
    st.title(":computer: AIVivaXaminer")

    defaults = {
        "examiner_logged_in": False,
        "messages": [],
        "question_count": 0,
        "viva_active": True,
        "max_questions": 10,
        "viva_start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "viva_end_reason": None,
        "generated_report": None,
        "generated_pdf": None
    }

    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    # -------------------------------
    # Examiner Login
    # -------------------------------
    if st.session_state.examiner_logged_in:
        st.sidebar.success("Examiner logged in")
        if st.sidebar.button("Log out"):
            st.session_state.examiner_logged_in = False
    else:
        pwd = st.sidebar.text_input("Examiner Password", type="password")
        if pwd and pwd == EXAMINER_PASSWORD:
            st.session_state.examiner_logged_in = True

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
            st.session_state.viva_end_reason = "Force stopped by examiner"

    # -------------------------------
    # Display chat history
    # -------------------------------
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if not st.session_state.viva_active:
        st.info("Viva session has ended.")

    # -------------------------------
    # Chat input
    # -------------------------------
    if st.session_state.viva_active:
        if user_input := st.chat_input("Enter your research title or response (type 'end viva' to finish):"):
            if user_input.lower() == "end viva":
                st.session_state.viva_active = False
                st.session_state.viva_end_reason = "Ended by student"
            else:
                with st.chat_message("user"):
                    st.markdown(user_input)
                st.session_state.messages.append({"role": "user", "content": user_input})

                with st.chat_message("assistant"):
                    placeholder = st.empty()
                    response = generate_response(user_input)
                    full = ""
                    for w in response.split():
                        full += w + " "
                        time.sleep(0.04)
                        placeholder.markdown(full + "▌")
                    placeholder.markdown(full)

                st.session_state.messages.append({"role": "assistant", "content": full})
                st.session_state.question_count += 1

                if st.session_state.question_count >= st.session_state.max_questions:
                    st.session_state.viva_active = False
                    st.session_state.viva_end_reason = "Maximum questions reached"

    # -------------------------------
    # MAIN PANEL – VIVA REPORT EXPORT
    # -------------------------------
    st.markdown("---")
    st.markdown("## 📄 Viva Report Export")

    col1, col2, col3 = st.columns([1, 1, 2])

    with col1:
        if st.button("Generate TXT Report"):
            st.session_state.generated_report = generate_viva_report()

    with col2:
        if st.button("Generate PDF Report"):
            st.session_state.generated_pdf = generate_viva_pdf()

    with col3:
        if st.session_state.generated_report:
            st.download_button(
                "⬇ Download TXT",
                st.session_state.generated_report,
                "viva_report.txt",
                "text/plain"
            )
        if st.session_state.generated_pdf:
            st.download_button(
                "⬇ Download PDF",
                st.session_state.generated_pdf,
                "viva_report.pdf",
                "application/pdf"
            )

if __name__ == "__main__":
    main()
