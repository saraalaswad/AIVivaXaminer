import streamlit as st
import time
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

# ─────────────────────────────────────────────
# PDF GENERATION
# ─────────────────────────────────────────────
def generate_viva_pdf(questions, responses, averages, overall, recommendation):
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph("AIVivaXaminer – Final Viva Report", styles["Title"]))
    elements.append(Spacer(1, 12))
    elements.append(Paragraph(f"<b>Overall Score:</b> {overall}", styles["Normal"]))
    elements.append(Paragraph(f"<b>Final Recommendation:</b> {recommendation}", styles["Normal"]))
    elements.append(Spacer(1, 12))

    elements.append(Paragraph("<b>Dimension Averages</b>", styles["Heading2"]))
    for k, v in averages.items():
        elements.append(Paragraph(f"{k}: {v}", styles["Normal"]))

    elements.append(Spacer(1, 12))
    elements.append(Paragraph("<b>Viva Questions & Responses</b>", styles["Heading2"]))

    for i, (q, r) in enumerate(zip(questions, responses), 1):
        elements.append(Spacer(1, 8))
        elements.append(Paragraph(f"<b>Q{i}:</b> {q}", styles["Normal"]))
        elements.append(Paragraph(f"<b>Response:</b> {r}", styles["Normal"]))

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    doc = SimpleDocTemplate(tmp.name, pagesize=A4)
    doc.build(elements)
    return tmp.name

# -------------------------------
# 4. Streamlit app
# -------------------------------
EXAMINER_PASSWORD = "exam123"  # <-- change this to a secure password

def main():
    st.set_page_config(page_title="AIVivaXaminer", page_icon=":computer:")
    st.title(":computer: AIVivaXaminer")

    # -------------------------------
    # Initialize persistent session state
    # -------------------------------
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
    # Examiner Authentication / Log out
    # -------------------------------
    if st.session_state.examiner_logged_in:
        st.sidebar.success("Examiner logged in")
        if st.sidebar.button("Log out"):
            st.session_state.examiner_logged_in = False
            st.sidebar.info("Logged out. Control panel hidden, session preserved.")
    else:
        password = st.sidebar.text_input("Examiner Password", type="password")
        if password and password == EXAMINER_PASSWORD:
            st.session_state.examiner_logged_in = True
            st.sidebar.success("Examiner authenticated. Control panel unlocked.")
        elif password:
            st.sidebar.error("Incorrect password!")

    # -------------------------------
    # Examiner Control Panel (Sidebar)
    # -------------------------------
    if st.session_state.examiner_logged_in:
        st.sidebar.header("Examiner Control Panel")

        # Max questions
        st.session_state.max_questions = st.sidebar.number_input(
            "Max questions", min_value=1, value=st.session_state.max_questions
        )

        # Manual override: Force Stop
        st.sidebar.markdown("**Manual Override**")
        force_stop = st.sidebar.button("Force Stop Viva")
        if force_stop:
            st.session_state.viva_active = False
            st.warning("Viva forcibly stopped by examiner.")

    # -------------------------------
    # Display chat messages
    # -------------------------------
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Stop if viva inactive
    if not st.session_state.viva_active:
        st.info("Viva session has ended. Thank you!")
        return

    # Accept user input
    if user_input := st.chat_input("Enter your reserach title to start (or type 'end viva' to finish):"):
        if user_input.strip().lower() == "end viva":
            st.session_state.viva_active = False
            st.success("Viva session ended by the student.")
            return

        # Add user message
        with st.chat_message("user"):
            st.markdown(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})

        # Generate assistant response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            assistant_response = generate_response(user_input)
            for chunk in assistant_response.split():
                full_response += chunk + " "
                time.sleep(0.05)
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})

        # Increment question count and check max
        st.session_state.question_count += 1
        if st.session_state.question_count >= st.session_state.max_questions:
            st.session_state.viva_active = False
            st.warning("Maximum number of questions reached. Viva session ended.")

    if st.session_state.viva_completed:
        averages, overall, rec = compute_final_result(st.session_state.scores)
        pdf = generate_viva_pdf(
            st.session_state.question_history,
            st.session_state.student_responses,
            averages, overall, rec
        )
        st.markdown(f"### 🧾 Final Recommendation: **{rec}**")
        with open(pdf, "rb") as f:
            st.download_button("📄 Download Viva Report (PDF)", f, "AIViva_Report.pdf")
        st.stop()

if __name__ == '__main__':
    main()


