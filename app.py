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
# 1. Load & Vectorise CSV Q&A data
# --------------------------------------------------
loader = CSVLoader(file_path="ts_response.csv")
documents = loader.load()

embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(documents, embeddings)

# --------------------------------------------------
# 2. Retrieval with HARD filtering
# --------------------------------------------------
def retrieve_unasked_docs(query):
    docs = db.similarity_search(query, k=6)
    filtered = []

    for doc in docs:
        q = doc.metadata.get("examiner_question", "")
        if q not in st.session_state.asked_questions:
            filtered.append(doc)

    return filtered


def select_next_doc(docs):
    # Prefer unseen categories
    for doc in docs:
        cat = doc.metadata.get("category", "General")
        if cat not in st.session_state.asked_categories:
            return doc

    return docs[0] if docs else None


# --------------------------------------------------
# 3. LLM & Prompt (HARDENED)
# --------------------------------------------------
llm = ChatOpenAI(
    temperature=0.6,
    model="gpt-4-turbo"
)

template = """
You are an experienced academic professor conducting a viva for an undergraduate student.

You have been provided:
- Student input: {message}
- Retrieved Q&A examples (none have been previously asked): {retrieved_qa}
- Categories already covered: {asked_categories}

Instructions:
1. Select ONE question only.
2. Do NOT repeat or paraphrase any previous question.
3. Prefer a category that has not yet been covered.
4. Maintain a supportive but academically rigorous tone.
5. Ask exactly ONE question.
6. Provide brief guidance explaining what a strong answer should include.

Output format (STRICT):
Category:
Question:
Guidance:
"""

prompt = PromptTemplate(
    input_variables=["message", "retrieved_qa", "asked_categories"],
    template=template
)

chain = LLMChain(llm=llm, prompt=prompt)


def generate_response(message):
    docs = retrieve_unasked_docs(message)
    selected = select_next_doc(docs)

    if not selected:
        return "Category:\nGeneral\nQuestion:\nThank you. This concludes the viva.\nGuidance:\n"

    retrieved_qa = selected.page_content
    category = selected.metadata.get("category", "General")
    question = selected.metadata.get("examiner_question")

    response = chain.run(
        message=message,
        retrieved_qa=retrieved_qa,
        asked_categories=list(st.session_state.asked_categories)
    )

    # HARD memory update
    st.session_state.asked_questions.add(question)
    st.session_state.asked_categories.add(category)

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
    styles.add(ParagraphStyle(name="RoleStyle", fontName="Helvetica-Bold", fontSize=11))
    styles.add(ParagraphStyle(name="ContentStyle", fontSize=11))

    story = []
    story.append(Paragraph("<b>Final Viva Report</b>", styles["Title"]))
    story.append(Spacer(1, 0.3 * inch))
    story.append(Paragraph(f"<b>Date:</b> {datetime.now()}", styles["Normal"]))
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
def main():
    st.set_page_config(page_title="AIVivaXaminer", page_icon="🎓")
    st.title("🎓 AIVivaXaminer")

    # --------------------------------------------------
    # Session State
    # --------------------------------------------------
    defaults = {
        "messages": [],
        "question_count": 0,
        "answer_count": -1,
        "viva_active": True,
        "viva_completed": False,
        "max_questions": 10,
        "asked_questions": set(),
        "asked_categories": set()
    }

    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    # --------------------------------------------------
    # Chat history
    # --------------------------------------------------
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # --------------------------------------------------
    # Chat input
    # --------------------------------------------------
    if st.session_state.viva_active:
        user_input = st.chat_input("Enter your response")

        if user_input:
            st.session_state.messages.append({"role": "user", "content": user_input})
            st.session_state.answer_count += 1

            with st.chat_message("assistant"):
                response = generate_response(user_input)
                animated = ""
                placeholder = st.empty()

                for word in response.split():
                    animated += word + " "
                    time.sleep(0.03)
                    placeholder.markdown(animated + "▌")

                placeholder.markdown(animated)

            st.session_state.messages.append(
                {"role": "assistant", "content": animated}
            )
            st.session_state.question_count += 1

            if st.session_state.question_count >= st.session_state.max_questions:
                st.session_state.viva_active = False
                st.session_state.viva_completed = True

    # --------------------------------------------------
    # PDF Export
    # --------------------------------------------------
    if st.session_state.viva_completed:
        st.markdown("---")
        if st.button("Generate Final Viva Report (PDF)"):
            pdf_path = generate_viva_pdf(st.session_state.messages)
            with open(pdf_path, "rb") as f:
                st.download_button(
                    "⬇️ Download Final Viva Report",
                    f,
                    file_name="Final_Viva_Report.pdf",
                    mime="application/pdf"
                )


if __name__ == "__main__":
    main()

