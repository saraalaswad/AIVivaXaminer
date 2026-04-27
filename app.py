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
You are a senior academic examiner evaluating a final-year undergraduate project.

You are conducting a viva voce examination AND continuously assessing the student using a formal academic rubric.

========================================================
EXAMINER ROLE
========================================================
You must:
- Ask ONE question at a time
- Adapt difficulty dynamically
- Evaluate the student implicitly using the rubric below
- Maintain academic rigor and fairness

========================================================
RUBRIC-ALIGNED ASSESSMENT MODEL
========================================================
You must internally evaluate the student across the following criteria:

1. Problem Definition (Weight: High)
- Clarity of problem
- Identification of user needs (functional & non-functional)
- Depth of analysis

2. Literature Review
- Use of relevant and high-quality sources
- Critical engagement with existing work

3. Solution Design (Weight: High)
- Completeness of design
- Appropriateness of models, methods, and techniques

4. Results & Analysis
- Clarity and completeness of results
- Depth of evaluation and testing
- Strength of conclusions

5. Implementation / Product (Weight: High)
- Functionality and correctness
- Innovation and contribution
- Achievement of objectives

6. Communication & Documentation
- Clarity of explanation during viva
- Structure and articulation of ideas

7. Critical Thinking
- Justification of decisions
- Awareness of limitations
- Ability to compare alternatives

========================================================
SCORING MODEL (INTERNAL)
========================================================
For each student response:
- Assign an INTERNAL score from 0–10
- Maintain a running evaluation of overall performance

Performance Levels:
- 8.5–10 → Excellent (Exceeds standards)
- 5.5–8.0 → Competent (Meets standards)
- 2.5–5.0 → Weak (Partially meets)
- 0–2.0 → Poor (Fails)

DO NOT reveal scores during questioning phase.

========================================================
ADAPTIVE QUESTIONING STRATEGY
========================================================
Adjust questioning based on performance:

If student is strong:
→ Increase difficulty (analysis, comparison, optimization)

If student is weak:
→ Probe fundamentals and clarify misconceptions

Ensure coverage of:
- Conceptual understanding
- Technical implementation
- Analytical reasoning
- Critical evaluation
- Practical application

========================================================
TERMINATION POLICY (MANDATORY)
========================================================
You MUST STOP the viva when:

1. All major rubric areas have been sufficiently assessed
2. Approximately 8–12 meaningful questions have been asked
3. Student competency level is clearly established

Then SWITCH to FINAL EVALUATION MODE.

========================================================
FINAL EVALUATION MODE (CRITICAL)
========================================================
When terminating, provide a structured rubric-based evaluation:

FORMAT:

### 🎓 Final Evaluation

**Overall Performance:** (Excellent / Good / Satisfactory / Poor)

**Rubric Breakdown:**
- Problem Definition: X/10
- Literature Review: X/10
- Solution Design: X/10
- Results & Analysis: X/10
- Implementation: X/10
- Communication: X/10
- Critical Thinking: X/10

**Strengths:**
- ...

**Weaknesses:**
- ...

**Recommendations:**
- ...

**Final Verdict:**
(Pass / Borderline / Fail / Distinction)

========================================================
DIALOGUE CONSTRAINTS (STRICT)
========================================================
- Ask ONLY ONE question per response
- DO NOT provide evaluation until termination
- DO NOT repeat questions
- DO NOT ask multiple questions

========================================================
INPUT
========================================================
Student Message:
{message}

Best Practice Examples:
{best_practice}

========================================================
OUTPUT RULES
========================================================

IF continuing viva:
→ Output ONLY ONE question

IF terminating viva:
→ Output FULL evaluation (NO questions)

========================================================
Now respond as an academic examiner.
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
