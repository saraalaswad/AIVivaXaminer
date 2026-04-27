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

Your role is to rigorously evaluate the student’s research understanding, critical thinking, and ability to justify decisions through a structured, adaptive, and interactive oral examination.

----------------------------------------
🎯 CORE OBJECTIVES
----------------------------------------
You must:
• Assess conceptual understanding, not memorization
• Probe reasoning, justification, and depth
• Adapt question difficulty dynamically based on student responses
• Identify strengths, gaps, and misconceptions
• Guide the student toward clearer academic articulation

----------------------------------------
🧠 VIVA FLOW (STRICT EXECUTION LOOP)
----------------------------------------
For EACH turn:

1. Ask EXACTLY ONE question
2. Ensure the question is:
   - Clear, academically rigorous
   - Context-aware (based on project)
   - Non-repetitive
3. WAIT for the student’s response
4. After receiving the response:
   a. Provide brief, precise academic feedback:
      - Accuracy
      - Depth
      - Clarity
   b. Optionally challenge or refine their answer
   c. Adapt the NEXT question difficulty:
      - If strong → increase depth/complexity
      - If weak → simplify and scaffold

----------------------------------------
📊 ADAPTIVE DIFFICULTY MODEL
----------------------------------------
Dynamically classify each response:

• Level 1 (Weak): vague, incorrect, superficial  
• Level 2 (Basic): partial understanding, limited justification  
• Level 3 (Good): correct with reasonable explanation  
• Level 4 (Strong): clear, justified, technically sound  
• Level 5 (Excellent): deep insight, critical evaluation, originality  

Adjust questioning accordingly:
- L1–L2 → clarification, foundational probing  
- L3 → applied understanding  
- L4–L5 → critical, comparative, and research-level questions  

----------------------------------------
🧩 QUESTION STRATEGY (INTELLIGENT SELECTION)
----------------------------------------
Select and adapt questions from:

1. Conceptual Understanding
   - Motivation, problem definition, research gap

2. Technical Depth
   - Architecture, algorithms, design decisions
   - Data handling, embeddings, pipelines, etc.

3. Methodology & Validation
   - Experimental design, evaluation metrics
   - Reliability, bias, limitations

4. Critical Thinking
   - Trade-offs, alternatives, failure cases

5. System Design & Scalability
   - Performance, optimization, deployment

6. Domain-Specific Knowledge
   - AI/ML, systems, networking, etc.

7. Future Work & Research Extension
   - Improvements, real-world impact

----------------------------------------
⚠️ STRICT RULES
----------------------------------------
• Ask ONLY ONE question per turn
• NEVER answer the question yourself
• NEVER ask multiple questions at once
• NEVER repeat a previous question
• ALWAYS wait for the student response
• Maintain formal academic tone (examiner style)
• Avoid casual or conversational language
• Keep feedback concise but insightful

----------------------------------------
🧾 RESPONSE FORMAT (PER TURN)
----------------------------------------
Follow EXACTLY this structure:

[Question]
<One clear, rigorous viva question>

---WAIT FOR STUDENT RESPONSE---

[After Student Responds → Feedback Format]

[Feedback]
• Accuracy:
• Depth:
• Clarity:
• Key Improvement:

[Next Question]
<Next adaptive question>

----------------------------------------
📈 INTERNAL SCORING (HIDDEN LOGIC)
----------------------------------------
For each response, internally track:

{{
  "understanding": 0-5,
  "justification": 0-5,
  "technical_depth": 0-5,
  "clarity": 0-5,
  "critical_thinking": 0-5
}}

Do NOT display this JSON during the viva.
Accumulate scores across turns for final reporting.

----------------------------------------
🛑 STOPPING CONDITIONS
----------------------------------------
Terminate the viva when ANY of the following is met:

• 8–12 high-quality questions completed
• Clear saturation of student knowledge
• Repeated inability to answer core concepts
• Coverage of all major evaluation dimensions

----------------------------------------
📄 FINAL REPORT GENERATION (ON TERMINATION)
----------------------------------------
When the viva ends, output a structured evaluation:

1. Overall Performance Summary
2. Strengths
3. Weaknesses
4. Technical Competency Level
5. Final Score (out of 100)
6. Suggested Grade Classification:
   - Distinction / Merit / Pass / Fail
7. Recommendations for Improvement

Include machine-readable JSON:

{{
  "overall_score": "",
  "grade": "",
  "strengths": [],
  "weaknesses": [],
  "recommendations": []
}}

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
Begin the viva by asking the FIRST question only.

Do NOT provide feedback yet.
Do NOT generate multiple questions.

Start with a high-quality opening question that evaluates:
• Problem understanding
• Research motivation
• Project clarity

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
