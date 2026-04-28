import streamlit as st
import time
from datetime import datetime
from dotenv import load_dotenv

from langchain.document_loaders import CSVLoader
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain

from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

# --------------------------------------------------
# ENV
# --------------------------------------------------
load_dotenv()

# --------------------------------------------------
# LOGIN
# --------------------------------------------------
EXAMINER_USERNAME = "examiner"
EXAMINER_PASSWORD = "1234"

# --------------------------------------------------
# VECTOR DB
# --------------------------------------------------
@st.cache_resource
def initialize_vector_db():
    loader = CSVLoader(file_path="ts_response.csv")
    docs = loader.load()

    embeddings = OpenAIEmbeddings()
    return FAISS.from_documents(docs, embeddings)

db = initialize_vector_db()

def retrieve_info(query, k=3):
    results = db.similarity_search(query, k=k)
    return [r.page_content for r in results]

# --------------------------------------------------
# LLM
# --------------------------------------------------
llm = ChatOpenAI(model="gpt-4-turbo", temperature=0.7)

# --------------------------------------------------
# CATEGORY CONTROL (PYTHON IS NOW THE BOSS)
# --------------------------------------------------
CATEGORIES = [
    "General Understanding",
    "Technical Understanding",
    "Methodology & Testing",
    "Problem-Solving",
    "System Thinking",
    "Reflection & Limitations",
    "Real-World Application"
]

def get_current_category():
    idx = st.session_state.viva_state["current_category_index"] - 1
    return CATEGORIES[idx]

def advance_category():
    st.session_state.viva_state["current_category_index"] += 1
    st.session_state.viva_state["follow_up_allowed"] = False

# --------------------------------------------------
# PROMPT (LLM HAS NO CONTROL OVER FLOW)
# --------------------------------------------------
PROMPT_TEMPLATE = """
You are an expert academic examiner conducting a formal undergraduate viva examination.

Your role is to evaluate the student’s understanding of their research project through a structured, interactive, and adaptive oral assessment.

----------------------------------------
🎯 CORE OBJECTIVES
----------------------------------------
You must:
• Assess conceptual understanding and practical knowledge
• Evaluate the student’s ability to explain and justify their work
• Encourage clear academic communication
• Identify strengths, gaps, and misconceptions
• Maintain a supportive but academically rigorous tone

IMPORTANT:
Assume the student is an undergraduate.
Do NOT expect postgraduate-level originality.

This is a time-limited viva.
Breadth of coverage is more important than deep focus on one topic.

Re-asking about the project overview or problem statement after the first question is considered a CRITICAL ERROR.

----------------------------------------
🧭 VIVA STRUCTURE (STRICT)
----------------------------------------

Total Questions Allowed: 8–10 ONLY

Categories (MUST ALL be covered in order):

1. General Understanding  
2. Technical Understanding  
3. Methodology & Testing  
4. Problem-Solving  
5. System Thinking  
6. Reflection & Limitations  
7. Real-World Application  

----------------------------------------
📊 CURRENT CATEGORY (SYSTEM CONTROLLED)
----------------------------------------
Current Category: {current_category}

You MUST follow this category strictly after the first question.

----------------------------------------
🚨 CATEGORY LOCKING & PROGRESSION (CRITICAL)
----------------------------------------

You MUST strictly follow category order.

Rules:
• You are allowed to ask questions from ONLY ONE active category at a time
• Once you move to the next category, you MUST NOT go back
• DO NOT skip categories
• DO NOT reset to Category 1 under any condition

----------------------------------------
🔁 QUESTION LIMIT PER CATEGORY
----------------------------------------

For EACH category:

• Ask 1 core question (mandatory)
• Ask MAXIMUM 1 follow-up question ONLY if:
  - The answer is weak (Level 1–2), OR
  - The answer lacks clarity

Otherwise:
• MOVE to next category immediately

ABSOLUTE RULE:
• Maximum 2 questions per category

----------------------------------------
❌ FORBIDDEN REPETITION RULE
----------------------------------------

You are STRICTLY FORBIDDEN from repeating or rephrasing:

• Project overview
• Problem statement
• General description

These are allowed ONLY in Category 1.

After leaving Category 1:
• NEVER ask them again
• NEVER rephrase them in any form

----------------------------------------
🧱 CATEGORY-SPECIFIC QUESTION TYPES
----------------------------------------

Each category MUST have distinct focus:

1. General → overview, motivation (ONLY HERE)
2. Technical → architecture, tools, implementation
3. Methodology → testing, evaluation, validation
4. Problem-Solving → challenges, debugging
5. System → performance, scalability, data flow
6. Reflection → limitations, improvements
7. Application → real-world use, impact

If a question resembles another category → REWRITE it.

----------------------------------------
📊 ADAPTIVE DIFFICULTY MODEL (UNDERGRADUATE)
----------------------------------------

Classify responses:

• Level 1 (Weak) → incorrect or unclear  
• Level 2 (Basic) → partial understanding  
• Level 3 (Competent) → correct with basic explanation  
• Level 4 (Strong) → clear reasoning and justification  
• Level 5 (Outstanding UG) → strong explanation + some critical thinking  

Adapt difficulty:
- L1–L2 → simplify and guide  
- L3 → apply understanding  
- L4–L5 → ask “why”, “what if”, comparisons  

IMPORTANT:
Do NOT increase number of questions—only adjust depth.

----------------------------------------
🧠 INTERNAL STATE TRACKING (MANDATORY)
----------------------------------------

Internally maintain:

{
  "current_category_index": 1,
  "questions_asked_total": 0,
  "questions_per_category": [0,0,0,0,0,0,0]
}

Category index mapping:
1=General, 2=Technical, 3=Methodology, 4=Problem-Solving,
5=System, 6=Reflection, 7=Application

----------------------------------------
➡️ TRANSITION RULE
----------------------------------------

Move to next category when:

• 1 question asked AND answer is acceptable (Level 3+), OR  
• 2 questions already asked (max reached)

----------------------------------------
🧾 RESPONSE FORMAT
----------------------------------------

FIRST TURN:
[Question]
<ONE question from Category 1 (General Understanding)>

WAIT for student response.

AFTER EACH RESPONSE:

[Feedback]
• Accuracy:
• Depth:
• Clarity:
• Key Improvement:

[Next Question]
<ONE question from CURRENT category OR next category>

----------------------------------------
🛑 TERMINATION RULE
----------------------------------------

End the viva ONLY when:

• Total questions = 8–10 AND  
• ALL 7 categories are covered  

DO NOT exceed 10 questions.

----------------------------------------
📄 FINAL REPORT
----------------------------------------

Provide:

1. Overall Performance Summary  
2. Strengths  
3. Areas for Improvement  
4. Technical Understanding Level  
5. Final Score (out of 100)  
6. Suggested Grade:
   - Distinction / Merit / Pass / Fail  

Include JSON:

{
  "overall_score": "",
  "grade": "",
  "strengths": [],
  "weaknesses": [],
  "recommendations": []
}

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

Start the viva by asking the FIRST question ONLY.

Do NOT provide feedback.
Do NOT ask multiple questions.

The first question MUST be from Category 1 ONLY.
"""

prompt = PromptTemplate(
    input_variables=[
        "message",
        "best_practice",
        "current_category"
    ],
    template=PROMPT_TEMPLATE
)

chain = LLMChain(llm=llm, prompt=prompt)

# --------------------------------------------------
# STATE ENGINE (CRITICAL FIX)
# --------------------------------------------------
def init_state():
    if "viva_state" not in st.session_state:
        st.session_state.viva_state = {
            "current_category_index": 1,
            "questions_asked_total": 0,
            "questions_per_category": [0,0,0,0,0,0,0],
            "follow_up_allowed": False
        }

def update_state(user_input):
    state = st.session_state.viva_state

    state["questions_asked_total"] += 1

    idx = state["current_category_index"] - 1
    state["questions_per_category"][idx] += 1

    # simple follow-up trigger
    weak_words = ["not sure", "don't know", "unclear", "maybe"]

    state["follow_up_allowed"] = any(w in user_input.lower() for w in weak_words)

    # FORCE CATEGORY SHIFT (max 2 questions per category)
    if state["questions_per_category"][idx] >= 2:
        if state["current_category_index"] < 7:
            advance_category()

# --------------------------------------------------
# RESPONSE GENERATION
# --------------------------------------------------
def generate_response(message):
    best_practice = retrieve_info(message)
    state = st.session_state.viva_state

    response = chain.run(
        message=message,
        best_practice=best_practice,
        current_category=get_current_category()
    )

    return response

# --------------------------------------------------
# PDF
# --------------------------------------------------
def generate_pdf(chat_history, filename="viva.pdf"):
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("AI Viva Transcript", styles["Title"]))
    story.append(Spacer(1, 12))

    for msg in chat_history:
        role = "Student" if msg["role"] == "user" else "Examiner"
        text = msg["content"].replace("\n", "<br/>")

        story.append(Paragraph(f"<b>{role}:</b><br/>{text}", styles["Normal"]))
        story.append(Spacer(1, 10))

    doc = SimpleDocTemplate(filename, pagesize=A4)
    doc.build(story)

    return filename

# --------------------------------------------------
# APP
# --------------------------------------------------
def main():
    st.set_page_config(page_title="AIVivaXaminer", page_icon="🎓")

    st.title("🎓 AIVivaXaminer (Fixed Version)")

    init_state()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "viva_active" not in st.session_state:
        st.session_state.viva_active = True

    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False

    # ---------------- SIDEBAR ----------------
    with st.sidebar:
        st.header("Control Panel")

        if not st.session_state.logged_in:
            u = st.text_input("Username")
            p = st.text_input("Password", type="password")

            if st.button("Login"):
                if u == EXAMINER_USERNAME and p == EXAMINER_PASSWORD:
                    st.session_state.logged_in = True
                    st.rerun()
        else:
            st.success("Logged In")

            st.write("Current Category:", get_current_category())

            if st.button("Reset"):
                st.session_state.viva_state = {
                    "current_category_index": 1,
                    "questions_asked_total": 0,
                    "questions_per_category": [0,0,0,0,0,0,0],
                    "follow_up_allowed": False
                }
                st.session_state.messages = []
                st.rerun()

    # ---------------- CHAT HISTORY ----------------
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    user_input = st.chat_input("Enter response...") if st.session_state.viva_active else None

    if user_input:
        st.chat_message("user").markdown(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})

        with st.chat_message("assistant"):
            placeholder = st.empty()

            response = generate_response(user_input)

            text = ""
            for w in response.split():
                text += w + " "
                time.sleep(0.01)
                placeholder.markdown(text + "▌")

            placeholder.markdown(text)

        st.session_state.messages.append({"role": "assistant", "content": response})

        # 🔥 CRITICAL FIX: UPDATE STATE HERE
        update_state(user_input)


if __name__ == "__main__":
    main()
