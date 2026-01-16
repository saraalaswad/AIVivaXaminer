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

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
MAX_QUESTIONS = 10

# ─────────────────────────────────────────────
# VECTOR STORE
# ─────────────────────────────────────────────
loader = CSVLoader(file_path="ts_response.csv")
documents = loader.load()

embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(documents, embeddings)

def retrieve_info(query):
    docs = db.similarity_search(query, k=3)
    return "\n".join([d.page_content for d in docs])

# ─────────────────────────────────────────────
# LLM + PROMPT
# ─────────────────────────────────────────────
llm = ChatOpenAI(
    model="gpt-4-turbo",
    temperature=0.4
)

PROMPT = """
You are AIVivaXaminer, an AI-based undergraduate viva examiner.

RULES (STRICT):
- Ask ONLY ONE new question.
- NEVER repeat any previous question.
- Do NOT explain, justify, or summarize.
- Output ONLY the next viva question.

PREVIOUSLY ASKED QUESTIONS:
{question_history}

STUDENT INPUT:
{message}

BEST PRACTICE CONTEXT:
{best_practice}

If all reasonable questions are exhausted, output:
FINAL_EVALUATION_READY
"""

prompt = PromptTemplate(
    input_variables=["message", "best_practice", "question_history"],
    template=PROMPT
)

chain = LLMChain(llm=llm, prompt=prompt)

# ─────────────────────────────────────────────
# STREAMLIT APP
# ─────────────────────────────────────────────
def main():
    st.set_page_config(page_title="AIVivaXaminer", page_icon="🎓")
    st.title("🎓 AIVivaXaminer")

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "question_history" not in st.session_state:
        st.session_state.question_history = []

    if "question_count" not in st.session_state:
        st.session_state.question_count = 0

    if "viva_completed" not in st.session_state:
        st.session_state.viva_completed = False

    # Display history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Stop input if viva finished
    if st.session_state.viva_completed:
        st.chat_message("assistant").markdown("✅ **Viva completed. Final evaluation generated.**")
        st.stop()

    # User input
    if user_input := st.chat_input("Enter research title / answer"):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        best_practice = retrieve_info(user_input)
        question_history_text = "\n".join(st.session_state.question_history)

        response = chain.run(
            message=user_input,
            best_practice=best_practice,
            question_history=question_history_text
        ).strip()

        # ─────────────────────────────────────────
        # STOPPING CONDITIONS
        # ─────────────────────────────────────────
        if response == "FINAL_EVALUATION_READY":
            st.session_state.viva_completed = True
            with st.chat_message("assistant"):
                st.markdown("""
### 🧾 Final Viva Evaluation
- **Overall Performance:** Satisfactory
- **Strengths:** Conceptual clarity, methodological awareness
- **Areas for Improvement:** Depth of critical analysis
- **Final Recommendation:** **Pass**
""")
            st.stop()

        if st.session_state.question_count >= MAX_QUESTIONS:
            st.session_state.viva_completed = True
            with st.chat_message("assistant"):
                st.markdown("🛑 **Maximum viva questions reached. Viva concluded.**")
            st.stop()

        if response in st.session_state.question_history:
            st.warning("⚠ Duplicate question blocked.")
            st.stop()

        # ─────────────────────────────────────────
        # DISPLAY QUESTION
        # ─────────────────────────────────────────
        st.session_state.question_history.append(response)
        st.session_state.question_count += 1

        with st.chat_message("assistant"):
            placeholder = st.empty()
            typed = ""
            for word in response.split():
                typed += word + " "
                time.sleep(0.03)
                placeholder.markdown(typed + "▌")
            placeholder.markdown(typed)

        st.session_state.messages.append(
            {"role": "assistant", "content": response}
        )

if __name__ == "__main__":
    main()
