import os
import streamlit as st
import time
import pandas as pd
import re
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from dotenv import load_dotenv

load_dotenv()

# -------------------------------
# 1. Cache vectorstore
# -------------------------------
@st.cache_resource
def load_vectorstore():
    loader = CSVLoader(file_path="ts_response.csv")
    documents = loader.load()
    embeddings = OpenAIEmbeddings()
    return FAISS.from_documents(documents, embeddings)

db = load_vectorstore()

# -------------------------------
# 2. Similarity search
# -------------------------------
def retrieve_info(query, k=3):
    try:
        similar_response = db.similarity_search(query, k=k)
        return [doc.page_content for doc in similar_response]
    except Exception as e:
        st.error(f"Error retrieving best practices: {e}")
        return []

# -------------------------------
# 3. LLM & prompt
# -------------------------------
llm = ChatOpenAI(temperature=0.7, model="gpt-4-turbo")

template = """
You are an experienced academic professor conducting a viva for an undergraduate student. 
Ask one question at a time with brief guidance. Avoid repeating questions.
Align style and logic with provided best practices.

You have been provided:
• The student’s message: {message}
• Best practices: {best_practice}
• Conversation history: {history}
• Question category focus: {categories}

Generate the next viva question with guidance.
"""

prompt = PromptTemplate(
    input_variables=["message", "best_practice", "history", "categories"],
    template=template
)

chain = LLMChain(llm=llm, prompt=prompt)

def generate_response(message, history="", categories="General", k=3):
    best_practice = retrieve_info(message, k)
    return chain.run(message=message, best_practice=best_practice, history=history, categories=categories)

# -------------------------------
# 4. Streamlit App
# -------------------------------
EXAMINER_PASSWORD = os.getenv("EXAMINER_PASSWORD", "exam123")

def main():
    st.set_page_config(page_title="AIVivaXaminer", page_icon=":computer:")
    st.title(":computer: AIVivaXaminer")

    # Session state defaults
    defaults = {
        "examiner_logged_in": False,
        "messages": [],
        "question_count": 0,
        "viva_active": True,
        "max_questions": 10,
        "asked_questions": set(),
        "question_category": "General",
        "similarity_k": 3
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    # Examiner Authentication
    if st.session_state.examiner_logged_in:
        st.sidebar.success("Examiner logged in")
        if st.sidebar.button("Log out"):
            st.session_state.examiner_logged_in = False
            st.sidebar.info("Logged out. Control panel hidden, session preserved.")
    else:
        password = st.sidebar.text_input("Examiner Password", type="password")
        if password:
            if password == EXAMINER_PASSWORD:
                st.session_state.examiner_logged_in = True
                st.sidebar.success("Examiner authenticated. Control panel unlocked.")
            else:
                st.sidebar.error("Incorrect password!")

    # Examiner Control Panel
    if st.session_state.examiner_logged_in:
        st.sidebar.header("Examiner Control Panel")

        st.session_state.max_questions = st.sidebar.number_input(
            "Max questions", min_value=1, value=st.session_state.max_questions
        )

        st.session_state.similarity_k = st.sidebar.number_input(
            "Number of best practice examples", min_value=1, value=st.session_state.similarity_k
        )

        st.session_state.question_category = st.sidebar.selectbox(
            "Question category", ["General", "Technical", "Problem-Solving/Critical Thinking",
                                  "Domain-Specific", "Future Scope"]
        )

        if st.sidebar.button("Force Stop Viva"):
            st.session_state.viva_active = False
            st.warning("Viva forcibly stopped by examiner.")

    # Display messages with role-colored chat bubbles
    for message in st.session_state.messages:
        role = message["role"]
        if role == "user":
            with st.chat_message("user", avatar="🧑"):
                st.markdown(f"<span style='color:blue'>{message['content']}</span>", unsafe_allow_html=True)
        else:
            with st.chat_message("assistant", avatar="👩‍🏫"):
                st.markdown(f"<span style='color:green'>{message['content']}</span>", unsafe_allow_html=True)

    if not st.session_state.viva_active:
        st.info("Viva session has ended. Thank you!")
        return

    # Accept user input
    if user_input := st.chat_input("Enter your research title to start (or 'end viva')"):
        if user_input.strip().lower() == "end viva":
            st.session_state.viva_active = False
            st.success("Viva session ended by the student.")
            return

        # Append student message
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user", avatar="🧑"):
            st.markdown(f"<span style='color:blue'>{user_input}</span>", unsafe_allow_html=True)

        # Generate assistant response
        conversation_history = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages])
        try:
            assistant_response = generate_response(
                user_input,
                history=conversation_history,
                categories=st.session_state.question_category,
                k=st.session_state.similarity_k
            )
        except Exception as e:
            st.error(f"Error generating response: {e}")
            assistant_response = "An error occurred. Please try again."

        # Streaming simulation (sentence by sentence)
        with st.chat_message("assistant", avatar="👩‍🏫") as message_placeholder:
            full_response = ""
            sentences = re.findall(r'.*?[.!?]\s', assistant_response)
            if not sentences:
                sentences = [assistant_response]
            for sentence in sentences:
                full_response += sentence
                message_placeholder.markdown(f"<span style='color:green'>{full_response}▌</span>", unsafe_allow_html=True)
                time.sleep(0.2)
            message_placeholder.markdown(f"<span style='color:green'>{full_response}</span>", unsafe_allow_html=True)

        st.session_state.messages.append({"role": "assistant", "content": full_response})

        # Increment question count & progress
        st.session_state.question_count += 1
        st.progress(min(st.session_state.question_count / st.session_state.max_questions, 1.0))

        if st.session_state.question_count >= st.session_state.max_questions:
            st.session_state.viva_active = False
            st.warning("Maximum questions reached. Viva ended.")

        # Export transcript
        transcript_df = pd.DataFrame(st.session_state.messages)
        st.download_button(
            "Export Viva Transcript",
            data=transcript_df.to_csv(index=False),
            file_name="viva_transcript.csv"
        )

if __name__ == '__main__':
    main()
