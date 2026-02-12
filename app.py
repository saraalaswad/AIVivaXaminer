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
# 1. Vectorstore caching for performance
# -------------------------------
@st.cache_resource
def load_vectorstore():
    loader = CSVLoader(file_path="ts_response.csv")
    documents = loader.load()
    embeddings = OpenAIEmbeddings()
    return FAISS.from_documents(documents, embeddings)

db = load_vectorstore()

# -------------------------------
# 2. Similarity search function
# -------------------------------
def retrieve_info(query, k=3):
    try:
        similar_response = db.similarity_search(query, k=k)
        return [doc.page_content for doc in similar_response]
    except Exception as e:
        st.error(f"Error retrieving best practices: {e}")
        return []

# -------------------------------
# 3. LLM & prompt setup
# -------------------------------
llm = ChatOpenAI(temperature=0.7, model="gpt-4-turbo")

template = """
You are an experienced academic professor conducting a viva for an undergraduate student. 
Your goal is to evaluate the student’s understanding by asking questions one at a time, then discussing their answer with constructive feedback.

You have been provided:
• The student’s message: {message}
• Best practices for responding: {best_practice}

Instructions:
1. Ask one question at a time, wait for the student’s full answer before moving on.
2. Avoid repeating questions.
3. Maintain a supportive but challenging tone.
4. Align style and logic with best practices.
5. Focus on selected question categories: {categories}

Conversation History:
{history}

Generate the next viva question along with brief guidance to the student.
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
# 4. Streamlit app
# -------------------------------
EXAMINER_PASSWORD = os.getenv("EXAMINER_PASSWORD", "exam123")  # fallback

def main():
    st.set_page_config(page_title="AIVivaXaminer", page_icon=":computer:")
    st.title(":computer: AIVivaXaminer")

    # -------------------------------
    # Initialize session state
    # -------------------------------
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

    # -------------------------------
    # Examiner Authentication
    # -------------------------------
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

    # -------------------------------
    # Examiner Control Panel
    # -------------------------------
    if st.session_state.examiner_logged_in:
        st.sidebar.header("Examiner Control Panel")

        st.session_state.max_questions = st.sidebar.number_input(
            "Max questions", min_value=1, value=st.session_state.max_questions
        )

        st.session_state.similarity_k = st.sidebar.number_input(
            "Number of best practice examples to retrieve", min_value=1, value=st.session_state.similarity_k
        )

        st.session_state.question_category = st.sidebar.selectbox(
            "Question category", ["General", "Technical", "Problem-Solving/Critical Thinking",
                                  "Domain-Specific", "Future Scope"]
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

    # Stop if viva inactive
    if not st.session_state.viva_active:
        st.info("Viva session has ended. Thank you!")
        return

    # -------------------------------
    # Accept user input
    # -------------------------------
    if user_input := st.chat_input("Enter your research title to start (or type 'end viva' to finish):"):
        if user_input.strip().lower() == "end viva":
            st.session_state.viva_active = False
            st.success("Viva session ended by the student.")
            return

        # Append user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Generate assistant response with context
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

        # Stream response in chunks
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            chunks = re.findall(r'.{1,80}(?:\s|$)', assistant_response)
            for chunk in chunks:
                full_response += chunk
                message_placeholder.markdown(full_response + "▌")
                time.sleep(0.05)
            message_placeholder.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})

        # Track question count and prevent repeats
        st.session_state.question_count += 1
        if st.session_state.question_count >= st.session_state.max_questions:
            st.session_state.viva_active = False
            st.warning("Maximum number of questions reached. Viva session ended.")

        # Display progress bar
        st.progress(min(st.session_state.question_count / st.session_state.max_questions, 1.0))

        # Option to export transcript
        transcript_df = pd.DataFrame(st.session_state.messages)
        st.download_button(
            "Export Viva Transcript",
            data=transcript_df.to_csv(index=False),
            file_name="viva_transcript.csv"
        )

if __name__ == '__main__':
    main()
