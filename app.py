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

Student message: {message}
Best practices: {best_practice}

Instructions:
- Ask one question at a time.
- Maintain supportive but challenging tone.
- Adapt to viva type: {viva_type}
- Adjust question difficulty: {difficulty_level}
- Follow best practices.
- Do not repeat questions.

Task: Generate the next viva question and guidance for the student.
"""

prompt = PromptTemplate(
    input_variables=["message", "best_practice", "viva_type", "difficulty_level"],
    template=template
)

chain = LLMChain(llm=llm, prompt=prompt)

def generate_response(message, viva_type, difficulty_level):
    best_practice = retrieve_info(message)
    return chain.run(message=message, best_practice=best_practice,
                     viva_type=viva_type, difficulty_level=difficulty_level)

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
        "max_questions": 10,
        "difficulty_level": "Medium",
        "viva_type": "Project"
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
            # DO NOT reset any other session variables!
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

        st.session_state.max_questions = st.sidebar.number_input(
            "Max questions", min_value=1, value=st.session_state.max_questions
        )
        st.session_state.difficulty_level = st.sidebar.select_slider(
            "Difficulty level", ["Easy", "Medium", "Hard"], value=st.session_state.difficulty_level
        )
        st.session_state.viva_type = st.sidebar.selectbox(
            "Viva type", ["Project", "Thesis", "Capstone"],
            index=["Project", "Thesis", "Capstone"].index(st.session_state.viva_type)
        )

        st.sidebar.markdown("**Manual Overrides**")
        force_stop = st.sidebar.button("Force Stop Viva")
        skip_question = st.sidebar.button("Skip Question")

        if force_stop:
            st.session_state.viva_active = False
            st.warning("Viva forcibly stopped by examiner.")
    else:
        # Use stored settings even if logged out
        max_questions = st.session_state.max_questions
        difficulty_level = st.session_state.difficulty_level
        viva_type = st.session_state.viva_type
        skip_question = False

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
    if user_input := st.chat_input("Your response (or type 'end viva' to finish):"):
        if user_input.strip().lower() == "end viva":
            st.session_state.viva_active = False
            st.success("Viva session ended by the student.")
            return

        # Add user message
        with st.chat_message("user"):
            st.markdown(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})

        # Skip question manually
        if st.session_state.examiner_logged_in and skip_question:
            st.session_state.question_count += 1
            st.info("Examiner skipped this question.")
            return

        # Generate assistant response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            assistant_response = generate_response(
                user_input,
                st.session_state.viva_type,
                st.session_state.difficulty_level
            )
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

if __name__ == '__main__':
    main()
