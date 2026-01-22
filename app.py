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

# 1. Vectorise the student-teacher response CSV data
loader = CSVLoader(file_path="ts_response.csv")
documents = loader.load()

embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(documents, embeddings)

# 2. Function for similarity search
def retrieve_info(query):
    similar_response = db.similarity_search(query, k=3)
    return [doc.page_content for doc in similar_response]

# 3. Setup LLMChain & prompts
llm = ChatOpenAI(temperature=0.7, model="gpt-4-turbo")

template = """
You are AIVivaXaminer, an AI-based academic examiner conducting a structured undergraduate viva assessment.

────────────────────────────────────
RULES
────────────────────────────────────
1. Ask ONE question at a time.
2. Wait for the student’s full response before proceeding.
3. NEVER repeat a question. Check against the list below before generating a new question.
4. Provide examiner-style qualitative feedback internally only.
5. Scores are INTERNAL only; NEVER reveal them.
6. Maintain a professional, academic examiner tone.

────────────────────────────────────
QUESTION HISTORY
────────────────────────────────────
Previously asked questions:
{question_history}

────────────────────────────────────
QUESTION STRATEGY
────────────────────────────────────
- Categories: General Understanding, Technical Depth (if applicable), Critical Thinking, Domain-Specific Inquiry, Future Scope & Application
- Start from foundational knowledge and progress to higher-order reasoning.
- Select questions NOT in {question_history}.

────────────────────────────────────
ASSESSMENT FRAMEWORK (INTERNAL ONLY)
────────────────────────────────────
Evaluate each response using these dimensions:
1. Conceptual Understanding
2. Methodological Rigor
3. Technical Depth (if applicable)
4. Critical Thinking
5. Communication & Academic Articulation

Scoring scale (internal only):
0 = Not demonstrated
1 = Weak
2 = Adequate
3 = Good
4 = Excellent

────────────────────────────────────
STOPPING RULES
────────────────────────────────────
Terminate the viva when ANY of the following occurs:
1. Minimum category coverage:
   - ≥2 general understanding questions
   - ≥2 technical questions (if applicable)
   - ≥1 critical thinking question
   - ≥1 future-oriented question
2. Performance stabilization:
   - Average scores change ≤ ±0.5 across three consecutive questions
3. Knowledge exhaustion:
   - Two consecutive weak responses (score ≤ 1) in the same dimension
4. Sustained excellence:
   - Average score ≥ 3.5 for three consecutive questions
5. Maximum question limit:
   - Undergraduate viva: 8–12 questions
   - Capstone project: up to 15 questions
6. Two consecutive non-substantive or irrelevant responses

────────────────────────────────────
OUTPUT RULES
────────────────────────────────────
- During viva: output ONLY the next question.
- After stopping: output ONLY the final evaluation report containing:
  • Overall performance level
  • Key strengths
  • Areas for improvement
  • Final recommendation (Pass / Pass with Minor Revisions / Borderline / Fail)
"""

prompt = PromptTemplate(
    input_variables=["message", "best_practice", "question_history"],
    template=template
)

chain = LLMChain(llm=llm, prompt=prompt)

# 4. Retrieval augmented generation
def generate_response(message, question_history):
    best_practice = retrieve_info(message)
    response = chain.run(
        message=message, 
        best_practice=best_practice, 
        question_history="\n".join(question_history)
    )
    return response

# 5. Build an app with Streamlit
def main():
    st.set_page_config(page_title="AIVivaXaminer", page_icon=":computer:")
    st.title(":computer: AIVivaXaminer")

    # Initialize chat and question history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "question_history" not in st.session_state:
        st.session_state.question_history = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if user_input := st.chat_input("Enter your research title or question:"):
        # Display user message in chat
        with st.chat_message("user"):
            st.markdown(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})

        # Generate assistant response with question deduplication
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            assistant_response = generate_response(user_input, st.session_state.question_history)

            # Add the question to history to prevent repeats
            st.session_state.question_history.append(assistant_response.strip())

            # Simulate typing effect
            for chunk in assistant_response.split():
                full_response += chunk + " "
                time.sleep(0.05)
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)

        st.session_state.messages.append({"role": "assistant", "content": full_response})

if __name__ == "__main__":
    main()
