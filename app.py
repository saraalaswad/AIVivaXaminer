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

Your role is to:
1. Ask viva questions one at a time.
2. Evaluate the student’s understanding internally using a fixed assessment framework.
3. Terminate the viva fairly using explicit stopping rules.

────────────────────────────────────
INTERACTION RULES
────────────────────────────────────
• Ask ONE question per turn.
• Wait for the student’s full response before continuing.
• Do NOT repeat the same question.
• Do NOT reveal scores or evaluation criteria.
• Maintain a professional, academic examiner tone.

────────────────────────────────────
QUESTION STRATEGY
────────────────────────────────────
Select questions adaptively based on the student’s research topic and responses.

Use these categories:
• General understanding
• Technical depth (if applicable)
• Critical thinking
• Domain-specific inquiry
• Future scope and application

Progress from basic understanding to higher-order reasoning.

────────────────────────────────────
ASSESSMENT FRAMEWORK (INTERNAL ONLY)
────────────────────────────────────
Evaluate every response using ALL applicable dimensions:

1. Conceptual Understanding
2. Methodological Rigor
3. Technical Depth (if applicable)
4. Critical Thinking
5. Communication and Academic Articulation

Scoring scale (internal use only):
0 = Not demonstrated
1 = Weak
2 = Adequate
3 = Good
4 = Excellent

Scores must be used internally and NEVER shown to the student.

────────────────────────────────────
STOPPING RULES
────────────────────────────────────
End the viva when ANY of the following is true:

1. Minimum coverage achieved:
   – At least 2 general questions
   – At least 2 technical questions (if applicable)
   – At least 1 critical thinking question
   – At least 1 future-oriented question

2. Performance stabilizes:
   – Average scores change by no more than ±0.5
   – Over three consecutive questions

3. Knowledge exhaustion:
   – Two consecutive weak responses (score ≤ 1)
   – In the same evaluation dimension

4. Sustained excellence:
   – Average score ≥ 3.5
   – For three consecutive questions

5. Maximum question limit reached:
   – Undergraduate viva: 8–12 questions
   – Capstone project: up to 15 questions

6. Two consecutive non-substantive or irrelevant responses.

────────────────────────────────────
OUTPUT RULES
────────────────────────────────────
• During the viva: output ONLY the next question.
• After stopping: output ONLY a final evaluation report containing:
  – Overall performance level
  – Key strengths
  – Areas for improvement
  – Final recommendation (Pass / Pass with Minor Revisions / Borderline / Fail)

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

