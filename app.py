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

# 1. Vectorise the student teacher response csv data
loader = CSVLoader(file_path="ts_response.csv")
documents = loader.load()

#print(len(documents))

embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(documents, embeddings)

# 2. Function for similarity search
def retrieve_info(query):
    similar_response = db.similarity_search(query, k=3)

    page_contents_array = [doc.page_content for doc in similar_response]

    #print(page_contents_array)

    return page_contents_array

# 3. Setup LLMChain & prompts
llm = ChatOpenAI(temperature=0.7, model="gpt-4-turbo")

template = """
You are AIVivaXaminer, an AI-based academic examiner conducting a structured undergraduate viva assessment.

────────────────────────────────────
RULES
────────────────────────────────────
1. Ask ONE question at a time.
2. Wait for the student’s full response before proceeding.
3. NEVER repeat a question. Before generating a question, check the list below.
4. Provide examiner-style qualitative feedback internally only.
5. Scores are INTERNAL only; NEVER reveal them.
6. Maintain a professional, academic examiner tone.

────────────────────────────────────
QUESTION HISTORY
────────────────────────────────────
Previously asked questions (update this list after each turn):
{question_history}

────────────────────────────────────
QUESTION STRATEGY
────────────────────────────────────
- Categories: General Understanding, Technical Depth (if applicable), Critical Thinking, Domain-Specific Inquiry, Future Scope & Application
- Start from foundational knowledge and progress to higher-order reasoning.
- Select questions NOT in {question_history}.
- Avoid asking questions that the student has already answered adequately in prior responses.

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

Use these scores to guide your questioning, but do not reveal them.

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
    input_variables=["message", "best_practice"],
    template=template
)

chain = LLMChain(llm=llm, prompt=prompt)

# 4. Retrieval augmented generation
def generate_response(message):
    best_practice = retrieve_info(message)
    response = chain.run(message=message, best_practice=best_practice)
    return response

# 5. Build an app with streamlit
def main():
    st.set_page_config(
        page_title="AIVivaXaminer", page_icon=":computer:")

    st.title(":computer: AIVivaXaminer")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("What is your research title?"):
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            assistant_response = generate_response(prompt)
            # Simulate stream of response with milliseconds delay
            for chunk in assistant_response.split():
                full_response += chunk + " "
                time.sleep(0.05)
                # Add a blinking cursor to simulate typing
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response})
    


if __name__ == '__main__':

    main()
