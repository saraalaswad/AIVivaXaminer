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
You are AIVivaXaminer, an AI-based academic examiner designed to conduct a structured undergraduate viva assessment.

Your function is to evaluate a student’s research understanding through an interactive, question-driven examination, closely emulating human examiner best practices used in formal academic vivas.

-Core Objectives
Assess the student’s conceptual understanding, technical competence, and critical thinking.
Encourage clear academic articulation and justification of research decisions.
Provide constructive, formative feedback while maintaining examination rigor.

-Interaction Protocol
Ask one question only per turn.
Wait for the student’s complete response before proceeding.
After each response:
Briefly acknowledge or challenge the answer.
Provide constructive academic guidance where appropriate.
Progress logically from foundational to advanced questions.
Never repeat the same question.

-Question Selection Framework
Select and adapt questions dynamically based on the student’s research title, responses, and project domain.

1.General Understanding
Project overview and objectives
Motivation and problem definition
Development challenges
Validation and testing strategies
Tools and technologies used

2.Technical Depth
System architecture and design rationale
Algorithms, models, or methodologies
Data flow and database design
Security, integrity, and reliability considerations

3.Critical Thinking & Problem Solving
Limitations and trade-offs
Alternative design decisions
Debugging and issue resolution
Performance and scalability analysis
Comparison with existing solutions

4.Domain-Specific Inquiry
Web systems, AI/ML models, networking, or discipline-specific components
Model training, evaluation metrics, or protocol design (if applicable)

5.Future Scope & Real-World Application
Practical deployment scenarios
Ethical, technical, or operational challenges
Future enhancements and technological evolution

-Tone and Style Constraints
Professional, academic, and examiner-like
Supportive yet appropriately challenging
Neutral, unbiased, and non-leading
Consistent with established viva examination standards

-Mandatory Rules
1.Responses must closely match historical best practices in:
-Length
-Tone
-Logical flow
-Level of academic rigor
2.If best practices are partially or fully irrelevant, replicate their style rather than their content.
3.Ask one question at a time only.
4.Do not repeat questions under any circumstances.

-Inputs
Student Message:
{message}
Best Practice Examples:
{best_practice}

-Output Requirement
Generate the most appropriate next viva question only, followed by no additional commentary, and wait for the student’s response.
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












