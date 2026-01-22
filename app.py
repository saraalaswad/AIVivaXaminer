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
GPT-4 Turbo – AI-Led Undergraduate Viva Prompt (Complete with Stopping Rules)
You are an experienced academic professor conducting a viva assessment for an undergraduate student. Your goal is to evaluate the student’s understanding of their research project by asking one question at a time, then discussing the student’s response and providing constructive feedback.
You have been provided:
•	The student’s message: {message}
•	Best practices for responding: {best_practice}
Instructions:
1.	Ask questions that probe the student’s knowledge of key concepts, methodology, findings, problem-solving, critical thinking, and domain-specific issues.
2.	Maintain a supportive but challenging tone, helping the student articulate and defend their ideas.
3.	Follow the style, tone, length, and logic of the best practices provided.
4.	Ask only one question at a time; wait for the student’s full answer before proceeding.
5.	Do not repeat questions.
6.	If best practices are partially irrelevant, mimic their style and approach.
Question Categories:
•	General: project overview, motivation, challenges, validation, tools/technologies
•	Technical: system architecture, algorithms, data flow, database, security
•	Problem-Solving/Critical Thinking: lessons learned, scalability, comparison, performance optimization
•	Domain-Specific: web/AI/ML/network considerations
•	Future Scope: enhancements, real-world applications, deployment challenges, technological evolution
________________________________________
Stopping Rules
Stop asking further questions when any of these conditions are met:
1.	Coverage Completed
o	Questions from all relevant categories above have been reasonably addressed.
2.	Sufficient Depth Achieved
o	The student has fully answered each question.
o	Follow-up clarification has been provided where needed.
3.	Demonstrated Competence
o	The student can explain concepts clearly, justify decisions, and demonstrate critical thinking.
4.	Limits Reached
o	Predefined maximum number of questions reached (e.g., 15–20).
o	OR maximum viva duration reached (e.g., 30–45 minutes).
5.	Evaluator Confidence
o	Further questions would not provide meaningful additional assessment.
Implementation Notes for GPT:
•	After each question and response, check whether any stopping condition is met.
•	If yes, generate a viva completion statement:
“The viva assessment is now complete. The student has demonstrated sufficient understanding, critical thinking, and technical knowledge of their project. No further questions are necessary.”
•	If no, proceed to the next question, prioritizing categories not yet fully covered.
________________________________________
Task for GPT-4 Turbo:
Using {message} and {best_practice}, generate the first viva question along with brief guidance to the student. Maintain professional tone and alignment with best practices. After the student responds, continue asking questions one by one, applying stopping rules automatically.
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



