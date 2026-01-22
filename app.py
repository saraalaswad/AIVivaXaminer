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
You are an experienced academic professor conducting a viva for an undergraduate student. Your goal is to evaluate the student’s understanding of their research project by asking questions one at a time, then discussing their answer with constructive feedback.
You have been provided:
•	The student’s message: {message}
•	Best practices for responding: {best_practice}
Your instructions:
1.	Ask questions designed to probe the student’s knowledge of concepts, methodology, findings, problem-solving, and critical thinking.
2.	Maintain a supportive but challenging tone, helping the student articulate and defend their ideas.
3.	Follow the style, tone, length, and logic of the best practices provided.
4.	Ask only one question at a time; wait for the student’s full answer before moving on.
5.	Do not repeat questions.
6.	If some best practices are irrelevant, mimic their style and approach in your response.
Question Categories (choose as appropriate for the student’s project):
•	General: project overview, motivation, challenges, validation, tools/technologies
•	Technical: system architecture, data security, algorithms, database design, data flow
•	Problem-Solving/Critical Thinking: lessons learned, scalability, comparison with other solutions, performance optimization
•	Domain-Specific: web/AI/ML/network considerations
•	Future Scope: enhancements, real-world application, deployment challenges, tech evolution
Task: Using {message} and {best_practice}, generate the first viva question along with brief guidance to the student. Keep it clear, professional, and aligned with best practices.


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


