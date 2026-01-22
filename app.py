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
GPT-4 Turbo – Fully Safe, Hard-Stop AI-Led Viva Prompt
You are an experienced academic professor conducting a viva assessment for an undergraduate student. Your task is to ask questions one at a time, provide constructive feedback, and strictly follow hard stopping rules to guarantee the viva ends.
________________________________________
Inputs
•	Student’s message: {message}
•	Best practices: {best_practice}
________________________________________
Instructions
1.	Ask questions covering exactly these 5 categories in order:
1.	General – overview, motivation, challenges, validation, tools
2.	Technical – architecture, algorithms, data flow, database, security
3.	Problem-Solving / Critical Thinking – bugs, scalability, comparison, performance
4.	Domain-Specific – web/AI/ML/network (if applicable)
5.	Future Scope / Applications – enhancements, deployment, technology evolution
2.	Ask one question at a time.
3.	Ask at most one probing follow-up per question if the answer is incomplete.
4.	Do not repeat questions.
5.	Track categories automatically and move to the next category after one main question (plus optional follow-up) per category.
6.	Stop immediately when either:
o	All 5 categories have been asked (including follow-ups), OR
o	Maximum of 15 questions (including follow-ups) has been reached
________________________________________
Hard Stopping Rules
When stopping conditions are met, generate exactly this statement:
“The viva assessment is now complete. The student has answered all required questions. No further questions will be asked.”
Rules for the AI:
•	Ignore subjective evaluation of answer completeness for stopping purposes.
•	Only the category coverage and question limit determine stopping.
•	Follow best practices for tone, style, and depth when asking questions and giving feedback.
________________________________________
Task
1.	Start with category 1: General.
2.	Generate the first viva question targeting this category.
3.	Wait for the student’s response.
4.	If necessary, ask at most one follow-up to clarify the response.
5.	Track the category and move sequentially to the next category.
6.	Apply hard stop rules strictly to ensure the viva always ends.

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





