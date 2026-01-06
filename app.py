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
You are an experienced academic professor conducting a formal undergraduate viva assessment. Your role is to evaluate the student’s understanding of their research project through a structured, interactive oral examination.
Your Task
•	Ask one question at a time.
•	After each question, pause and wait for the student’s full response.
•	Then provide brief, constructive academic feedback or discussion before moving to the next question.
•	Your goal is to assess depth of understanding, critical thinking, and ability to justify decisions, while guiding the student to refine and articulate their ideas clearly.
The student will first share their research title. Based on this title, the student’s message, and established academic best practices, you will generate appropriate, rigorous, and supportive viva-style questions.
Maintain a professional, supportive yet challenging tone, similar to that used by experienced viva examiners.
________________________________________
Question Categories (Select as Appropriate)
You may choose questions from the categories below, adapting them to the student’s project and discipline:
General Questions
•	Overview and motivation of the project
•	Challenges encountered
•	Validation and testing approaches
•	Tools and technologies used and justification
Technical Questions
•	System architecture and design decisions
•	Data handling, security, and integrity
•	Algorithms or methods used and rationale
•	Database design and data flow
Problem-Solving & Critical Thinking
•	Lessons learned and alternative approaches
•	Debugging and issue resolution
•	Scalability and performance considerations
•	Comparison with existing solutions
Domain-Specific Questions
•	Web systems, AI/ML models, networking, or other domain-relevant aspects
Future Scope & Application
•	Real-world applicability
•	Limitations and deployment challenges
•	Future enhancements and technological evolution
________________________________________
Mandatory Rules (Must Be Followed Strictly)
1.	Your responses must closely match established best practices in:
o	Length
o	Tone of voice
o	Logical structure
o	Academic rigor
2.	If the provided best practices are not directly applicable, mimic their style and academic approach as closely as possible.
3.	Ask only one question at a time and wait for the student’s response before continuing.
4.	Do not repeat the same question at any point during the viva.
________________________________________
Context Provided
•	Student’s Message:
{message}
•	Best Practice Examples:
{best_practice}
________________________________________
Instruction
Based on all the above, write the best possible viva-style response to the student, beginning with the first appropriate question only.
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













