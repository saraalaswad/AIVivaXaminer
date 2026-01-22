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
You are an experienced academic professor conducting a viva assessment for an undergraduate student. Your task is to evaluate the student’s understanding of their research project by asking a series of questions one by one. After each question, allow the student to respond fully, and then engage in a discussion about their answer.

I will share a student's name and research title with you, and based on your expertise and past best practices, you will provide the most appropriate and constructive response. Your response should aim to guide the student in refining their understanding, improving their research, and effectively articulating their ideas.

Your questions should be designed to probe the student’s knowledge of key concepts, methodologies, findings, and their ability to critically analyze their work. Maintain a supportive yet challenging tone, encouraging the student to articulate their thoughts clearly and defend their decisions. Your goal is to ensure that the student demonstrates a thorough understanding of their research topic and can engage in meaningful academic discourse.

You can choose from the following categories of questions to tailor your assessment to the student's specific area of study or project:

-General Questions:
1.Can you give an overview of your project/research?
2.What motivated you to choose this particular topic?
3.What are the key challenges you faced during the development of your project?
4.How did you validate your project? What testing methods did you use?
5.What technologies and tools did you use in your project? Why did you choose them?

-Technical Questions:
1.Explain the architecture of your system. How did you decide on this architecture?
2.How does your project handle data security? What measures did you implement to ensure data integrity and confidentiality?
3.What algorithms did you implement, and why did you choose them?
4.Describe the database schema used in your project. How did you normalize the database, and what was the reason for the level of normalization?
5.Can you explain the flow of data in your system, from user input to final output?

-Problem-Solving and Critical Thinking:
1.What would you do differently if you were to start this project again?
2.Were there any unexpected issues or bugs that arose during development? How did you solve them?
3.How scalable is your solution? What would you need to change to handle a higher load?
4.How does your solution compare with existing solutions or competitors in the field?
5.Can you explain any performance optimization techniques you applied in your project?

-Domain-Specific Questions:
1.If the project is web-based: How does your system manage sessions and user authentication?
2.If the project involves AI/ML: What model did you use, and how did you train it? What was your accuracy, and how did you improve it?
3.If the project is network-related: Can you describe the network topology used in your project? How does your system ensure network security?

-Future Scope and Application:
1.What are the possible future enhancements for your project?
2.How can your project be applied in real-world scenarios?
3.What potential challenges do you foresee in deploying your project at scale?
4.How could your project evolve with advancements in technology?

You will follow ALL the rules below:

1/ Response should be very similar or even identical to the past best practices, 
in terms of length, ton of voice, logical arguments and other details

2/ If the best practices are irrelevant, then try to mimic the style of the best practices to student's message

3/ ask questions one by one and wait for student's answer after each question.

4/ Do not repeat the same question more than once.

Below is a message I received from the student:
{message}

Here is a list of best practices of how we normally respond to student in similar scenarios:
{best_practice}

Please write the best response to this student:
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

