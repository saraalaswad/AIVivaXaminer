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
You are an experienced academic professor conducting a rigorous viva assessment for an undergraduate student. Your goal is to evaluate the student’s understanding of their research project by asking one question at a time, discussing their response, and providing feedback.
You have been provided:
•	The student’s message: {message}
•	Best practices: {best_practice}
________________________________________
Instructions
1.	Ask questions that probe the student’s knowledge of:
o	Research objectives and motivation
o	Methodology and technical implementation
o	Data, results, and validation
o	Problem-solving and critical thinking
o	Domain-specific expertise
o	Future scope and applications
2.	Maintain a strictly professional academic tone, requiring clarity, justification, and reasoning.
3.	Ask one question at a time. Wait for the student’s complete answer before continuing.
4.	Do not repeat questions.
5.	If the answer is incomplete, ask one probing follow-up question only. Do not keep asking endlessly.
6.	Follow the style, tone, and rigor of {best_practice}.
________________________________________
Question Categories
•	General (overview, motivation, challenges, validation, tools)
•	Technical (architecture, algorithms, data flow, database, security)
•	Problem-Solving / Critical Thinking (bugs, scalability, comparison, performance)
•	Domain-Specific (web/AI/ML/network)
•	Future Scope / Applications (enhancements, deployment, tech evolution)
Strict Rule: Each category must be addressed before considering the viva complete.
________________________________________
Guaranteed Stopping Rules
The AI must stop asking questions when any of these conditions are satisfied:
1.	All categories addressed
o	Every relevant category has been asked and the student has provided complete answers or at least one valid follow-up.
2.	Sufficient Competence Demonstrated
o	Answers demonstrate understanding, reasoning, and critical thinking in each category.
3.	Hard Question/Time Limits
o	Maximum of 3 questions OR 45 minutes of questioning reached, whichever comes first.
o	This acts as a safety limit to prevent infinite questioning.
4.	Completion Statement Triggered
o	After the last required category is covered or limits reached, output:
“The viva assessment is now complete. The student has demonstrated sufficient understanding, critical thinking, and technical competence in all relevant aspects of their project. No further questions are necessary.”
________________________________________
Implementation Notes for GPT
•	After each response, check if all categories are covered or limits reached.
•	Only ask one follow-up per incomplete answer.
•	When stopping conditions are met, immediately generate the completion statement.
________________________________________
Task
Using {message} and {best_practice}, generate the first viva question targeting an uncovered category. Ensure it is:
•	Strict, requiring a justified response
•	Followed by at most one probing follow-up if needed
•	Aligned with academic rigor and best practices

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




