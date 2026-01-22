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

# -------------------------------
# 1. Vectorise the student-teacher response csv data
# -------------------------------
loader = CSVLoader(file_path="ts_response.csv")
documents = loader.load()
embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(documents, embeddings)

# -------------------------------
# 2. Function for similarity search
# -------------------------------
def retrieve_info(query):
    similar_response = db.similarity_search(query, k=3)
    return [doc.page_content for doc in similar_response]

# -------------------------------
# 3. Setup LLMChain & prompts
# -------------------------------
llm = ChatOpenAI(temperature=0.7, model="gpt-4-turbo")

template = """
You are an experienced academic professor conducting a viva for an undergraduate student. Your goal is to evaluate the student’s understanding of their research project by asking questions one at a time, then discussing their answer with constructive feedback.

Student message: {message}
Best practices: {best_practice}

Instructions:
- Ask one question at a time.
- Maintain supportive but challenging tone.
- Follow best practices.
- Do not repeat questions.

Task: Generate the next viva question and guidance for the student.
"""

prompt = PromptTemplate(
    input_variables=["message", "best_practice"],
    template=template
)

chain = LLMChain(llm=llm, prompt=prompt)

def generate_response(message):
    best_practice = retrieve_info(me
