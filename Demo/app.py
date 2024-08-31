import streamlit as st
from dotenv import load_dotenv
from langchain_openai import OpenAI  # Updated import
from langchain_core.prompts import PromptTemplate  # Updated import
from langchain.chains import LLMChain, SequentialChain
from langchain.memory import ConversationBufferMemory
import os

# Load environment variables
load_dotenv()
openai_api_key = os.environ.get("OPENAI_API_KEY")

# Check if the API key is set
if not openai_api_key:
    st.error("OpenAI API key not found. Please set the API key in your environment variables.")
    st.stop()

# OpenAI LLMs
llm = OpenAI(temperature=0.8, openai_api_key=openai_api_key)  # Pass API key directly

# Streamlit framework
st.title("Celebrity Search Results")
input_text = st.text_input("Search the topic you want...")

# Memory
person_memory = ConversationBufferMemory(input_key='name', memory_key="chat_history")
dob_memory = ConversationBufferMemory(input_key='person', memory_key='chat_history')
descr_memory = ConversationBufferMemory(input_key='dob', memory_key='description_history')

# Prompt Templates
first_input_prompt = PromptTemplate(
    input_variables=['name'],
    template="Tell me about celebrity {name}."
)
chain = LLMChain(
    llm=llm,
    prompt=first_input_prompt,
    verbose=True,
    output_key="person",
    memory=person_memory
)

second_input_prompt = PromptTemplate(
    input_variables=['person'],
    template="When was {person} born?"
)
chain2 = LLMChain(
    llm=llm,
    prompt=second_input_prompt,
    verbose=True,
    output_key='dob',
    memory=dob_memory
)

third_input_prompt = PromptTemplate(
    input_variables=['dob'],
    template="Mention 5 major events that happened around {dob} in the world."
)
chain3 = LLMChain(
    llm=llm,
    prompt=third_input_prompt,
    verbose=True,
    output_key='description',
    memory=descr_memory
)

parent_chain = SequentialChain(
    chains=[chain, chain2, chain3],
    input_variables=['name'],
    output_variables=['person', 'dob', 'description'],
    verbose=True
)

if input_text:
    result = parent_chain({'name': input_text})
    st.write(result)

    with st.expander('Person Name'):
        st.info(person_memory.buffer)

    with st.expander('Date of Birth'):
        st.info(dob_memory.buffer)

    with st.expander('Major Events'):
        st.info(descr_memory.buffer)
