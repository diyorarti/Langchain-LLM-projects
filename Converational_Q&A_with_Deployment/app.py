import streamlit as st
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()


st.set_page_config("Chat")
st.header("Hi, Let's chat abou what you want")

chat = ChatOpenAI(temperature=0.7)

if "flowmessages" not in st.session_state:
    st.session_state["flowmessages"] = [
        SystemMessage(content="You are a comedian AI assistant")
    ]

def get_chatmodel_response(question):
    st.session_state['flowmessages'].append(HumanMessage(content=question))
    answer = chat(st.session_state['flowmessages'])
    st.session_state['flowmessages'].append(AIMessage(content=answer.content))
    return answer.content

input=st.text_input("Input: ",key="input")
response=get_chatmodel_response(input)

submit=st.button("Ask the question")

## If ask button is clicked

if submit:
    st.subheader("The Response is")
    st.write(response)