from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os 
import streamlit as st
load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

model = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    openai_api_key=openai_api_key
)

st.header('Research Tool')
user_input = st.text_input('Enter your Prompt:')

if st.button('Submit'):
    result = model.invoke(user_input)
    st.write(result.content)

