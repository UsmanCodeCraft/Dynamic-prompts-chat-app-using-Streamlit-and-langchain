from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
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

paper_input = st.selectbox("Select Research paper Name",["Attention Is All You Need","BERT: Pre-training of Deep Bidirectional Transformers","GPT-3: Language Models are Few-Shot Learners","Diffusion Models Beat GANs on Image Synthesis"])

style_input = st.selectbox("Select Writing Style",["Beginner-Friendly","Technical","Code-Oriented","mathematical"])  

length_input = st.selectbox("Select Length of Summary",["Short (1-2 paragraphs)","Medium (3-5 paragraphs)","Long (Detailed Explanation)"])

# Template
template = PromptTemplate(
    template="""
Please summarize the research paper titled "{paper_input}"with the following specifications:
Explanation Style: {style_input}
Explanation Length: {length_input}
1. Mathematical Details:
	- include relevant mathematical equations if present in the paper.
	- Explain the mathematical concept using simple, intuitive code snippets where applicable. 
2. Analogies:
	- Use relatebale analogies to simply complex ideas. 
if certain is not available in the paper, respond with: "insufficient information available " instead of guessing. 
insure the summary is clear, accurate and aligned with the provided style and length.
""",
input_variables=["paper_input","style_input","length_input"]
)

# Fill the placeholders
prompt = template.invoke({
    "paper_input": paper_input,
    "style_input": style_input,
    "length_input": length_input
})

if st.button('Summarize'):
    result = model.invoke(prompt)
    st.write(result.content)


