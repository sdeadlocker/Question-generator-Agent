import streamlit as st
import source_code 
Question_agent = source_code.Generate_QA_Agent()

st.title("Question Generator")
st.subheader("This agent will generate MCQs/Subjective questions from the provided document based on user query")

text = st.empty()
query=text.text_input("Enter your query", value="",key="1")
if query:
   st.write("Please wait generating response")
   res = Question_agent.agent_chain.run(query) 
   st.write(res)
   
   