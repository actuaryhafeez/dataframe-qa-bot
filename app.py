import streamlit as st
import pandas as pd
import os
from langchain.agents import create_pandas_dataframe_agent
from langchain.llms import OpenAI

# Set your OpenAI API key
api_key = os.environ.get("OPENAI_API_KEY")
if api_key is None:
    raise Exception("Please set your OPENAI_API_KEY environment variable.")

# Create a Streamlit app with the title "Ask Your DataFrame"
st.title("Ask Your DataFrame")

# Upload a CSV file
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    # Read the uploaded CSV file
    df = pd.read_csv(uploaded_file)
    
    # Initialize the agent
    agent = create_pandas_dataframe_agent(OpenAI(temperature=0), df, verbose=True)
    
    # Collect the user's question
    user_question = st.text_input("Ask a question:")
    
    if user_question:
        # Run the agent on the user's question
        response = agent.run(user_question)
        
        # Display the response
        st.write("Answer:", response)
