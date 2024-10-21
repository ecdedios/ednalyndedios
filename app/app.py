# __import__(pysqlite3)
# import sys
# sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# import os
# print(os.path.dirname(sys.executable))


import streamlit as st
from streamlit_chat import message
import pprint
from langchain.schema import AIMessage

from rag import get_graph

st.set_page_config(page_title="All about Ednalyn C. De Dios")

def process_input(inputs):

    last_output = None  # To store the last output from the stream

    for output in get_graph().stream(inputs):
        # Safely check if 'retrieve' exists in the output
        if 'retrieve' in output and 'messages' in output['retrieve']:
            # Display the first message from the 'retrieve' messages
            st.markdown(output['retrieve']['messages'][0])
    
        # Store the last output for further processing after the loop
        last_output = output

    # Ensure last_output is available and contains 'messages'
    if last_output and 'agent' in last_output and 'messages' in last_output['agent']:
        # Extract the first AIMessage object
        ai_message = last_output['agent']['messages'][0]

        # Check if ai_message is an AIMessage object (assuming AIMessage class exists)
        if isinstance(ai_message, AIMessage):
            ai_message_content = ai_message.content

            # Display the AIMessage content
            st.markdown(">>>" + str(ai_message_content))
        else:
            st.markdown(">>>" + "The message is not an AIMessage instance.")
    else:
        st.markdown()
        st.markdown(">>>" + "No valid AIMessage found.")
        

def page():

    st.markdown(
        """
        <style>
        button {
            background: none!important;
            border: none;
            padding: 0!important;
            color: white !important;
            text-decoration: none;
            cursor: pointer;
            border: none !important;
        }
        button:hover {
            text-decoration: none;
            color: white !important;
        }
        button:focus {
            outline: none !important;
            box-shadow: none !important;
            color: white !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    if st.button("Where did Ednalyn get her masters degree?"):
        inputs = {
            "messages": [
                ("user", "Where did Ednalyn get her masters degree?"),
                    ]
                }        
        process_input(inputs)

    if st.button("What are Ednalyn's technical skills?"):
        inputs = {
            "messages": [
                ("user", "What are Ednalyn's technical skills?"),
                    ]
                }        
        process_input(inputs)

    if st.button("Tell me about Ednalyn's ten most recent development projects?"):
        inputs = {
            "messages": [
                ("user", "Tell me about Ednalyn's ten most recent development projects."),
                    ]
                }        
        process_input(inputs)

    # Create a text input box in Streamlit
    question = st.text_input("OR, ask me a question:")

    inputs = {
        "messages": [
            ("user", question),
                ]
            }
    
    if question:
        process_input(inputs)

if __name__ == "__main__":
    page()