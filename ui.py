import streamlit as st
import random
import time
import language 
import mapping

# Streamed response emulator
def response_generator(query, location,context=None,contextual=False):
    response = language.generator(query, location,context,contextual)
    time.sleep(0.05)
    return response

st.title("GlaucoBot")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What is up?"):

    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Retrieve relevant document for this query
    doc = mapping.doc_retrieve(prompt)

    # Memory mechanism kicks in after 2 user-assistant pairs (4 messages)
    if len(st.session_state.messages) >= 2:
        # Summarize the previous conversation (excluding current user message)
        summary = language.summarize_conversation(st.session_state.messages[:])
        # Combine the summary and the current prompt for contextual response
        response = response_generator(prompt, str(doc),summary,True)
    else:
        # Basic response generation
        response = response_generator(prompt, str(doc))

    # Display assistant response
    with st.chat_message("assistant"):
        st.markdown(response)

    # Save assistant response in session history
    st.session_state.messages.append({"role": "assistant", "content": response})
