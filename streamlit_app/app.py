import streamlit as st
from rag import rag_chain

st.title("🤰 Pregnancy Care Bot")

def stream_response(prompt):
    response_container = st.empty()
    full_response = ""
    for chunk in rag_chain.stream(prompt):  # this yields chunks of text
        full_response += chunk
        response_container.markdown(full_response + "_")  # Show a _ effect while streaming
    response_container.markdown(full_response)  # Final message without cursor
    return full_response

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("আমি প্রেগনেন্সী কেয়ার বট। গর্ভাবস্থায় যেকোনো জিজ্ঞাসা আমাকে করতে পারেন।"):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        response = stream_response(prompt)

    st.session_state.messages.append({"role": "assistant", "content": response})





