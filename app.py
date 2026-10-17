
from agentic_chatbot_backend import chatbot
from langchain_core.messages import HumanMessage
import streamlit as st

st.title("Agentic Chatbot with Langgraph")

thread_id = "1"
CONFIG = {'configurable': {'thread_id': thread_id}}

if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []


# Loading the conversion history
for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.text(message['content'])


user_input = st.chat_input('type here')


if user_input:
    # First add the message to message_history
    st.session_state['message_history'].append({'role': 'user', 'content' : user_input})
    with st.chat_message('user'):
        st.text(user_input)


    # First add the message to message_history
    with st.chat_message('assistant'):

        ai_message = st.write_stream(
            message_chunk.content for message_chunk, metadata in chatbot.stream(
                {'messages': [HumanMessage(content=user_input)]},
                config= CONFIG,
                stream_mode= "messages"
            )
        )

    st.session_state['message_history'].append({'role': 'assistant', 'content' : ai_message})
    with st.chat_message('assistant'):
        st.text(ai_message)