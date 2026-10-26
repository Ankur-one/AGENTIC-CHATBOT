from agentic_chatbot_backend import chatbot
from langchain_core.messages import BaseMessage, HumanMessage
import streamlit as st
import uuid

# Generateb a unique thread ID for each new conversation
def generate_thread_id():
    return str(uuid.uuid4())

# Add a new thread ID to the conversationb list
def add_thread(thread_id):

    if thread_id not in st.session_state["chat_thread"]:
        st.session_state["chat_thread"].append(thread_id)

st.title("Agentic Cahtbot By Langgraph")
st.subheader("By ANKUR KUMAR")

CONFIG = {'configurable': {'thread_id': 'thread_id'}}

if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

# Create a list for strong all conversation thread IDs
if "chat_thread" not in st.session_state:
    st.session_state["chat_thread"] = []


# Display the sidebar tilte
st.sidebar.title("My Conversations")


# Loading the conversatiion history
for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.text(message['content'])

user_input = st.chat_input('Type Here')

if user_input:
    st.session_state['message_history'].append({'role': 'user', 'content': user_input})
    with st.chat_message('assistant'):
        st.text(user_input)


    with st.chat_message('assistant'):
        ai_message = st.write_stream(
            message_chunk.content for message_chunk, metadata in chatbot.stream(
                {'messages': [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode='messages'
            )
        )

    st.session_state['message_history'].append({'role': 'assistant', 'content' : ai_message})