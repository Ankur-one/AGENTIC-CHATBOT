
import streamlit as st
import uuid
import re

from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    AIMessageChunk,
)

from agentic_chatbot_backend import chatbot


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Agentic Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown(
    """
    <style>
    /* Main application */
    .stApp {
        background-color: #212121;
        color: #ececec;
    }

    /* Main content area */
    .main .block-container {
        max-width: 900px;
        padding-top: 2rem;
        padding-bottom: 7rem;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #171717;
        border-right: 1px solid #303030;
    }

    [data-testid="stSidebar"] * {
        color: #ececec;
    }

    /* Buttons */
    .stButton > button {
        border-radius: 10px;
        border: 1px solid #414141;
        background-color: #262626;
        color: #ececec;
        transition: background-color 0.2s ease;
    }

    .stButton > button:hover {
        background-color: #383838;
        border-color: #666666;
    }

    /* Chat messages */
    [data-testid="stChatMessage"] {
        background-color: transparent;
        border-radius: 14px;
        padding: 0.7rem;
    }

    /* Chat input */
    [data-testid="stChatInput"] {
        background-color: #303030;
        border: 1px solid #505050;
        border-radius: 18px;
    }

    [data-testid="stChatInput"] textarea {
        color: #ffffff;
    }

    /* Welcome heading */
    .welcome-title {
        font-size: 2rem;
        font-weight: 650;
        text-align: center;
        margin-top: 16vh;
        margin-bottom: 0.5rem;
        color: #f5f5f5;
    }

    .welcome-subtitle {
        text-align: center;
        font-size: 1rem;
        color: #a3a3a3;
        margin-bottom: 2rem;
    }

    /* Small application label */
    .app-label {
        color: #a3a3a3;
        font-size: 0.85rem;
        margin-bottom: 1rem;
    }

    /* Hide Streamlit's default decoration */
    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    header[data-testid="stHeader"] {
        background: transparent;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------

def generate_thread_id():
    """Create a unique ID for each conversation."""
    return str(uuid.uuid4())


def suggest_title(message, max_words=7):
    """
    Suggest a conversation title from the first message.
    No extra LLM request is required.
    """
    text = re.sub(r"\s+", " ", message).strip()
    text = re.sub(r"[^\w\s'-]", "", text)

    if not text:
        return "New conversation"

    words = text.split()
    title = " ".join(words[:max_words])

    if len(words) > max_words:
        title += "..."

    return title[:60]


def add_thread(thread_id):
    """Register a conversation in the sidebar."""
    if thread_id not in st.session_state.chat_threads:
        st.session_state.chat_threads.append(thread_id)

    if thread_id not in st.session_state.chat_titles:
        st.session_state.chat_titles[thread_id] = "New conversation"


def reset_chat():
    """Start a fresh conversation."""
    new_thread_id = generate_thread_id()

    st.session_state.thread_id = new_thread_id
    st.session_state.message_history = []

    add_thread(new_thread_id)


def load_conversation(thread_id):
    """
    Load messages from the LangGraph checkpointer.
    """
    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    try:
        state = chatbot.get_state(config)
        messages = state.values.get("messages", [])

        history = []

        for message in messages:
            if isinstance(message, HumanMessage):
                history.append({
                    "role": "user",
                    "content": message.content,
                })

            elif isinstance(message, AIMessage):
                # Skip empty assistant messages.
                if message.content:
                    history.append({
                        "role": "assistant",
                        "content": message.content,
                    })

        return history

    except Exception as error:
        st.error(f"Could not load conversation: {error}")
        return []


def select_conversation(thread_id):
    """Switch to an existing conversation."""
    st.session_state.thread_id = thread_id
    st.session_state.message_history = load_conversation(thread_id)


def delete_conversation(thread_id):
    """
    Remove a conversation from the current UI session.
    This does not delete its checkpoint from LangGraph storage.
    """
    st.session_state.chat_threads = [
        item
        for item in st.session_state.chat_threads
        if item != thread_id
    ]

    st.session_state.chat_titles.pop(thread_id, None)

    if st.session_state.thread_id == thread_id:
        reset_chat()


def stream_ai_response(user_input):
    """Stream only the assistant's generated text."""

    config = {
        "configurable": {
            "thread_id": st.session_state.thread_id
        }
    }

    for message_chunk, metadata in chatbot.stream(
        {
            "messages": [
                HumanMessage(content=user_input)
            ]
        },
        config=config,
        stream_mode="messages",
    ):
        # Ignore user messages and tool messages.
        if not isinstance(message_chunk, AIMessageChunk):
            continue

        # Avoid displaying internal reasoning or tool calls.
        if message_chunk.tool_calls:
            continue

        # Stream text content only.
        content = message_chunk.content

        if isinstance(content, str) and content:
            yield content


# --------------------------------------------------
# INITIALIZE SESSION STATE
# --------------------------------------------------

if "chat_threads" not in st.session_state:
    st.session_state.chat_threads = []

if "chat_titles" not in st.session_state:
    st.session_state.chat_titles = {}

if "message_history" not in st.session_state:
    st.session_state.message_history = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = generate_thread_id()

if "rename_thread_id" not in st.session_state:
    st.session_state.rename_thread_id = None

# Register the initial conversation.
add_thread(st.session_state.thread_id)


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.markdown("## 🤖 Agentic Chatbot")
    st.caption("Powered by LangGraph")

    st.divider()

    # New chat button
    if st.button(
        "＋  New Chat",
        key="new_chat",
        use_container_width=True,
    ):
        reset_chat()
        st.rerun()

    st.markdown("### Your conversations")

    # Conversation list, newest first.
    for thread_id in reversed(st.session_state.chat_threads):

        title = st.session_state.chat_titles.get(
            thread_id,
            "New conversation",
        )

        col1, col2 = st.columns([5, 1])

        with col1:
            if st.button(
                f"💬 {title}",
                key=f"open_{thread_id}",
                use_container_width=True,
                help=title,
                type=(
                    "primary"
                    if thread_id == st.session_state.thread_id
                    else "secondary"
                ),
            ):
                select_conversation(thread_id)
                st.rerun()

        with col2:
            if st.button(
                "⋯",
                key=f"options_{thread_id}",
                help="Rename or remove this conversation",
                use_container_width=True,
            ):
                st.session_state.rename_thread_id = thread_id

        # Show rename/delete controls for the selected menu.
        if st.session_state.rename_thread_id == thread_id:

            with st.container(border=True):
                st.caption("Conversation options")

                new_title = st.text_input(
                    "Conversation title",
                    value=title,
                    key=f"rename_input_{thread_id}",
                    label_visibility="collapsed",
                    placeholder="Enter conversation title",
                )

                if st.button(
                    "Save title",
                    key=f"save_title_{thread_id}",
                    use_container_width=True,
                ):
                    cleaned_title = new_title.strip()

                    if cleaned_title:
                        st.session_state.chat_titles[thread_id] = (
                            cleaned_title[:60]
                        )

                    st.session_state.rename_thread_id = None
                    st.rerun()

                if st.button(
                    "Delete from sidebar",
                    key=f"delete_{thread_id}",
                    use_container_width=True,
                ):
                    delete_conversation(thread_id)
                    st.session_state.rename_thread_id = None
                    st.rerun()

                if st.button(
                    "Cancel",
                    key=f"cancel_{thread_id}",
                    use_container_width=True,
                ):
                    st.session_state.rename_thread_id = None
                    st.rerun()

    st.divider()

    st.caption("Developed by Ankur Kumar")
    st.caption("AI responses may contain mistakes.")


# --------------------------------------------------
# MAIN CHAT HEADER
# --------------------------------------------------

st.markdown(
    '<div class="app-label">LANGGRAPH · CONVERSATIONAL AI</div>',
    unsafe_allow_html=True,
)

current_title = st.session_state.chat_titles.get(
    st.session_state.thread_id,
    "New conversation",
)

st.title(current_title)


# --------------------------------------------------
# DISPLAY CHAT HISTORY
# --------------------------------------------------

if not st.session_state.message_history:

    st.markdown(
        """
        <div class="welcome-title">
            How can I help you today?
        </div>
        <div class="welcome-subtitle">
            Ask a question to start a conversation with your AI assistant.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Suggested prompts
    st.markdown("")

    col1, col2 = st.columns(2)

    with col1:
        if st.button(
            "💡 Explain a technical concept",
            use_container_width=True,
        ):
            st.session_state.suggested_prompt = (
                "Explain a technical concept with examples."
            )
            st.rerun()

    with col2:
        if st.button(
            "💻 Help me with Python code",
            use_container_width=True,
        ):
            st.session_state.suggested_prompt = (
                "Help me understand and write Python code."
            )
            st.rerun()

else:

    for message in st.session_state.message_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


# --------------------------------------------------
# CHAT INPUT
# --------------------------------------------------

suggested_prompt = st.session_state.pop(
    "suggested_prompt",
    None,
)

user_input = st.chat_input(
    "Message your AI assistant..."
)

if suggested_prompt:
    user_input = suggested_prompt


# --------------------------------------------------
# HANDLE USER MESSAGE AND STREAM RESPONSE
# --------------------------------------------------

if user_input:

    # Generate a suggested title from the first user message.
    is_first_message = (
        len(st.session_state.message_history) == 0
    )

    if is_first_message:
        st.session_state.chat_titles[
            st.session_state.thread_id
        ] = suggest_title(user_input)

    # Save and display the user message.
    st.session_state.message_history.append({
        "role": "user",
        "content": user_input,
    })

    with st.chat_message("user"):
        st.markdown(user_input)

    # Stream the assistant response.
    try:
        with st.chat_message("assistant"):

            response = st.write_stream(
                stream_ai_response(user_input)
            )

        # Save the complete assistant response.
        st.session_state.message_history.append({
            "role": "assistant",
            "content": response,
        })

    except Exception as error:
        error_message = (
            f"Sorry, an error occurred while generating the response: "
            f"{error}"
        )

        with st.chat_message("assistant"):
            st.error(error_message)

    # Refresh the title and sidebar after the first message.
    st.rerun()
