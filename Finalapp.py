import re
import uuid
from pathlib import Path

import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage, AIMessageChunk

from Final_Agent_backend import (
    chatbot,
    add_pdf_to_knowledge_base,
    get_all_threads,
    delete_thread_from_db,
)


# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="Agentic Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ==========================================================
# CUSTOM CSS
# ==========================================================

st.markdown(
    """
    <style>
    .stApp {
        background-color: #212121;
        color: #ececec;
    }

    .main .block-container {
        max-width: 900px;
        padding-top: 2rem;
        padding-bottom: 7rem;
    }

    [data-testid="stSidebar"] {
        background-color: #171717;
        border-right: 1px solid #303030;
    }

    [data-testid="stSidebar"] * {
        color: #ececec;
    }

    .stButton > button {
        border-radius: 10px;
        border: 1px solid #414141;
        background-color: #262626;
        color: #ececec;
    }

    .stButton > button:hover {
        background-color: #383838;
        border-color: #666666;
    }

    [data-testid="stChatMessage"] {
        background-color: transparent;
        border-radius: 14px;
        padding: 0.7rem;
    }

    [data-testid="stChatInput"] {
        background-color: #303030;
        border: 1px solid #505050;
        border-radius: 18px;
    }

    [data-testid="stChatInput"] textarea {
        color: #ffffff;
    }

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

    .app-label {
        color: #a3a3a3;
        font-size: 0.85rem;
        margin-bottom: 1rem;
    }

    #MainMenu, footer {
        visibility: hidden;
    }

    header[data-testid="stHeader"] {
        background: transparent;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ==========================================================
# HELPER FUNCTIONS
# ==========================================================

def generate_thread_id():
    return str(uuid.uuid4())


def extract_text_content(content):
    """Safely extract plain text from string or structured content blocks."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text" and block.get("text"):
                parts.append(block["text"])
        return "".join(parts)
    return str(content) if content is not None else ""


def suggest_title(message, max_words=7):
    text = re.sub(r"\s+", " ", message).strip()
    text = re.sub(r"[^\w\s'-]", "", text)

    if not text:
        return "New conversation"

    words = text.split()
    title = " ".join(words[:max_words])

    if len(words) > max_words:
        title += "..."

    return title[:60]


def get_thread_title(thread_id):
    """Retrieve title from session state or derive from the first user message."""
    if thread_id in st.session_state.get("chat_titles", {}):
        return st.session_state.chat_titles[thread_id]

    try:
        config = {"configurable": {"thread_id": thread_id}}
        state = chatbot.get_state(config)
        messages = state.values.get("messages", [])
        for message in messages:
            if isinstance(message, HumanMessage):
                text = extract_text_content(message.content)
                if text.strip():
                    title = suggest_title(text.strip())
                    st.session_state.setdefault("chat_titles", {})[thread_id] = title
                    return title
    except Exception:
        pass

    return "New conversation"


def add_thread(thread_id):
    if thread_id not in st.session_state.chat_threads:
        st.session_state.chat_threads.append(thread_id)

    if thread_id not in st.session_state.chat_titles:
        st.session_state.chat_titles[thread_id] = get_thread_title(thread_id)


def reset_chat():
    new_thread_id = generate_thread_id()
    st.session_state.thread_id = new_thread_id
    st.session_state.message_history = []
    add_thread(new_thread_id)


def load_conversation(thread_id):
    config = {"configurable": {"thread_id": thread_id}}

    try:
        state = chatbot.get_state(config)
        messages = state.values.get("messages", [])
        history = []

        for message in messages:
            if isinstance(message, HumanMessage):
                text = extract_text_content(message.content)
                if text.strip():
                    history.append({
                        "role": "user",
                        "content": text,
                    })

            elif isinstance(message, AIMessage):
                # Do not display internal tool-call messages.
                if not message.tool_calls:
                    text = extract_text_content(message.content)
                    if text.strip():
                        history.append({
                            "role": "assistant",
                            "content": text,
                        })

        return history

    except Exception as error:
        print(f"Conversation loading error: {error}")
        return []


def select_conversation(thread_id):
    st.session_state.thread_id = thread_id
    st.session_state.message_history = load_conversation(thread_id)


def delete_conversation(thread_id):
    st.session_state.chat_threads = [
        item for item in st.session_state.chat_threads
        if item != thread_id
    ]
    st.session_state.chat_titles.pop(thread_id, None)
    delete_thread_from_db(thread_id)

    if st.session_state.thread_id == thread_id:
        reset_chat()


def stream_ai_response(user_input):
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
        if not isinstance(message_chunk, AIMessageChunk):
            continue

        # Ignore tool calls; display only generated text.
        if message_chunk.tool_calls:
            continue

        content = message_chunk.content

        if isinstance(content, str) and content:
            yield content

        elif isinstance(content, list):
            for block in content:
                if isinstance(block, str) and block:
                    yield block
                elif (
                    isinstance(block, dict)
                    and block.get("type") == "text"
                    and block.get("text")
                ):
                    yield block["text"]


def process_uploaded_pdfs(uploaded_files):
    """Save and index each uploaded PDF."""

    upload_dir = Path(__file__).resolve().parent / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for uploaded_file in uploaded_files:
        safe_name = Path(uploaded_file.name).name

        if Path(safe_name).suffix.lower() != ".pdf":
            st.error(f"{safe_name}: only PDF files are supported.")
            results.append({"name": safe_name, "success": False})
            continue

        saved_path = upload_dir / (
            f"{uuid.uuid4().hex}_{safe_name}"
        )

        try:
            saved_path.write_bytes(uploaded_file.getvalue())

            with st.spinner(f"Indexing {safe_name}..."):
                page_count, chunk_count = add_pdf_to_knowledge_base(
                    saved_path
                )

            st.success(
                f"**{safe_name}** added to your knowledge base. "
                f"Pages: {page_count} · Chunks: {chunk_count}"
            )
            results.append({
                "name": safe_name,
                "success": True,
                "pages": page_count,
                "chunks": chunk_count,
            })

        except Exception as error:
            st.error(
                f"Could not index {safe_name}. "
                "Check the terminal for details."
            )
            print(f"PDF indexing error for {safe_name}: {error}")
            results.append({"name": safe_name, "success": False})

    return results


# ==========================================================
# SESSION STATE
# ==========================================================

if "chat_threads" not in st.session_state:
    saved_threads = get_all_threads()
    st.session_state.chat_threads = saved_threads if saved_threads else []

if "chat_titles" not in st.session_state:
    st.session_state.chat_titles = {}

if "thread_id" not in st.session_state:
    if st.session_state.chat_threads:
        st.session_state.thread_id = st.session_state.chat_threads[0]
    else:
        st.session_state.thread_id = generate_thread_id()

if "message_history" not in st.session_state:
    st.session_state.message_history = load_conversation(st.session_state.thread_id)

if "rename_thread_id" not in st.session_state:
    st.session_state.rename_thread_id = None

if "suggested_prompt" not in st.session_state:
    st.session_state.suggested_prompt = None

add_thread(st.session_state.thread_id)


# ==========================================================
# SIDEBAR
# ==========================================================

with st.sidebar:
    st.markdown("## 🤖 Agentic Chatbot")
    st.caption("Gemini · LangGraph · RAG")
    st.divider()

    if st.button(
        "＋ New Chat",
        key="new_chat",
        use_container_width=True,
    ):
        reset_chat()
        st.rerun()

    st.markdown("### Your conversations")

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
                help="Rename or delete conversation",
                use_container_width=True,
            ):
                if st.session_state.rename_thread_id == thread_id:
                    st.session_state.rename_thread_id = None
                else:
                    st.session_state.rename_thread_id = thread_id
                st.rerun()

        if st.session_state.rename_thread_id == thread_id:
            with st.container(border=True):
                new_title = st.text_input(
                    "Conversation title",
                    value=title,
                    key=f"rename_input_{thread_id}",
                    placeholder="Enter a title",
                )

                if st.button(
                    "Save title",
                    key=f"save_{thread_id}",
                    use_container_width=True,
                ):
                    if new_title.strip():
                        st.session_state.chat_titles[thread_id] = (
                            new_title.strip()[:60]
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
    st.markdown("### Features")
    st.markdown(
        """
        - 📎 Attach PDF documents
        - 📚 Search uploaded documents
        - 🔎 FAISS semantic retrieval
        - ✨ Gemini responses
        - 🌐 Optional Tavily search
        """
    )
    st.caption("Developed by Ankur Kumar")


# ==========================================================
# MAIN CHAT
# ==========================================================

st.markdown(
    '<div class="app-label">LANGGRAPH · CONVERSATIONAL AI</div>',
    unsafe_allow_html=True,
)

current_title = st.session_state.chat_titles.get(
    st.session_state.thread_id,
    "New conversation",
)

st.title(current_title)


# ==========================================================
# DISPLAY HISTORY
# ==========================================================

if not st.session_state.message_history:
    st.markdown(
        """
        <div class="welcome-title">How can I help you today?</div>
        <div class="welcome-subtitle">
            Ask a question or attach a PDF using the control below.
        </div>
        """,
        unsafe_allow_html=True,
    )

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


# ==========================================================
# CHAT INPUT WITH ATTACHMENT CONTROL
# ==========================================================

suggested_prompt = st.session_state.pop(
    "suggested_prompt",
    None,
)

submission = st.chat_input(
    "Message your AI assistant...",
    accept_file="multiple",
    file_type=["pdf"],
    max_upload_size=200,
    key="agentic_chat_input",
)

user_input = ""
uploaded_files = []

if submission is not None:
    if isinstance(submission, str):
        user_input = submission.strip()
    else:
        user_input = (getattr(submission, "text", "") or "").strip()
        uploaded_files = getattr(submission, "files", []) or []

elif suggested_prompt:
    user_input = suggested_prompt


# ==========================================================
# INDEX ATTACHED PDFS
# ==========================================================

upload_results = []
if uploaded_files:
    upload_results = process_uploaded_pdfs(uploaded_files)

# If only files were uploaded without accompanying question
if uploaded_files and not user_input:
    successful_uploads = [r["name"] for r in upload_results if r.get("success")]
    if successful_uploads:
        ack_message = (
            f"📎 **Uploaded and indexed:** {', '.join(successful_uploads)}\n\n"
            "You can now ask questions about the contents of this document!"
        )
        st.session_state.message_history.append({
            "role": "assistant",
            "content": ack_message,
        })
        st.rerun()


# ==========================================================
# HANDLE MESSAGE AND STREAM ANSWER
# ==========================================================

if user_input:
    is_first_message = (
        len(st.session_state.message_history) == 0
    )

    if is_first_message:
        st.session_state.chat_titles[
            st.session_state.thread_id
        ] = suggest_title(user_input)

    st.session_state.message_history.append({
        "role": "user",
        "content": user_input,
    })

    with st.chat_message("user"):
        st.markdown(user_input)

    try:
        with st.chat_message("assistant"):
            response = st.write_stream(
                stream_ai_response(user_input)
            )

        if not isinstance(response, str):
            response = str(response)

        if response.strip():
            st.session_state.message_history.append({
                "role": "assistant",
                "content": response,
            })
        else:
            fallback = "I couldn't generate a response. Please try again."
            st.session_state.message_history.append({
                "role": "assistant",
                "content": fallback,
            })
            st.markdown(fallback)

    except Exception as error:
        error_msg = f"An error occurred while generating the response: {error}"
        st.error(error_msg)
        print(f"Chatbot response error: {error}")

    # Rerun on first message so the sidebar title updates immediately
    if is_first_message:
        st.rerun()
