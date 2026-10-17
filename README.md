# 🤖 Agentic AI Chatbot

An Agentic AI Chatbot built using **Python, LangGraph, LangChain, and Groq LLM**. This project explores stateful AI workflows, conversational context, and the foundations of building intelligent AI agents.

The project uses `uv` for fast Python dependency management and environment setup.

## 🚀 Features

- **LLM Integration:** Connect to Groq-hosted language models.
- **LangGraph Workflow:** Build stateful chatbot workflows using graph nodes and edges.
- **Conversation State:** Manage messages using LangGraph's message reducer.
- **Memory Checkpointing:** Maintain conversation state across calls using thread IDs.
- **Modular Backend:** Separate chatbot logic from the application entry point.
- **Fast Environment Setup:** Manage dependencies using `uv`.
- **Extensible Architecture:** Designed to support future tool calling, RAG, and advanced agent workflows.

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| LangChain | LLM integration and message abstractions |
| LangGraph | Stateful agent workflow orchestration |
| Groq API | Hosted LLM inference |
| Streamlit | Planned interactive web interface |
| uv | Python package and environment management |
| python-dotenv | Environment variable configuration |

## 📁 Project Structure

```text
Agentic-Chatbot/
├── agentic_chatbot_backend.py
├── app.py
├── .env.example
├── .gitignore
├── requirements.txt
├── pyproject.toml
├── uv.lock
└── README.md
```

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/Ankur-one/Agentic-Chatbot.git
cd Agentic-Chatbot
```

### 2. Install Python and uv

Use Python 3.12 and install `uv` if it is not already installed.

```bash
pip install uv
```

### 3. Install dependencies

If the project contains `pyproject.toml` and `uv.lock`:

```bash
uv sync
```

Alternatively, if you are installing from the requirements file:

```bash
uv venv --python 3.12
uv pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=openai/gpt-oss-20b
```

Obtain an API key from the [Groq Console](https://console.groq.com/).

Check the current model availability and account usage limits before running the application.

### 5. Run the chatbot

```bash
uv run python app.py
```

The chatbot response will be printed in the terminal.

## 🧠 How It Works

1. A user submits a message.
2. LangGraph stores and manages the message in the graph state.
3. The chatbot node sends the conversation to the configured Groq language model.
4. The model generates a response.
5. The response is added to the state using the `add_messages` reducer.
6. The graph reaches its end node and returns the result.

A checkpointer and a consistent `thread_id` allow the application to maintain conversation history across multiple graph invocations.

## 🔮 Planned Enhancements

- Tool calling and custom Python tools
- Conditional routing and multi-step agent execution
- Persistent conversation storage
- Retrieval-Augmented Generation (RAG)
- PDF and document question answering
- Web search integration
- Streaming responses
- Streamlit chat interface
- Structured output and human-in-the-loop approval
- Testing, logging, and deployment

## 🔐 Security

- Store API keys in environment variables.
- Never commit `.env` or other secret files.
- Validate user inputs and tool arguments.
- Require appropriate approval before sensitive actions.

## 📌 Current Status

**In Development**

The current implementation provides a basic LangGraph chatbot with Groq integration and in-memory checkpointing. Advanced agent tools, RAG, and the interactive web interface are planned enhancements and may not yet be implemented.

## 👨‍💻 Author

**Ankur Kumar**

- GitHub: [@Ankur-one](https://github.com/Ankur-one)
- LinkedIn: [Ankur Kumar](https://www.linkedin.com/in/ankur-kumar-198031287/)

## 📄 License

Choose an open-source license, such as the MIT License, before distributing the project under that license.
