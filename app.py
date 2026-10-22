
import streamlit as st
from answer import answer_question

st.set_page_config(
    page_title="HR Policy Assistant",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Premium dark theme ----------
st.markdown("""
<style>
.stApp {
    background: #0b0d12;
    color: #f1f3f9;
}

[data-testid="stHeader"] {
    background: rgba(11, 13, 18, 0.9);
}

[data-testid="stSidebar"] {
    background: #10131b;
    border-right: 1px solid #252a38;
}

.hero {
    padding: 1.5rem 0 1rem 0;
}

.hero h1 {
    font-size: 2.5rem;
    font-weight: 750;
    letter-spacing: -1px;
    margin-bottom: 0.5rem;
    color: #f5f3ff;
}

.hero p {
    color: #a5a9ba;
    font-size: 1rem;
}

.eyebrow {
    color: #b5a2ff;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
}

div[data-testid="stChatMessage"] {
    border: 1px solid #252a38;
    border-radius: 16px;
    padding: 1rem;
    margin-bottom: 0.75rem;
    background: #12151e;
}

div[data-testid="stChatInput"] {
    border-color: #37314f;
    border-radius: 18px;
}

div[data-testid="stChatInput"] textarea {
    color: #f1f3f9;
}

div.stButton > button {
    border-radius: 12px;
    border: 1px solid #302a46;
    background: #171622;
    color: #eee9ff;
    min-height: 42px;
    transition: all 0.2s ease;
}

div.stButton > button:hover {
    border-color: #9b87f5;
    background: #211b34;
    color: white;
}

.status {
    color: #a5a9ba;
    font-size: 0.82rem;
    padding: 0.75rem 0;
}

.footer {
    color: #777d90;
    font-size: 0.78rem;
    text-align: center;
    padding: 1.5rem 0;
}
</style>
""", unsafe_allow_html=True)

# ---------- Session state ----------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "suggested_question" not in st.session_state:
    st.session_state.suggested_question = None

# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("## 💬 HR Assistant")
    st.caption("Your company policy companion")

    st.markdown("---")
    st.markdown("### Suggested questions")

    suggestions = [
        "How many annual leave days do employees receive?",
        "How many unused annual leave days can be carried forward?",
        "How many days per week can employees work remotely?",
        "How long do managers have to approve leave requests?",
    ]

    for index, suggestion in enumerate(suggestions):
        if st.button(
            suggestion,
            key=f"suggestion_{index}",
            use_container_width=True,
        ):
            st.session_state.suggested_question = suggestion
            st.rerun()

    st.markdown("---")

    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.suggested_question = None
        st.rerun()

    st.markdown(
        '<div class="status">● Local AI · Ollama</div>',
        unsafe_allow_html=True,
    )
    st.caption("Answers are based on the indexed HR policy documents.")

# ---------- Main page ----------
st.markdown("""
<div class="hero">
    <div class="eyebrow">YOUR WORKPLACE KNOWLEDGE BASE</div>
    <h1>HR Policy Assistant</h1>
    <p>
        Get clear answers about annual leave, remote work,
        leave applications, and company policies.
    </p>
</div>
""", unsafe_allow_html=True)

if not st.session_state.messages:
    st.info(
        "👋 Welcome! Ask a question below or select a suggested "
        "question from the sidebar to get started."
    )

# Display conversation history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
typed_question = st.chat_input("Ask anything about HR policies...")

question = (
    st.session_state.suggested_question
    or typed_question
)

if question:
    st.session_state.suggested_question = None

    st.session_state.messages.append({
        "role": "user",
        "content": question,
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching policies and preparing your answer..."):
            try:
                answer = answer_question(question)
            except Exception as error:
                answer = (
                    "Sorry, I couldn't process that question. "
                    "Please check that Ollama is running and try again."
                )
                st.error(f"Processing error: {error}")

        st.markdown(answer)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
    })

    st.rerun()

st.markdown(
    '<div class="footer">'
    'Powered by local embeddings · ChromaDB · Ollama'
    '</div>',
    unsafe_allow_html=True,
)
