import streamlit as st
from streamlit_extras.stylable_container import stylable_container
from langchain.memory import ConversationBufferWindowMemory
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
import os

# --- CONFIGURATION ---
GOOGLE_API_KEY = "AIzaSyDxsw9J-iEMveIeydO9o3qaJKZoT6VJaZ4"
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

st.set_page_config(page_title="Bakery Assistant", page_icon="🍰", layout="centered")

# --- Custom CSS for Beautiful Chat UI ---
st.markdown("""
    <style>
        .chat-bubble {
            border-radius: 12px;
            padding: 12px 16px;
            margin: 8px 0;
            max-width: 80%;
            word-wrap: break-word;
            animation: fadeIn 0.4s ease-in-out;
        }
        .user {
            background-color: #DCF8C6;
            align-self: flex-end;
        }
        .bot {
            background-color: #F1F0F0;
            align-self: flex-start;
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px);}
            to { opacity: 1; transform: translateY(0);}
        }
    </style>
""", unsafe_allow_html=True)

# --- Initialize Memory (keep last 20 messages only) ---
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferWindowMemory(k=20)

# --- Initialize Gemini 1.5 Flash Model ---
llm = ChatGoogleGenerativeAI(
    model="models/gemini-1.5-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.6
)

# --- Define Prompt Template ---
prompt_template = PromptTemplate(
    input_variables=["history", "input"],
    template="""
You are a helpful Bakery Assistant.
Previous conversation:
{history}

Human: {input}
Assistant:"""
)

# --- Conversation Chain ---
conversation = ConversationChain(
    llm=llm,
    prompt=prompt_template,
    memory=st.session_state.memory
)

# --- Display Header ---
st.markdown("<h1 style='text-align: center; color: #FF69B4;'>🍰 Bakery Assistant</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Ask anything about cakes, pastries, timings or placing an order!</p>", unsafe_allow_html=True)

# --- Initialize Chat History ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- Input Box ---
user_input = st.text_input("👤 You:", "", key="user_input")

# --- Generate Bot Response ---
if user_input:
    response = conversation.run(user_input)

    # Save to history
    st.session_state.chat_history.append(("user", user_input))
    st.session_state.chat_history.append(("bot", response))

    # Limit to last 20 messages (10 pairs)
    st.session_state.chat_history = st.session_state.chat_history[-20:]

# --- Display Chat History ---
for sender, message in st.session_state.chat_history:
    role_class = "user" if sender == "user" else "bot"
    with stylable_container(f"{sender}_{message}", css_styles=f""):
        st.markdown(
            f'<div class="chat-bubble {role_class}">{message}</div>',
            unsafe_allow_html=True
        )
