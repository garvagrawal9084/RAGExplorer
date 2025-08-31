import streamlit as st
from indexing import indexing
from openai import OpenAI
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
import os

load_dotenv()

# Set page config
st.set_page_config(page_title="RAG Project", page_icon="📄", layout="centered")

st.title("📚 RAG PROJECT")

# Sidebar for API key
# with st.sidebar:
#     st.markdown("## 🔑 OpenAI API Key")
#     st.markdown("Please enter your OpenAI API key below to use the app.")
#     api_key_input = st.text_input("API Key", type="password", placeholder="sk-...")

#     if api_key_input:
#         st.session_state.openai_api_key = api_key_input

# Check for API key before showing uploader
# if st.session_state.get("openai_api_key"):
#     uploaded_file = st.file_uploader("📤 Upload your file")
# else:
#     st.warning("⚠️ Please enter your OpenAI API key in the sidebar to continue.")
    # st.stop()

uploaded_file = st.file_uploader("📤 Upload your file")

# Run indexing after file upload
if uploaded_file:
    response = indexing(uploaded_file)

    if response:
        st.success(response)

        # Initialize message histories
        if "full_messages" not in st.session_state:
            st.session_state.full_messages = []

        if "display_messages" not in st.session_state:
            st.session_state.display_messages = []

        # Chat input
        if query := st.chat_input("💬 Enter what you want to search"):

            with st.spinner("🔍 Searching from docs..."):
                # Use user's API key
                client = OpenAI()

                embedding = OpenAIEmbeddings(
                    model="text-embedding-3-large",
                )

                vectorDB = QdrantVectorStore.from_existing_collection(
                    url = "http://localhost:6333" ,
                    collection_name=uploaded_file.name.split(".")[0],
                    embedding=embedding
                )

                search_result = vectorDB.similarity_search(query=query)

                context = "\n\n".join(
                    [f"Page Content: {result.page_content}\nPage Number: {result.metadata.get('page_label', 'N/A')}" for result in search_result]
                )

                SYSTEM_PROMPT = f"""
                You are a helpful AI assistant. You will help the user based on the retrieved context from a PDF file, along with page content and page number.

                Use only the provided context to respond. Suggest the page number for the user to check full details.

                Context:
                {context}
                """

                # Store messages for memory
                st.session_state.full_messages.append({"role": "system", "content": SYSTEM_PROMPT})
                st.session_state.full_messages.append({"role": "user", "content": query})
                st.session_state.display_messages.append({"role": "user", "content": query})

                # Get response from OpenAI
                response = client.chat.completions.create(
                    model="gpt-5-mini",  # or gpt-3.5-turbo, gpt-4 etc.
                    messages=st.session_state.full_messages,
                )

                assistant_content = response.choices[0].message.content

                # Save response
                st.session_state.full_messages.append({"role": "assistant", "content": assistant_content})
                st.session_state.display_messages.append({"role": "assistant", "content": assistant_content})

                # Display messages
                for msg in st.session_state.display_messages:
                    with st.chat_message(name=msg["role"], avatar="👤" if msg["role"] == "user" else "🤖"):
                        st.write(msg["content"])
