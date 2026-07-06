import os
import streamlit as st
from langchain_qdrant import QdrantVectorStore
from langchain_groq import ChatGroq

# Import from classic chains and core types
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.embeddings import Embeddings

# ==========================================================
# 🛡️ SYSTEM EMBEDDINGS (Matches your database vectors)
# ==========================================================
class OfficeFirewallSafeEmbeddings(Embeddings):
    def __init__(self):
        self.dims = 384 

    def _get_hash_vector(self, text: str) -> list[float]:
        import hashlib
        import math
        state = hashlib.sha256(text.encode('utf-8')).digest()
        vector = []
        for i in range(self.dims):
            byte_idx = (i * 3) % len(state)
            val = state[byte_idx] + (state[(byte_idx + 1) % len(state)] * 256)
            coord = math.sin(val + i)
            vector.append(coord)
        magnitude = math.sqrt(sum(x*x for x in vector))
        return [x / magnitude for x in vector]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._get_hash_vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._get_hash_vector(text)

# ==========================================================
# ⚙️ CONFIGURATION & API SECURITY BYPASS
# ==========================================================
st.set_page_config(page_title="MUST AI Assistant", page_icon="🎓", layout="centered")
st.title("🎓 MUST University AI Assistant")
st.caption("Connected to Qdrant Vector DB")

# Safe multi-environment credential fallback
try:
    api_key_string = st.secrets["GROQ_API_KEY"]
except Exception:
    api_key_string = "PASTE_YOUR_GROQ_API_KEY_HERE"

os.environ["GROQ_API_KEY"] = api_key_string

# ==========================================================
# 🧠 LOAD OR CREATE DATABASE & LINK PIPELINE
# ==========================================================
@st.cache_resource
def load_rag_chain():
    embeddings = OfficeFirewallSafeEmbeddings()
    db_path = "./qdrant_university_db"
    collection_name = "must_university_info"
    
    # 🔓 FORCE CLIENT RESET: Prevents locks during hot-reloads
    import gc
    gc.collect()

    try:
        # Try to load existing collection safely
        qdrant_store = QdrantVectorStore.from_existing_collection(
            embedding=embeddings,
            path=db_path,
            collection_name=collection_name,
        )
    except Exception:
        # If it fails, build it from the text file instantly
        with st.spinner("Initializing vector database build from text file..."):
            from langchain_community.document_loaders import TextLoader
            from langchain_text_splitters import RecursiveCharacterTextSplitter
            
            if not os.path.exists("project_uni_data.txt"):
                raise FileNotFoundError("Missing 'project_uni_data.txt' file in your directory!")
                
            loader = TextLoader("project_uni_data.txt", encoding="utf-8")
            university_data = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
            splits = text_splitter.split_documents(university_data)
            
            qdrant_store = QdrantVectorStore.from_documents(
                documents=splits,
                embedding=embeddings,
                path=db_path,
                collection_name=collection_name,
            )
        
    retriever = qdrant_store.as_retriever(search_kwargs={"k": 4})
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.2)
    
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", "Given a chat history and the latest user question, formulate a standalone question. Do NOT answer it."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
    
    system_prompt = (
        "You are the official AI Assistant for Mody University of Science and Technology (MUST).\n"
        "Answer campus inquiries strictly using the verified context blocks provided below.\n\n"
        "CRITICAL NOTE ON FEES:\n"
        "All fees stated represent Academic Tuition Fees ONLY. Exclude hostel, uniforms, etc.\n\n"
        "Context:\n{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    return create_retrieval_chain(history_aware_retriever, question_answer_chain)

# Initialize pipeline seamlessly
try:
    rag_chain = load_rag_chain()
except Exception as e:
    st.error(f"Failed to initialize database pipeline: {e}")
    st.stop()

# Persistent session states for browser interface
if "messages" not in st.session_state:
    st.session_state.messages = []
if "langchain_history" not in st.session_state:
    st.session_state.langchain_history = []

# ==========================================================
# 💬 RENDER CHAT WINDOW
# ==========================================================
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_input := st.chat_input("Ask me about MUST (courses, fees, hostels)..."):
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    with st.chat_message("assistant"):
        with st.spinner("Searching database..."):
            response = rag_chain.invoke({
                "input": user_input, 
                "chat_history": st.session_state.langchain_history
            })
            answer = response['answer']
            st.markdown(answer)
            
    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.session_state.langchain_history.extend([
        HumanMessage(content=user_input),
        AIMessage(content=answer),
    ])