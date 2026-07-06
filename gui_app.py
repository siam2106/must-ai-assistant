# ==========================================================
# 🧠 LOAD OR CREATE DATABASE & LINK PIPELINE
# ==========================================================
@st.cache_resource
def load_rag_chain():
    embeddings = OfficeFirewallSafeEmbeddings()
    db_path = "./qdrant_university_db"
    collection_name = "must_university_info"
    
    # 🔓 FORCE CLIENT RESET: Prevents "already accessed by another instance" locks during hot-reloads
    from qdrant_client import QdrantClient
    try:
        import gc
        # Clear out any stale, dangling client objects lurking in memory
        gc.collect()
    except Exception:
        pass

    try:
        # Try to load existing collection safely
        qdrant_store = QdrantVectorStore.from_existing_collection(
            embedding=embeddings,
            path=db_path,
            collection_name=collection_name,
        )
    except Exception:
        # If it fails, falls back, or is locked, initialize fresh documents
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