# ==========================================================
# 🧠 IN-MEMORY DATABASE & LINK PIPELINE
# ==========================================================
@st.cache_resource
def load_rag_chain():
    embeddings = OfficeFirewallSafeEmbeddings()
    collection_name = "must_university_info"
    
    # Use pure in-memory client to completely kill folder locks
    from qdrant_client import QdrantClient
    client = QdrantClient(location=":memory:")
    
    with st.spinner("Loading and vectorizing university data into cloud memory..."):
        from langchain_community.document_loaders import TextLoader
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        
        if not os.path.exists("project_uni_data.txt"):
            raise FileNotFoundError("Missing 'project_uni_data.txt' file in your directory!")
            
        loader = TextLoader("project_uni_data.txt", encoding="utf-8")
        university_data = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
        splits = text_splitter.split_documents(university_data)
        
        # FIXED: Correct way to initialize with an explicit in-memory client
        qdrant_store = QdrantVectorStore(
            client=client,
            collection_name=collection_name,
            embedding=embeddings
        )
        # Add the documents into the memory store safely
        qdrant_store.add_documents(splits)
        
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