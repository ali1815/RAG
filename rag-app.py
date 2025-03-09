# app.py
import os
import streamlit as st
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch

# Set page config
st.set_page_config(page_title="RAG Demo with DeepSeek", page_icon="📚", layout="wide")

# App title
st.title("📚 RAG Application with DeepSeek")

# Sidebar with user instructions
with st.sidebar:
    st.header("Instructions")
    st.markdown("""
    1. Upload PDF documents
    2. Click on 'Process Documents' 
    3. Ask questions about your documents
    """)
    
    model_name = st.selectbox(
        "Select DeepSeek Model",
        ["deepseek-ai/deepseek-coder-6.7b-instruct", "deepseek-ai/deepseek-llm-7b-chat"],
        index=1
    )
    
    st.subheader("About")
    st.markdown("""
    This app uses:
    - **Streamlit** for the interface
    - **DeepSeek** model for text generation
    - **FAISS** for vector similarity search
    - **LangChain** for the RAG pipeline
    """)

# Create tabs
tab1, tab2 = st.tabs(["📁 Document Processing", "🔍 Query Documents"])

# Document Processing Tab
with tab1:
    st.header("Document Processing")
    
    # File uploader
    uploaded_files = st.file_uploader("Upload PDF documents", type="pdf", accept_multiple_files=True)
    
    if uploaded_files:
        # Create a temporary directory to store the uploaded files
        if not os.path.exists("temp_docs"):
            os.makedirs("temp_docs")
        
        # Save uploaded files to the temp directory
        for file in uploaded_files:
            with open(os.path.join("temp_docs", file.name), "wb") as f:
                f.write(file.getbuffer())
        
        st.success(f"{len(uploaded_files)} files uploaded successfully!")
        
        # Process button
        if st.button("Process Documents"):
            with st.spinner("Processing documents..."):
                # Load documents
                loader = DirectoryLoader("temp_docs", glob="**/*.pdf", loader_cls=PyPDFLoader)
                documents = loader.load()
                st.info(f"Loaded {len(documents)} document pages.")
                
                # Split documents into chunks
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200,
                    length_function=len
                )
                chunks = text_splitter.split_documents(documents)
                st.info(f"Split into {len(chunks)} chunks.")
                
                # Create embeddings
                embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2",
                    model_kwargs={'device': 'cpu'} # Force CPU to avoid CUDA issues
                )
                
                # Create vector store
                vectorstore = FAISS.from_documents(chunks, embeddings)
                vectorstore.save_local("faiss_index")
                
                st.success("Documents processed and indexed successfully!")

# Query Documents Tab
with tab2:
    st.header("Query Your Documents")
    
    # Check if index exists
    if os.path.exists("faiss_index"):
        # Load DeepSeek model
        @st.cache_resource
        def load_llm(model_name):
            st.info(f"Loading {model_name}... This may take a few minutes.")
            
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto",
                trust_remote_code=True
            )
            
            # Create pipeline
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=512,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,
                device_map="auto"
            )
            
            llm = HuggingFacePipeline(pipeline=pipe)
            return llm
        
        # Load the model
        with st.spinner("Preparing the model... This may take a while for the first time."):
            llm = load_llm(model_name)
            st.success("Model loaded successfully!")
        
        # Load embeddings and vector store
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'} # Force CPU to avoid CUDA issues
        )
        vectorstore = FAISS.load_local("faiss_index", embeddings)
        
        # Create retriever
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        
        # Create QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )
        
        # Query input
        query = st.text_input("Ask a question about your documents")
        
        if query:
            with st.spinner("Generating answer..."):
                result = qa_chain({"query": query})
                
                # Display answer
                st.header("Answer")
                st.write(result["result"])
                
                # Display sources
                st.header("Sources")
                for i, doc in enumerate(result["source_documents"]):
                    st.markdown(f"**Source {i+1}**")
                    st.markdown(f"**Content:** {doc.page_content[:300]}...")
                    st.markdown(f"**Source:** {doc.metadata.get('source', 'Unknown')}, Page: {doc.metadata.get('page', 'Unknown')}")
                    st.divider()
    else:
        st.warning("Please upload and process documents first in the Document Processing tab.")

# Clean up function (optional)
def cleanup():
    import shutil
    if os.path.exists("temp_docs"):
        shutil.rmtree("temp_docs")

# Uncomment to enable cleanup when the app stops
# atexit.register(cleanup)
