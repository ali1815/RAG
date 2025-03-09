# app.py
import os
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
import torch
import tempfile
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from langchain.llms.base import LLM
from typing import Optional, List, Mapping, Any

# Set page config
st.set_page_config(page_title="RAG Demo with Lightweight Models", page_icon="📚", layout="wide")

# App title
st.title("📚 RAG Application with Lightweight Models")

# Sidebar with user instructions
with st.sidebar:
    st.header("Instructions")
    st.markdown("""
    1. Upload PDF documents
    2. Click on 'Process Documents' 
    3. Ask questions about your documents
    """)
    
    model_name = st.selectbox(
        "Select Language Model",
        [
            "google/flan-t5-small",  # 80M parameters
            "google/flan-t5-base",   # 250M parameters
            "facebook/opt-350m",     # 350M parameters
            "facebook/opt-1.3b",     # 1.3B parameters
            "mistralai/Mistral-7B-Instruct-v0.1"  # 7B parameters
        ],
        index=2  # Default to facebook/opt-350m which you mentioned is working
    )
    
    st.subheader("About")
    st.markdown("""
    This app uses:
    - *Streamlit* for the interface
    - *Lightweight Hugging Face models* for text generation
    - *FAISS* for vector similarity search
    - *LangChain* for the RAG pipeline
    """)
    
    st.info("💡 Tip: Models like flan-t5-small and flan-t5-base are much lighter and faster, suitable for machines with limited RAM.")

# Create a custom embedding class using SentenceTransformer directly
class CustomEmbeddings:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name, device="cpu")
        
    def embed_documents(self, texts):
        embeddings = self.model.encode(texts)
        return embeddings.tolist()
    
    def embed_query(self, text):
        embedding = self.model.encode([text])[0]
        return embedding.tolist()
    
    # Add this method to make the object callable
    def __call__(self, text):
        if isinstance(text, list):
            return self.embed_documents(text)
        return self.embed_query(text)

# Custom LLM classes for different model types
class CustomT5LLM(LLM):
    def __init__(self, pipe):
        self.pipe = pipe
        super().__init__()
        
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        result = self.pipe(prompt, max_length=512)[0]["generated_text"]
        return result
        
    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"name": "T5 Model"}
        
    @property
    def _llm_type(self) -> str:
        return "custom_t5"

class CustomOPTLLM(LLM):
    def __init__(self, pipe):
        self.pipe = pipe
        super().__init__()
        
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        formatted_prompt = f"Question: {prompt}\nAnswer:"
        result = self.pipe(formatted_prompt, max_new_tokens=100)[0]["generated_text"]
        # Try to extract just the answer part
        try:
            response = result.split("Answer:")[-1].strip()
            return response
        except:
            # If extraction fails, return the full result
            return result
        
    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"name": "OPT Model"}
        
    @property
    def _llm_type(self) -> str:
        return "custom_opt"

class CustomMistralLLM(LLM):
    def __init__(self, pipe):
        self.pipe = pipe
        super().__init__()
        
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        formatted_prompt = f"<s>[INST] {prompt} [/INST]"
        result = self.pipe(formatted_prompt, max_new_tokens=256)[0]["generated_text"]
        # Try to extract just the response part
        try:
            response = result.split("[/INST]")[-1].strip()
            return response
        except:
            # If extraction fails, return the full result
            return result
        
    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"name": "Mistral Model"}
        
    @property
    def _llm_type(self) -> str:
        return "custom_mistral"

# Create tabs
tab1, tab2 = st.tabs(["📁 Document Processing", "🔍 Query Documents"])

# Document Processing Tab
with tab1:
    st.header("Document Processing")
    
    # File uploader
    uploaded_files = st.file_uploader("Upload PDF documents", type="pdf", accept_multiple_files=True)
    
    if uploaded_files:
        # Create a temporary directory to store the uploaded files
        temp_dir = tempfile.mkdtemp()
        st.session_state['temp_dir'] = temp_dir
        
        # Save uploaded files to the temp directory
        file_paths = []
        for file in uploaded_files:
            file_path = os.path.join(temp_dir, file.name)
            with open(file_path, "wb") as f:
                f.write(file.getbuffer())
            file_paths.append(file_path)
        
        st.success(f"{len(uploaded_files)} files uploaded successfully!")
        
        # Process button
        if st.button("Process Documents"):
            with st.spinner("Processing documents..."):
                # Load documents
                documents = []
                for file_path in file_paths:
                    loader = PyPDFLoader(file_path)
                    documents.extend(loader.load())
                
                st.info(f"Loaded {len(documents)} document pages.")
                
                # Split documents into chunks
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200,
                    length_function=len
                )
                chunks = text_splitter.split_documents(documents)
                st.info(f"Split into {len(chunks)} chunks.")
                
                # Create custom embeddings
                embeddings = CustomEmbeddings()
                
                # Create vector store
                try:
                    # Extract texts and metadata from chunks
                    texts = [chunk.page_content for chunk in chunks]
                    metadatas = [chunk.metadata for chunk in chunks]
                    
                    # Create FAISS index using from_texts
                    vector_store = FAISS.from_texts(
                        texts,
                        embeddings,  # Pass the embeddings object directly
                        metadatas=metadatas
                    )
                    
                    # Save vector store
                    vector_store_path = os.path.join(temp_dir, "faiss_index")
                    vector_store.save_local(vector_store_path)
                    st.session_state['vector_store_path'] = vector_store_path
                    
                    st.success("Documents processed and indexed successfully!")
                except Exception as e:
                    st.error(f"Error processing documents: {str(e)}")

# Query Documents Tab
with tab2:
    st.header("Query Your Documents")
    
    # Check if vector store exists in session state
    if 'vector_store_path' in st.session_state and os.path.exists(st.session_state['vector_store_path']):
        vector_store_path = st.session_state['vector_store_path']
        
        # Initialize custom embeddings
        embeddings = CustomEmbeddings()
        
        # Load language model
        @st.cache_resource
        def load_llm(model_name):
            st.info(f"Loading {model_name}... This may take a few minutes.")
            
            try:
                # Specific handling for T5 models which use a different model class
                if "t5" in model_name.lower():
                    tokenizer = AutoTokenizer.from_pretrained(model_name)
                    model = AutoModelForSeq2SeqLM.from_pretrained(
                        model_name,
                        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                        low_cpu_mem_usage=True
                    )
                    
                    # T5 models use 'text2text-generation' task
                    pipe = pipeline(
                        "text2text-generation",
                        model=model,
                        tokenizer=tokenizer,
                        max_length=512
                    )
                    
                    # Create a custom LLM that wraps the T5 pipeline
                    return CustomT5LLM(pipe)
                
                # For Mistral models
                elif "mistral" in model_name.lower():
                    tokenizer = AutoTokenizer.from_pretrained(model_name)
                    model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                        low_cpu_mem_usage=True
                    )
                    
                    # Create pipeline
                    pipe = pipeline(
                        "text-generation",
                        model=model,
                        tokenizer=tokenizer,
                        max_new_tokens=256,
                        temperature=0.7,
                        top_p=0.9,
                        repetition_penalty=1.1,
                        do_sample=True
                    )
                    
                    # Return custom Mistral LLM
                    return CustomMistralLLM(pipe)
                    
                # For OPT and other causal language models
                else:
                    tokenizer = AutoTokenizer.from_pretrained(model_name)
                    # Make sure padding token is set
                    if tokenizer.pad_token is None:
                        tokenizer.pad_token = tokenizer.eos_token
                        
                    model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                        low_cpu_mem_usage=True
                    )
                    
                    # Create pipeline with optimized settings for OPT
                    pipe = pipeline(
                        "text-generation",
                        model=model,
                        tokenizer=tokenizer,
                        max_new_tokens=100,  # Shorter responses to avoid hallucination
                        temperature=0.3,     # Lower temperature for more factual responses
                        top_p=0.9,
                        repetition_penalty=1.2
                    )
                    
                    # Return custom OPT LLM for better prompt formatting
                    if "opt" in model_name.lower():
                        return CustomOPTLLM(pipe)
                    else:
                        llm = HuggingFacePipeline(pipeline=pipe)
                        return llm
                    
            except Exception as e:
                st.error(f"Error loading model: {str(e)}")
                st.error("Try a different model or reduce memory usage.")
                return None
        
        # Load the model
        with st.spinner("Preparing the model... This may take a while for the first time."):
            llm = load_llm(model_name)
            if llm:
                st.success("Model loaded successfully!")
                
                try:
                    # Load vector store with the allow_dangerous_deserialization flag
                    vector_store = FAISS.load_local(
                        vector_store_path, 
                        embeddings,
                        allow_dangerous_deserialization=True
                    )
                    
                    # Create retriever
                    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
                    
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
                            try:
                                result = qa_chain({"query": query})
                                
                                # Display answer
                                st.header("Answer")
                                st.write(result["result"])
                                
                                # Display sources
                                st.header("Sources")
                                for i, doc in enumerate(result["source_documents"]):
                                    st.markdown(f"*Source {i+1}*")
                                    st.markdown(f"*Content:* {doc.page_content[:300]}...")
                                    st.markdown(f"*Source:* {doc.metadata.get('source', 'Unknown')}, Page: {doc.metadata.get('page', 'Unknown')}")
                                    st.divider()
                            except Exception as e:
                                st.error(f"Error generating answer: {str(e)}")
                                st.error("This might be due to model limitations. Try with a simpler question or a different model.")
                except Exception as e:
                    st.error(f"Error loading vector store: {str(e)}")
    else:
        st.warning("Please upload and process documents first in the Document Processing tab.")

# Add memory usage monitoring
if st.sidebar.checkbox("Show Memory Usage"):
    import psutil
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    memory_usage_mb = memory_info.rss / 1024 / 1024
    st.sidebar.info(f"Current Memory Usage: {memory_usage_mb:.2f} MB")

# Handle session state cleanup
if 'temp_dir' in st.session_state and os.path.exists(st.session_state['temp_dir']):
    # Add an option to clear processed files
    if st.sidebar.button("Clear processed files"):
        import shutil
        try:
            shutil.rmtree(st.session_state['temp_dir'])
            st.session_state.pop('temp_dir', None)
            st.session_state.pop('vector_store_path', None)
            st.sidebar.success("Processed files cleared successfully!")
            st.experimental_rerun()
        except Exception as e:
            st.sidebar.error(f"Error clearing files: {str(e)}")

# Add system info in the sidebar
st.sidebar.subheader("System Information")
st.sidebar.info(f"PyTorch CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    st.sidebar.info(f"GPU: {torch.cuda.get_device_name(0)}")
    st.sidebar.info(f"CUDA Version: {torch.version.cuda}")
