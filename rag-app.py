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
st.set_page_config(page_title="RAG Demo with Reliable Models", page_icon="📚", layout="wide")

# App title
st.title("📚 RAG Application with Reliable Models")

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
            "microsoft/phi-1_5",        # 1.3B parameter model that performs well
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0", # Excellent small model
            "TheBloke/Phi-2-DML",       # Quantized powerful model
            "google/flan-t5-base",      # Good for question answering
        ],
        index=1  # Default to TinyLlama
    )
    
    # Add device selection to manage memory
    device = st.radio(
        "Device for model inference:",
        ["cpu", "cuda", "auto"],
        index=2,  # Default to auto
        help="Choose where to run the model. Auto will use GPU if available."
    )
    
    # Add quantization option for memory efficiency
    quantize = st.checkbox(
        "Use 8-bit quantization", 
        value=True,
        help="Reduces memory usage but slightly lowers quality"
    )
    
    st.subheader("About")
    st.markdown("""
    This app uses:
    - *Streamlit* for the interface
    - *Better performing small models* for text generation
    - *FAISS* for vector similarity search
    - *LangChain* for the RAG pipeline
    - *Optimized settings* for reliable results
    """)
    
    st.info("💡 Tip: Use 8-bit quantization on machines with limited RAM.")

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

class CustomChatLLM(LLM):
    def __init__(self, pipe, model_name):
        self.pipe = pipe
        self.model_name = model_name
        super().__init__()
        
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        # Different formatting depending on model type
        if "tinyllama" in self.model_name.lower():
            formatted_prompt = f"<|system|>\nYou are a helpful assistant that answers questions based on provided context.\n<|user|>\nContext: {prompt}\nAnswer the question based on the context.\n<|assistant|>"
        elif "phi" in self.model_name.lower():
            formatted_prompt = f"Context: {prompt}\n\nAnswer based on the above context only."
        else:
            formatted_prompt = f"Context: {prompt}\n\nQuestion: Using the above context, please provide an answer.\nAnswer:"
        
        result = self.pipe(formatted_prompt, max_new_tokens=256)[0]["generated_text"]
        
        # Extract just the response based on model type
        if "tinyllama" in self.model_name.lower():
            try:
                response = result.split("<|assistant|>")[-1].strip()
                return response
            except:
                return result
        elif "phi" in self.model_name.lower():
            # Phi models usually complete directly
            try:
                response = result.split("Answer based on the above context only.")[-1].strip()
                return response
            except:
                return result
        else:
            try:
                response = result.split("Answer:")[-1].strip()
                return response
            except:
                return result
        
    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"name": self.model_name}
        
    @property
    def _llm_type(self) -> str:
        return "custom_chat_llm"

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
        def load_llm(model_name, device="auto", use_8bit=False):
            st.info(f"Loading {model_name}... This may take a few minutes.")
            
            try:
                # Determine the device
                if device == "auto":
                    device_map = "auto"
                    is_cuda = torch.cuda.is_available()
                else:
                    device_map = device
                    is_cuda = device == "cuda"
                
                # Set the quantization configuration
                if use_8bit and is_cuda:
                    quantization_config = {"load_in_8bit": True}
                else:
                    quantization_config = {}
                
                # Specific handling for T5 models which use a different model class
                if "t5" in model_name.lower():
                    tokenizer = AutoTokenizer.from_pretrained(model_name)
                    model = AutoModelForSeq2SeqLM.from_pretrained(
                        model_name,
                        torch_dtype=torch.float16 if is_cuda else torch.float32,
                        low_cpu_mem_usage=True,
                        device_map=device_map,
                        **quantization_config
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
                
                # For all other models (better performing small models)
                else:
                    tokenizer = AutoTokenizer.from_pretrained(model_name)
                    # Make sure padding token is set
                    if tokenizer.pad_token is None:
                        tokenizer.pad_token = tokenizer.eos_token
                    
                    model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        torch_dtype=torch.float16 if is_cuda else torch.float32,
                        low_cpu_mem_usage=True,
                        device_map=device_map,
                        **quantization_config
                    )
                    
                    # Create pipeline with optimized settings
                    pipe = pipeline(
                        "text-generation",
                        model=model,
                        tokenizer=tokenizer,
                        max_new_tokens=256,
                        temperature=0.1,  # Lower temperature for more focused responses
                        top_p=0.95,
                        repetition_penalty=1.15,
                        do_sample=True
                    )
                    
                    # Return custom Chat LLM
                    return CustomChatLLM(pipe, model_name)
                    
            except Exception as e:
                st.error(f"Error loading model: {str(e)}")
                st.error("Try a different model or reduce memory usage.")
                return None
        
        # Load the model
        with st.spinner("Preparing the model... This may take a while for the first time."):
            llm = load_llm(model_name, device=device, use_8bit=quantize)
            if llm:
                st.success("Model loaded successfully!")
                
                try:
                    # Load vector store with the allow_dangerous_deserialization flag
                    vector_store = FAISS.load_local(
                        vector_store_path, 
                        embeddings,
                        allow_dangerous_deserialization=True
                    )
                    
                    # Create retriever with more retrieved documents for better context
                    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
                    
                    # Create QA chain with return_source_documents=True to see sources
                    qa_chain = RetrievalQA.from_chain_type(
                        llm=llm,
                        chain_type="stuff",  # Combines documents into a single context
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
                                st.error("This might be due to model limitations. Try a different model or reformulate your question.")
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
    
    # Show RAM warning if we're using a lot of memory
    if memory_usage_mb > 4000:  # Over 4GB
        st.sidebar.warning("High memory usage detected. Consider using 8-bit quantization or a smaller model.")

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
