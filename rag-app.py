@st.cache_resource
def load_llm(model_name):
    st.info(f"Loading {model_name}... This may take a few minutes.")
    
    try:
        # Specific handling for T5 models
        if "t5" in model_name.lower():
            from transformers import AutoModelForSeq2SeqLM  # Use Seq2Seq model for T5
            
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                low_cpu_mem_usage=True
            )
            
            # Create T5 pipeline with correct task
            pipe = pipeline(
                "text2text-generation",
                model=model,
                tokenizer=tokenizer,
                max_length=512
            )
            
            # Wrapper for LangChain compatibility
            from langchain.llms.base import LLM
            from typing import Optional, List, Mapping, Any
            
            class CustomT5LLM(LLM):
                def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
                    result = pipe(prompt, max_length=512)[0]["generated_text"]
                    return result
                    
                @property
                def _identifying_params(self) -> Mapping[str, Any]:
                    return {"name": model_name}
                    
                @property
                def _llm_type(self) -> str:
                    return "custom_t5"
            
            return CustomT5LLM()
            
        # For Mistral models
        elif "mistral" in model_name.lower():
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                low_cpu_mem_usage=True
            )
            
            # Create pipeline with better parameters for instruction models
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=256,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
            
            # Add proper instruction formatting function
            def generate_text(prompt):
                formatted_prompt = f"<s>[INST] {prompt} [/INST]"
                result = pipe(formatted_prompt)[0]["generated_text"]
                # Extract only the response part
                response = result.split("[/INST]")[-1].strip()
                return response
                
            # Custom LLM class for Mistral
            from langchain.llms.base import LLM
            
            class MistralLLM(LLM):
                def _call(self, prompt, stop=None):
                    return generate_text(prompt)
                
                @property
                def _identifying_params(self):
                    return {"name": model_name}
                
                @property
                def _llm_type(self):
                    return "mistral"
                    
            return MistralLLM()
            
        # For OPT models
        elif "opt" in model_name.lower():
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            # Make sure padding token is set
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                low_cpu_mem_usage=True
            )
            
            # Create pipeline with better parameters for OPT
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=100,  # Shorter responses to avoid hallucination
                temperature=0.3,     # Lower temperature for more factual responses
                top_p=0.9,
                repetition_penalty=1.2,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
            
            # Add proper formatting function for OPT
            def generate_text(prompt):
                formatted_prompt = f"Question: {prompt}\nAnswer:"
                result = pipe(formatted_prompt)[0]["generated_text"]
                # Extract only the answer part
                response = result.split("Answer:")[-1].strip()
                return response
                
            # Custom LLM class for OPT
            from langchain.llms.base import LLM
            
            class OptLLM(LLM):
                def _call(self, prompt, stop=None):
                    return generate_text(prompt)
                
                @property
                def _identifying_params(self):
                    return {"name": model_name}
                
                @property
                def _llm_type(self):
                    return "opt"
                    
            return OptLLM()
        
        # Default case for other models
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                low_cpu_mem_usage=True
            )
            
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=256,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1
            )
            
            llm = HuggingFacePipeline(pipeline=pipe)
            return llm
            
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None
