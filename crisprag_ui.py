#!/usr/bin/env python3
"""
mlxrag_ui.py: Streamlit UI for MLXrag vector search engine

Features:
- Document indexing with multiple backends (Qdrant, Milvus, LanceDB, etc.)
- Multiple embedding providers (MLX, Ollama, FastEmbed)
- Advanced search with hybrid retrieval
- Collapsible settings for cleaner interface
"""

import streamlit as st
import threading
import os
import tempfile
import shutil
import time
import sys
import json
from typing import List, Dict, Any, Tuple, Optional
import fnmatch
from pathlib import Path
import subprocess
import platform
import queue
from threading import Thread
import time
import importlib
import requests

# LLM-related constants
LLM_SYSTEM_PROMPT = """You are a helpful assistant. Answer the question based on the context provided.
If the answer cannot be determined from the context, say so clearly. Don't make things up."""

DEFAULT_OLLAMA_MODEL = "cas/llama-3.2-3b-instruct:latest"  # Default Ollama model for RAG
DEFAULT_MLX_MODEL = "mlx-community/Llama-3.2-3B-Instruct-4bit"  # Default MLX model
DEFAULT_TRANSFORMERS_MODEL = "unsloth/Llama-3.2-3B-Instruct-unsloth-bnb-4bit"  # Default HF model
DEFAULT_LLAMACPP_MODEL = "~/llama.cpp/models/Llama-3.2-3B-Instruct-Q4_K_M.gguf"  # Default path for llama.cpp model

# Enhanced function calling implementation for multiple backends

# Define the tool specification for consistent use across providers
SEARCH_TOOL_SPEC = {
    "name": "search_vector_database",
    "description": "Search for relevant information in the vector database",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query to find relevant documents"
            },
            "search_type": {
                "type": "string",
                "enum": ["hybrid", "vector", "sparse", "keyword"],
                "description": "Type of search to perform",
                "default": "hybrid"
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of results to return",
                "default": 5
            }
        },
        "required": ["query"]
    }
}

# System prompts for different stages
SEARCH_SYSTEM_PROMPT = """
You are a search assistant that helps formulate effective search queries.
When asked a question, think about what keywords and terms would best find relevant information.
Always use the search_vector_database function to perform searches. 
Create concise and specific search queries that will find relevant documents.
"""

RAG_SYSTEM_PROMPT = """
You are a helpful assistant that answers questions based on the provided context.
Always focus on the context to provide accurate answers.
If the answer cannot be determined from the context, say so clearly.
Don't make things up or provide information not supported by the context.
"""

# Global lock for MPS/GPU-critical sections
gpu_lock = threading.Lock()


# Function for the LLM to call to perform a search
def perform_vector_search(query, search_type="hybrid", limit=10):
    """
    Perform a vector search using the currently initialized database and processor.
    
    Args:
        query: The search query
        search_type: Type of search (hybrid, vector, sparse, keyword)
        limit: Maximum number of results to return
        
    Returns:
        JSON string containing search results
    """
    try:
        if not st.session_state.search_initialized or not st.session_state.db_manager or not st.session_state.processor:
            return json.dumps({"error": "Search not initialized. Please initialize search first."})
        
        # Get settings from session state
        prefetch_limit = 50  # Default
        fusion_type = "rrf"  # Default
        context_size = st.session_state.context_size if 'context_size' in st.session_state else 500
        rerank = False  # Default
        
        # Get sidebar settings if available
        if hasattr(st.session_state, 'prefetch_limit'):
            prefetch_limit = st.session_state.prefetch_limit
        if hasattr(st.session_state, 'fusion_type'):
            fusion_type = st.session_state.fusion_type
        if hasattr(st.session_state, 'rerank'):
            rerank = st.session_state.rerank
        if hasattr(st.session_state, 'score_threshold'):
            score_threshold = st.session_state.score_threshold if st.session_state.score_threshold > 0 else None
        else:
            score_threshold = None
            
        # Perform search
        results = gpu_safe_call(st.session_state.db_manager.search,
            query=query,
            search_type=search_type,
            limit=limit,
            processor=st.session_state.processor,
            prefetch_limit=prefetch_limit,
            fusion_type=fusion_type,
            relevance_tuning=True,
            context_size=context_size,
            score_threshold=score_threshold,
            rerank=rerank
        )
        
        # Store the results in session state for the interface
        st.session_state.search_results = results
        
        # Also store in the last auto search info
        st.session_state.last_auto_search = query
        st.session_state.auto_searched_queries.append(query)
        
        # Convert the results to a simplified format for the LLM
        simplified_results = []
        for result in results.get('results', []):
            simplified_results.append({
                'rank': result.get('rank', 0),
                'score': result.get('score', 0),
                'file_name': result.get('file_name', ''),
                'file_path': result.get('file_path', ''),
                'chunk_index': result.get('chunk_index', 0),
                'text': result.get('text', '')
            })
        
        results_dict = {
            'query': results.get('query', query),
            'search_type': results.get('search_type', search_type),
            'count': len(simplified_results),
            'results': simplified_results
        }
        
        # Return simplified results as JSON
        return json.dumps(results_dict)
    except Exception as e:
        error_msg = str(e)
        if st.session_state.show_debug:
            import traceback
            error_msg = traceback.format_exc()
        return json.dumps({"error": error_msg})
    
def get_search_query_from_llm(question):
    """Get search query from LLM using function calling"""
    provider = st.session_state.llm_provider
    model = st.session_state.llm_model
    
    # Different implementation based on provider
    if provider == "Ollama":
        return get_query_from_ollama(model, question)
    elif provider == "MLX":
        return get_query_from_mlx(model, question)
    elif provider == "Transformers":
        return get_query_from_transformers(model, question)
    elif provider == "llama.cpp":
        return get_query_from_llamacpp(model, question)
    else:
        # Fallback - just extract keywords from the question
        return question

def get_query_from_ollama(model, question):
    """Get search query from Ollama model using function calling"""
    try:
        import requests
        
        # Check if Ollama is running
        try:
            requests.get("http://localhost:11434/api/version", timeout=1)
        except:
            st.session_state.llm_response_queue.put("⚠️ Cannot connect to Ollama server. Defaulting to original query.")
            return question
        
        # Set up the function calling
        tools = [{
            "type": "function",
            "function": SEARCH_TOOL_SPEC
        }]
        
        # Make the request to Ollama
        url = "http://localhost:11434/api/chat"
        data = {
            "model": model,
            "messages": [
                {"role": "system", "content": SEARCH_SYSTEM_PROMPT},
                {"role": "user", "content": f"Create a search query to find information to answer this question: {question}"}
            ],
            "stream": False,
            "tools": tools
        }
        
        response = requests.post(url, json=data, timeout=30)
        response.raise_for_status()
        response_data = response.json()
        
        # Parse the response for function calls
        if "message" in response_data and "tool_calls" in response_data["message"]:
            for tool_call in response_data["message"]["tool_calls"]:
                if tool_call["function"]["name"] == "search_vector_database":
                    try:
                        arguments = json.loads(tool_call["function"]["arguments"])
                        if "query" in arguments:
                            return arguments["query"]
                    except:
                        pass
        
        # If no function call or error, check if the model gave a textual search suggestion
        if "message" in response_data and "content" in response_data["message"]:
            content = response_data["message"]["content"]
            # Look for search terms in quotes
            import re
            search_terms = re.findall(r'"([^"]*)"', content)
            if search_terms:
                return search_terms[0]
        
        # If all else fails, return the original question
        return question
    
    except Exception as e:
        if st.session_state.show_debug:
            st.error(f"Error in Ollama function calling: {e}")
        return question  # Fallback to using the question itself

def get_query_from_mlx(model, question):
    """Get search query from MLX model"""
    try:
        # Import mlx-lm here to avoid loading on startup
        from mlx_lm import load, generate
        import mlx.core as mx
        
        st.session_state.llm_response_queue.put("Using MLX model to improve search query...")
        
        try:
            # Load the model and tokenizer
            model, tokenizer = gpu_safe_call(load, model)
            
            # Create prompt for query extraction
            prompt = f"""I need to search a document database to answer the following question:

Question: "{question}"

Give me a concise, specific search query that will find relevant information to answer this question.
Your response should be just the search query with no other text or explanation.

Search query:"""
            
            # Generate the response
            response = gpu_safe_call(generate, model, tokenizer, 
                prompt=prompt,
                max_tokens=64,
                temperature=0.1,
                repetition_penalty=1.1,
                verbose=False
            )
            
            # Clean up the response
            query = response.strip()
            
            # Remove any common prefixes that models might generate
            prefixes_to_remove = [
                "Search query:", "Query:", "The search query is:", 
                "I would search for:", "Search for:"
            ]
            
            for prefix in prefixes_to_remove:
                if query.startswith(prefix):
                    query = query[len(prefix):].strip()
            
            # Remove quotes if present
            if query.startswith('"') and query.endswith('"'):
                query = query[1:-1].strip()
            
            if query and len(query) >= 3:
                return query
            
            # If we got an empty or too short query, fall back to the original question
            return question
            
        except Exception as e:
            if st.session_state.show_debug:
                st.error(f"Error in MLX query extraction: {e}")
                import traceback
                st.error(traceback.format_exc())
            
            # Fall back to keyword extraction
            # Define important keywords (domain-specific terms should be prioritized)
            important_keywords = set([
                "ai", "algorithm", "data", "machine learning", "neural network", 
                "statistics", "computer", "inference", "model", "dataset",
                "training", "analysis", "search", "vector", "embedding",
                "transformer", "llm", "bert", "rag", "python"
            ])
            
            # Simple keyword extraction
            words = question.lower().split()
            keywords = [word for word in words if word in important_keywords]
            
            # If no keywords found, return the original question
            if not keywords:
                return question
            
            # Create a more focused query with the keywords
            return " ".join(keywords)
            
    except Exception as e:
        if st.session_state.show_debug:
            st.error(f"Error in MLX query extraction: {e}")
            import traceback
            st.error(traceback.format_exc())
        return question

def get_mlx_alternative_query(model, question, first_query):
    """Generate an alternative search query using MLX when first query fails"""
    try:
        # Import mlx-lm here to avoid loading on startup
        from mlx_lm import load, generate
        import mlx.core as mx
        
        # Load the model and tokenizer
        model, tokenizer = gpu_safe_call(load, model)
        
        # Create prompt for alternative query generation
        prompt = f"""I tried searching for information with the query: "{first_query}"
But I didn't find relevant results.

Original question: "{question}"

Please provide an alternative search query that might find better results.
Your response should be just the alternative search query with no other text.

Alternative search query:"""
        
        # Generate the response
        response = gpu_safe_call(generate, model, tokenizer, 
            prompt=prompt,
            max_tokens=64,
            temperature=0.3,  # Slightly higher temperature for more variety
            repetition_penalty=1.1,
            verbose=False
        )
        
        # Clean up the response
        query = response.strip()
        
        # Remove any common prefixes
        prefixes_to_remove = [
            "Alternative search query:", "Query:", "New search query:", 
            "Try searching for:", "Alternative query:"
        ]
        
        for prefix in prefixes_to_remove:
            if query.startswith(prefix):
                query = query[len(prefix):].strip()
        
        # Remove quotes if present
        if query.startswith('"') and query.endswith('"'):
            query = query[1:-1].strip()
        
        if query and len(query) >= 3 and query != first_query:
            return query
        
        # If we got an empty, too short, or duplicate query, try a fallback approach
        # Extract key terms from the question but exclude terms from the first query
        first_query_terms = set(first_query.lower().split())
        question_terms = question.lower().split()
        
        # Filter out stop words and short words
        stop_words = set(["the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "with", "is", "are"])
        new_terms = [term for term in question_terms 
                    if term not in first_query_terms 
                    and term not in stop_words
                    and len(term) > 2]
        
        # If we have new terms, use them
        if new_terms:
            return " ".join(new_terms[:3])  # Use up to 3 new terms
        
        # If all else fails, use a more general version of the original query
        if len(first_query.split()) > 1:
            return first_query.split()[0]  # Just use the first word
        
        return question  # Fall back to the original question
        
    except Exception as e:
        if st.session_state.show_debug:
            st.error(f"Error generating alternative MLX query: {e}")
            import traceback
            st.error(traceback.format_exc())
        
        # Simple fallback
        return " ".join(question.split()[:3])  # First 3 words of the question

def get_query_from_transformers(model, question):
    """Get search query from Transformers model using function calling if available"""
    try:
        # Import required libraries
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
        import torch
        
        st.session_state.llm_response_queue.put("Generating search query with Transformers...")
        
        # Check if a GPU is available and if we should use it
        device = "cuda" if torch.cuda.is_available() else "cpu"
        use_gpu = device == "cuda" and st.session_state.get("use_gpu", True)
        
        # Set model precision - lower for efficiency with minimal quality loss
        dtype = torch.float16 if use_gpu else torch.bfloat16
        
        # Load tokenizer first to check if the model supports function calling
        try:
            tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        except Exception as e:
            if st.session_state.show_debug:
                st.error(f"Error loading tokenizer: {e}")
            # Early return to fallback
            return fallback_keyword_extraction(question)
        
        # Try loading the model with minimal context for query extraction
        model_kwargs = {
            "torch_dtype": dtype,
            "device_map": "auto" if use_gpu else "cpu",
            "low_cpu_mem_usage": True,
        }
        
        # For function calling support (if available)
        if hasattr(tokenizer, "function_calling"):
            supports_function_calling = True
        else:
            # Check if it might be a Llama model
            supports_function_calling = "llama" in model.lower() and any(term in model.lower() for term in ["3", "3.1", "3.2"])
        
        # Try to load the model with only necessary components
        try:
            # Set up the function calling tools for models that support it
            if supports_function_calling:
                tools = [SEARCH_TOOL_SPEC]
                model_obj = AutoModelForCausalLM.from_pretrained(
                    model, 
                    trust_remote_code=True,
                    **model_kwargs
                )
                
                # Create prompt for function calling
                messages = [
                    {"role": "system", "content": SEARCH_SYSTEM_PROMPT},
                    {"role": "user", "content": f"Create a search query to find information to answer this question: {question}"}
                ]
                
                # Format the prompt according to the model's requirements
                # This is different for different model families
                try:
                    inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to(model_obj.device)
                except Exception as e:
                    if st.session_state.show_debug:
                        st.error(f"Error applying chat template: {e}")
                    # Try a manual template for Llama-style models
                    formatted_prompt = f"<s>[INST] <<SYS>>\n{SEARCH_SYSTEM_PROMPT}\n<</SYS>>\n\nCreate a search query to find information to answer this question: {question} [/INST]"
                    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model_obj.device)
                
                # Generate with tool calling
                try:
                    with torch.no_grad():
                        outputs = model_obj.generate(
                            inputs.input_ids,
                            max_new_tokens=100,
                            temperature=0.1,
                            do_sample=True, 
                            tool_choice="auto",
                            tools=tools
                        )
                    
                    # Process the generated result for function calls
                    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=False)
                    
                    # Parse tool calls - this depends on the model
                    import re
                    
                    # Extract JSON from the output using regex
                    # Look for JSON patterns in the output
                    json_pattern = r'(\{.*?\})'
                    matches = re.findall(json_pattern, decoded_output)
                    
                    for potential_json in matches:
                        try:
                            data = json.loads(potential_json)
                            if "query" in data:
                                return data["query"]
                        except json.JSONDecodeError:
                            pass
                
                except Exception as e:
                    if st.session_state.show_debug:
                        st.error(f"Error with function calling generation: {e}")
                    # Continue to text generation approach
            
            # Standard text generation approach (fallback or primary)
            # Use text generation pipeline with simpler prompt
            gen_kwargs = {
                "max_new_tokens": 50,
                "temperature": 0.1,
                "top_p": 0.95,
                "do_sample": True,
                "repetition_penalty": 1.1
            }
            
            try:
                # Create generation pipeline
                generator = pipeline(
                    "text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    torch_dtype=dtype,
                    device_map="auto" if use_gpu else "cpu",
                    trust_remote_code=True
                )
                
                # Simple prompt format
                simple_prompt = f"""Given this question: "{question}"
Generate a search query with just the essential keywords to find relevant information.
Search query:"""
                
                # Generate
                response = generator(
                    simple_prompt,
                    **gen_kwargs
                )[0]["generated_text"]
                
                # Extract just the generated portion
                query = response[len(simple_prompt):].strip()
                
                # Clean up the query
                prefixes_to_remove = [
                    "Search query:", "Query:", "The search query is:", 
                    "I would search for:", "Search for:"
                ]
                
                for prefix in prefixes_to_remove:
                    if query.startswith(prefix):
                        query = query[len(prefix):].strip()
                
                # Remove quotes if present
                if query.startswith('"') and query.endswith('"'):
                    query = query[1:-1].strip()
                
                if query and len(query) >= 3:
                    return query
                
                # If query is too short, fall back to keyword extraction
                return fallback_keyword_extraction(question)
                
            except Exception as e:
                if st.session_state.show_debug:
                    st.error(f"Error in pipeline text generation: {e}")
                # Fall back to keyword extraction
                return fallback_keyword_extraction(question)
                
        except Exception as e:
            if st.session_state.show_debug:
                st.error(f"Error loading model: {e}")
            # Fall back to keyword extraction
            return fallback_keyword_extraction(question)
            
    except Exception as e:
        if st.session_state.show_debug:
            st.error(f"Error in Transformers query extraction: {e}")
        return fallback_keyword_extraction(question)


def fallback_keyword_extraction(question):
    """Extract keywords for search when model-based generation fails"""
    # Common stopwords to filter out
    stop_words = set([
        "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for", 
        "with", "by", "about", "against", "between", "into", "through", 
        "during", "before", "after", "above", "below", "from", "up", "down", 
        "is", "are", "am", "was", "were", "be", "been", "being", "have", 
        "has", "had", "having", "do", "does", "did", "doing", "can", "could", 
        "should", "would", "may", "might", "must", "will", "shall"
    ])
    
    # Common question words to filter out
    question_words = set([
        "what", "when", "where", "which", "who", "whom", "whose", "why", 
        "how", "is", "are", "do", "does", "did", "can", "could", "would", 
        "should", "will"
    ])
    
    # Normalize the question
    words = question.lower().replace("?", "").replace(".", "").replace(",", "").replace(";", "").split()
    
    # Remove stop words and question words, keep words of sufficient length
    keywords = [word for word in words if word not in stop_words 
                and word not in question_words 
                and len(word) > 2]
    
    # If no keywords found, just use the original words filtered by length
    if not keywords:
        keywords = [word for word in words if len(word) > 3]
    
    # If still no keywords, return the original question
    if not keywords:
        return question
    
    # Use the top 5 keywords (or fewer if there aren't 5)
    return " ".join(keywords[:5])

def get_transformers_alternative_query(model, question, first_query):
    """Generate an alternative search query using Transformers when first query fails"""
    try:
        # Import required libraries
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
        import torch
        
        # Check if a GPU is available and if we should use it
        device = "cuda" if torch.cuda.is_available() else "cpu"
        use_gpu = device == "cuda" and st.session_state.get("use_gpu", True)
        
        # Set model precision
        dtype = torch.float16 if use_gpu else torch.bfloat16
        
        # Load tokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        except Exception as e:
            if st.session_state.show_debug:
                st.error(f"Error loading tokenizer for alternative query: {e}")
            return fallback_alternative_query(question, first_query)
        
        # Try text generation pipeline approach
        try:
            # Create generation pipeline
            generator = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                torch_dtype=dtype,
                device_map="auto" if use_gpu else "cpu",
                trust_remote_code=True
            )
            
            # Create prompt for alternative query
            alt_prompt = f"""I tried searching for information with the query: "{first_query}"
But I didn't find relevant results.

Original question: "{question}"

Please provide an alternative search query that might find better results.
Just return the alternative search query with no explanation.

Alternative search query:"""
            
            # Generation parameters
            gen_kwargs = {
                "max_new_tokens": 50,
                "temperature": 0.3,  # Higher temperature for diversity
                "top_p": 0.95,
                "do_sample": True,
                "repetition_penalty": 1.1
            }
            
            # Generate
            response = generator(
                alt_prompt,
                **gen_kwargs
            )[0]["generated_text"]
            
            # Extract just the generated portion
            query = response[len(alt_prompt):].strip()
            
            # Clean up the query
            prefixes_to_remove = [
                "Alternative search query:", "Query:", "New search query:", 
                "Try searching for:", "Alternative query:", "I suggest:"
            ]
            
            for prefix in prefixes_to_remove:
                if query.startswith(prefix):
                    query = query[len(prefix):].strip()
            
            # Remove quotes if present
            if query.startswith('"') and query.endswith('"'):
                query = query[1:-1].strip()
            
            # Validate the alternative query
            if query and len(query) >= 3 and query.lower() != first_query.lower():
                return query
            
            # If query is invalid, fall back to alternative method
            return fallback_alternative_query(question, first_query)
            
        except Exception as e:
            if st.session_state.show_debug:
                st.error(f"Error generating alternative query with Transformers: {e}")
                import traceback
                st.error(traceback.format_exc())
            return fallback_alternative_query(question, first_query)
            
    except Exception as e:
        if st.session_state.show_debug:
            st.error(f"Error in Transformers alternative query generation: {e}")
        return fallback_alternative_query(question, first_query)

def fallback_alternative_query(question, first_query):
    """Generate an alternative query when model-based generation fails"""
    
    # Strategy 1: Use terms from question that weren't in first query
    first_query_terms = set(first_query.lower().split())
    question_terms = question.lower().replace("?", "").replace(".", "").split()
    
    # Common stopwords to filter out
    stop_words = set([
        "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for", 
        "with", "by", "about", "against", "between", "into", "through", 
        "during", "before", "after", "above", "below", "from", "up", "down", 
        "is", "are", "am", "was", "were", "be", "been", "being", "have", 
        "has", "had", "having", "do", "does", "did", "doing", "can", "could", 
        "should", "would", "may", "might", "must", "will", "shall"
    ])
    
    # Find terms in question that weren't in the first query
    new_terms = [term for term in question_terms 
                 if term not in first_query_terms 
                 and term not in stop_words
                 and len(term) > 2]
    
    # Strategy 2: If the first query has multiple terms, try using just one main term
    # with different context terms
    if len(first_query_terms) > 1:
        # Find the longest term in the first query (likely most important)
        main_term = max(first_query_terms, key=len)
        
        # Find terms in question that weren't in the first query to combine with main term
        if new_terms:
            # Combine main term with new terms
            return f"{main_term} {' '.join(new_terms[:2])}"
    
    # Strategy 3: If we have new terms, use them
    if new_terms:
        return " ".join(new_terms[:3])  # Use up to 3 new terms
    
    # Strategy 4: Use synonyms for key terms in the first query
    # This is a simplified approach without actually using a thesaurus
    # In a real implementation, you might use WordNet or a similar resource
    first_query_main_terms = [term for term in first_query_terms 
                             if term not in stop_words and len(term) > 3]
    
    if first_query_main_terms:
        # Simplified approach: add "about" or similar qualifier to the query
        main_term = max(first_query_main_terms, key=len)
        return f"about {main_term}"
    
    # Strategy 5: Last resort - make the query more general
    if len(first_query.split()) > 2:
        return " ".join(first_query.split()[:2])  # Just use first two words
    
    # If all else fails, return the original question
    return question

def get_query_from_llamacpp(model, question):
    """Get search query from llama.cpp model"""
    try:
        # Check if llama_cpp is available
        llamacpp_spec = importlib.util.find_spec("llama_cpp")
        if llamacpp_spec is None:
            return question
            
        from llama_cpp import Llama
        
        # Expand user path (~/ to absolute path)
        model_path = os.path.expanduser(model)
        
        # Get parameters from session state
        n_gpu_layers = 0
        n_threads = 4
        
        # Format the prompt for query extraction
        prompt = f"""<s>[INST] <<SYS>>
You are a search assistant. Given a question, your task is to create an effective search query to find relevant documents.
Create a concise and specific search query.
<</SYS>>

Given this question: "{question}"
What would be the best search query to find relevant information? [/INST]

The best search query would be: """
        
        # Initialize model with minimal settings for quick inference
        llm = Llama(
            model_path=model_path,
            n_gpu_layers=n_gpu_layers,
            n_threads=n_threads,
            n_ctx=512,  # Smaller context for faster inference
        )
        
        # Generate with minimal tokens
        output = llm(
            prompt,
            max_tokens=50,
            temperature=0.1,  # Low temperature for more focused output
            stop=["</s>", "\n\n"]  # Stop at end of generation
        )
        
        if "choices" in output and len(output["choices"]) > 0:
            query = output["choices"][0]["text"].strip()
            
            # Strip quotes if present
            if query.startswith('"') and query.endswith('"'):
                query = query[1:-1]
                
            return query if query else question
        
        return question
    except Exception as e:
        if st.session_state.show_debug:
            st.error(f"Error in llama.cpp query extraction: {e}")
        return question
    
def get_alternative_search_query(question, first_query):
    """Generate an alternative search query when first query fails to find results"""
    provider = st.session_state.llm_provider
    model = st.session_state.llm_model
    
    # Different strategies for different providers
    if provider == "Ollama":
        try:
            import requests
            
            # Check if Ollama is running
            try:
                requests.get("http://localhost:11434/api/version", timeout=1)
            except:
                return question  # Fallback to the original question
            
            # Set up the function calling
            tools = [{
                "type": "function",
                "function": SEARCH_TOOL_SPEC
            }]
            
            # Make the request to Ollama with prompt to try a different approach
            url = "http://localhost:11434/api/chat"
            data = {
                "model": model,
                "messages": [
                    {"role": "system", "content": SEARCH_SYSTEM_PROMPT},
                    {"role": "user", "content": f"I tried searching for '{first_query}' but found no results. Please create an alternative search query that might find relevant information for this question: {question}"}
                ],
                "stream": False,
                "tools": tools
            }
            
            response = requests.post(url, json=data, timeout=30)
            response.raise_for_status()
            response_data = response.json()
            
            # Parse the response for function calls
            if "message" in response_data and "tool_calls" in response_data["message"]:
                for tool_call in response_data["message"]["tool_calls"]:
                    if tool_call["function"]["name"] == "search_vector_database":
                        try:
                            arguments = json.loads(tool_call["function"]["arguments"])
                            if "query" in arguments:
                                return arguments["query"]
                        except:
                            pass
            
            # If no function call, check if the model gave a textual search suggestion
            if "message" in response_data and "content" in response_data["message"]:
                content = response_data["message"]["content"]
                # Look for search terms in quotes
                import re
                search_terms = re.findall(r'"([^"]*)"', content)
                if search_terms:
                    return search_terms[0]
            
            # Fallback strategy: Simplify the first query
            words = first_query.split()
            if len(words) >= 3:
                # Just use the most important 1-2 words
                return " ".join(words[:2])
            else:
                # Try a more general version of the query
                return question
        except Exception as e:
            if st.session_state.show_debug:
                st.error(f"Error generating alternative query: {e}")
            return question
    elif provider == "MLX":
        # Use our MLX-specific function
        return get_mlx_alternative_query(model, question, first_query)
    elif provider == "Transformers":
        # Use our Transformers-specific function
        return get_transformers_alternative_query(model, question, first_query)
    elif provider == "llama.cpp":
        # Use a simpler approach for llama.cpp to avoid loading the model again
        # This could be replaced with a full implementation similar to the others
        try:
            # Simplified approach for llama.cpp
            words = question.split()
            # Filter question words and common stopwords
            filtered_words = [w for w in words if w.lower() not in [
                "what", "when", "where", "which", "who", "how", "why", 
                "a", "an", "the", "and", "or", "but", "in", "on", "at", 
                "can", "does", "is", "are", "was", "were"
            ] and len(w) > 3]
            
            # Use words that weren't in the first query
            first_query_words = set(first_query.lower().split())
            new_words = [w for w in filtered_words if w.lower() not in first_query_words]
            
            if new_words:
                return " ".join(new_words[:3])  # Use up to 3 new words
            
            # If no new words, use a generalization approach
            if len(first_query.split()) > 1:
                return first_query.split()[0]  # Use just the first word
            
            return question
        except Exception as e:
            if st.session_state.show_debug:
                st.error(f"Error generating alternative query with llama.cpp approach: {e}")
            return question
    else:
        # Simple fallback for other providers - just use a more general version of the query
        words = question.split()
        important_words = [w for w in words if len(w) > 3 and w.lower() not in ["what", "when", "where", "which", "who", "how", "why", "can", "does", "is", "are", "was", "were"]]
        if important_words:
            return " ".join(important_words[:3])
        return question

# Set page configuration
st.set_page_config(
    page_title="CrispRAG v0.2",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Add the current directory to the path to import the module
sys.path.append('.')

# Try to import from our codebase
try:
    with gpu_lock:
        from utils import (
            TextProcessor, FileUtils, ModelUtils, GeneralUtils, 
            ResultProcessor, SearchAlgorithms, ChunkUtils, TextExtractorUtils,
            HAS_MLX, HAS_MLX_EMBEDDING_MODELS, HAS_TRANSFORMERS, HAS_PYTORCH
        )
        
        try:
            from mlx_utils import MLX_EMBEDDING_REGISTRY, DEFAULT_DENSE_MODEL, DEFAULT_SPARSE_MODEL
        except ImportError:
            MLX_EMBEDDING_REGISTRY = {}
            DEFAULT_DENSE_MODEL = "bge-small"
            DEFAULT_SPARSE_MODEL = "distilbert-splade"
        
        try:
            from fastembed_utils import FASTEMBED_MODELS_REGISTRY, DEFAULT_FASTEMBED_MODEL, DEFAULT_FASTEMBED_SPARSE_MODEL, HAS_FASTEMBED
        except ImportError:
            FASTEMBED_MODELS_REGISTRY = {}
            DEFAULT_FASTEMBED_MODEL = "BAAI/bge-small-en-v1.5"
            DEFAULT_FASTEMBED_SPARSE_MODEL = "prithivida/Splade_PP_en_v1"
            HAS_FASTEMBED = False
        
        try:
            from ollama_utils import OLLAMA_MODELS_REGISTRY, DEFAULT_OLLAMA_EMBED_MODEL, HAS_OLLAMA
        except ImportError:
            OLLAMA_MODELS_REGISTRY = {}
            DEFAULT_OLLAMA_EMBED_MODEL = "nomic-embed-text"
            HAS_OLLAMA = False
            
        from vector_db_interface import DBFactory
    
    MODULE_LOADED = True
except Exception as e:
    st.error(f"Error importing modules: {str(e)}")
    MODULE_LOADED = False

# Define CSS for styling
st.markdown("""
<style>
    /* Result styling */
    .result-container {
        background-color: #f5f7f9;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 15px;
        border: 1px solid #ddd;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .highlight {
        background-color: #ffeb3b;
        padding: 2px;
        border-radius: 3px;
        font-weight: bold;
        color: #000;
    }
    .search-info {
        background-color: #1e1e1e;
        color: #ffffff;
        padding: 15px;
        margin-bottom: 15px;
        border-radius: 5px;
        border: 1px solid #bbdefb;
    }
    .filename {
        font-weight: bold;
        font-size: 16px;
        color: #2196f3;
    }
    .meta-info {
        color: #666;
        font-size: 14px;
        margin-top: 5px;
    }
    .preview-text {
        background-color: #fff;
        color: #000;
        padding: 10px;
        border-radius: 5px;
        border: 1px solid #eee;
        margin-top: 10px;
        white-space: pre-wrap;
        line-height: 1.5;
    }
    
    /* UI improvements */
    .stButton button {
        width: 100%;
    }
    .search-header {
        margin-bottom: 20px;
    }
    .pagination-btn {
        margin: 0 5px;
    }
    .page-info {
        text-align: center;
        margin: 10px 0;
    }
    .centered {
        display: flex;
        justify-content: center;
    }
    
    /* Sidebar customization */
    .sidebar-section {
        margin-bottom: 10px;
        border-bottom: 1px solid #ddd;
        padding-bottom: 10px;
    }
    
    /* Settings block styling */
    .settings-block {
        background-color: #f8f9fa;
        border-radius: 5px;
        padding: 10px;
        margin-bottom: 10px;
        border: 1px solid #eaecef;
    }
    
    /* Custom expandable sections */
    .expandable-section {
        margin-bottom: 10px;
    }
    .expandable-header {
        background-color: #f0f2f6;
        padding: 8px 12px;
        border-radius: 5px;
        margin-bottom: 5px;
        cursor: pointer;
        font-weight: bold;
        display: flex;
        justify-content: space-between;
    }
    .expandable-header:hover {
        background-color: #e6e9ef;
    }
    .expandable-content {
        padding: 5px 15px;
        margin-left: 0;
        margin-right: 0;
        border-left: 2px solid #f0f2f6;
    }
    
    /* Status indicators */
    .status-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 6px;
    }
    .status-green {
        background-color: #4CAF50;
    }
    .status-red {
        background-color: #F44336;
    }
    .status-yellow {
        background-color: #FFC107;
    }
</style>
""", unsafe_allow_html=True)

if not MODULE_LOADED:
    st.error("Failed to load the required modules. Please make sure the MLXrag codebase is available.")
    st.stop()

# We use gpu_lock around anything using MLX, Ollama, FastEmbed, or PyTorch models
def gpu_safe_call(fn, *args, **kwargs):
    with gpu_lock:
        return fn(*args, **kwargs)

# Initialize session state variables
def initialize_session_state():
    if 'db_manager' not in st.session_state:
        st.session_state.db_manager = None
    if 'processor' not in st.session_state:
        st.session_state.processor = None
    if 'search_results' not in st.session_state:
        st.session_state.search_results = None
    if 'search_initialized' not in st.session_state:
        st.session_state.search_initialized = False
    if 'show_debug' not in st.session_state:
        st.session_state.show_debug = False
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 1
    if 'search_limit' not in st.session_state:
        st.session_state.search_limit = 10
    if 'context_size' not in st.session_state:
        st.session_state.context_size = 500
    if 'last_query' not in st.session_state:
        st.session_state.last_query = ""
    if 'total_results' not in st.session_state:
        st.session_state.total_results = 0
    if 'directories' not in st.session_state:
        st.session_state.directories = []
    # Settings sections expanded state
    if 'connection_expanded' not in st.session_state:
        st.session_state.connection_expanded = True
    if 'embedding_expanded' not in st.session_state:
        st.session_state.embedding_expanded = False
    if 'search_settings_expanded' not in st.session_state:
        st.session_state.search_settings_expanded = False
    if 'advanced_expanded' not in st.session_state:
        st.session_state.advanced_expanded = False
    # LLM-specific session state variables
    if 'llm_provider' not in st.session_state:
        st.session_state.llm_provider = "Ollama"
    if 'llm_model' not in st.session_state:
        st.session_state.llm_model = DEFAULT_OLLAMA_MODEL
    if 'llm_system_prompt' not in st.session_state:
        st.session_state.llm_system_prompt = LLM_SYSTEM_PROMPT
    if 'llm_temperature' not in st.session_state:
        st.session_state.llm_temperature = 0.7
    if 'llm_max_tokens' not in st.session_state:
        st.session_state.llm_max_tokens = 1024
    if 'llm_streaming' not in st.session_state:
        st.session_state.llm_streaming = True
    if 'llm_history' not in st.session_state:
        st.session_state.llm_history = []
    if 'llm_response_queue' not in st.session_state:
        st.session_state.llm_response_queue = queue.Queue()
    if 'llm_thread' not in st.session_state:
        st.session_state.llm_thread = None
    if 'llm_is_generating' not in st.session_state:
        st.session_state.llm_is_generating = False
    if 'llm_settings_expanded' not in st.session_state:
        st.session_state.llm_settings_expanded = True
    if 'context_from_search' not in st.session_state:
        st.session_state.context_from_search = True
    if 'context_strategy' not in st.session_state:
        st.session_state.context_strategy = "top_results"
    
    # Add auto-search related state
    if 'auto_search_enabled' not in st.session_state:
        st.session_state.auto_search_enabled = True
    if 'last_auto_search' not in st.session_state:
        st.session_state.last_auto_search = ""
    if 'auto_search_results' not in st.session_state:
        st.session_state.auto_search_results = None
    if 'auto_search_strategy' not in st.session_state:
        st.session_state.auto_search_strategy = "when_needed"
    if 'max_auto_searches' not in st.session_state:
        st.session_state.max_auto_searches = 2
    if 'search_attempts' not in st.session_state:
        st.session_state.search_attempts = 0
    if 'auto_searched_queries' not in st.session_state:
        st.session_state.auto_searched_queries = []
    if 'chat_context' not in st.session_state:
        st.session_state.chat_context = ""


# Expandable section component
def expandable_section(title, key, default=False, icon=""):
    # Determine if this section is expanded
    if key not in st.session_state:
        st.session_state[key] = default
    
    # Create the expandable header
    clicked = st.button(
        f"{icon} {title} {'▼' if st.session_state[key] else '▶'}",
        key=f"btn_{key}",
        help=f"Click to {'collapse' if st.session_state[key] else 'expand'} this section"
    )
    
    # Toggle the state if clicked
    if clicked:
        st.session_state[key] = not st.session_state[key]
        st.rerun()
    
    # Return True if expanded, False if collapsed
    return st.session_state[key]

# Call initialization
initialize_session_state()

# Main app with tabs
tab1, tab2, tab3 = st.tabs(["Index Documents", "Search Documents", "Chat with Documents"])

# Tab 1: Index Documents
with tab1:
    st.title("Document Indexer")
    
    # Initialize database and embedding models on the sidebar
    with st.sidebar:
        st.title("Settings")
        
        # Connection settings
        if expandable_section("Connection Settings", "connection_expanded", True, "🔌"):
            connection_type = st.radio("Connection Type", ["Local Storage", "Remote Server"])
            
            if connection_type == "Local Storage":
                host = "localhost"
                port = 6333
                storage_path = os.path.join(os.getcwd(), "mlxrag_storage")
                st.info(f"Using local storage at: {storage_path}")
            else:
                host = st.text_input("Host", "localhost")
                port = st.number_input("Port", value=6333, min_value=1, max_value=65535)
                storage_path = None
            
            # Database selection
            db_type = st.selectbox(
                "Database Backend", 
                ["qdrant", "milvus", "lancedb", "meilisearch", "elasticsearch", "chromadb"],
                help="Select the vector database backend to use"
            )
            
            # Add specific database settings based on selection
            if db_type == "milvus":
                st.text_input("Milvus User", key="milvus_user", value="")
                st.text_input("Milvus Password", key="milvus_password", value="", type="password")
                st.checkbox("Secure Connection", key="milvus_secure", value=False)
            elif db_type == "lancedb":
                st.text_input("LanceDB URI", key="lancedb_uri", value="")
            elif db_type == "meilisearch":
                st.text_input("Meilisearch URL", key="meilisearch_url", value="http://localhost:7700")
                st.text_input("Meilisearch API Key", key="meilisearch_api_key", value="", type="password")
            elif db_type == "elasticsearch":
                es_hosts = st.text_input("Elasticsearch Hosts (comma-separated)", key="es_hosts", value="http://localhost:9200")
                st.text_input("Elasticsearch API Key", key="es_api_key", value="", type="password")
                st.text_input("Elasticsearch Username", key="es_username", value="")
                st.text_input("Elasticsearch Password", key="es_password", value="", type="password")
            
            # Collection name
            collection_name = st.text_input("Collection Name", "documents")
            recreate_collection = st.checkbox("Recreate Collection if Exists")
            
        # Embedding model settings
        if expandable_section("Embedding Settings", "embedding_expanded", False, "🧠"):
            # Embedding provider selection
            embedding_provider = st.radio(
                "Embedding Provider",
                ["MLX", "Ollama", "FastEmbed"], 
                help="Select the embedding provider to use"
            )
            
            if embedding_provider == "MLX":
                if HAS_MLX_EMBEDDING_MODELS:
                    # Get available models from registry
                    dense_models = [k for k, v in MLX_EMBEDDING_REGISTRY.items() if not v.get("lm_head")]
                    sparse_models = [k for k, v in MLX_EMBEDDING_REGISTRY.items() if v.get("lm_head")]
                    
                    if dense_models:
                        dense_model = st.selectbox(
                            "Dense Model", 
                            dense_models,
                            index=dense_models.index(DEFAULT_DENSE_MODEL) if DEFAULT_DENSE_MODEL in dense_models else 0
                        )
                    else:
                        dense_model = st.text_input("Dense Model", DEFAULT_DENSE_MODEL)
                    
                    if sparse_models:
                        sparse_model = st.selectbox(
                            "Sparse Model",
                            sparse_models,
                            index=sparse_models.index(DEFAULT_SPARSE_MODEL) if DEFAULT_SPARSE_MODEL in sparse_models else 0
                        )
                    else:
                        sparse_model = st.text_input("Sparse Model", DEFAULT_SPARSE_MODEL)
                    
                    custom_repo_id = st.text_input(
                        "Custom Repo ID", 
                        "",
                        help="Optional: Specify a custom HuggingFace repository"
                    )
                    
                    # Advanced MLX settings in expander
                    with st.expander("Advanced MLX Settings"):
                        top_k = st.number_input("Top-k tokens (SPLADE)", value=64, min_value=1, max_value=512)
                        custom_ndim = st.number_input("Custom Dimension", value=0, min_value=0, max_value=4096)
                        custom_pooling = st.selectbox("Pooling Method", ["mean", "first", "max"])
                        custom_normalize = st.checkbox("Normalize Embeddings", value=True)
                        custom_max_length = st.number_input("Max Sequence Length", value=512, min_value=64, max_value=2048)
                else:
                    st.warning("MLX Embedding Models not available. You need to install mlx-embedding-models.")
                    dense_model = DEFAULT_DENSE_MODEL
                    sparse_model = DEFAULT_SPARSE_MODEL
                    custom_repo_id = ""
                    top_k = 64
            
            elif embedding_provider == "Ollama":
                if HAS_OLLAMA:
                    if OLLAMA_MODELS_REGISTRY:
                        ollama_models = list(OLLAMA_MODELS_REGISTRY.keys())
                        ollama_model = st.selectbox(
                            "Ollama Model",
                            ollama_models,
                            index=ollama_models.index(DEFAULT_OLLAMA_EMBED_MODEL) if DEFAULT_OLLAMA_EMBED_MODEL in ollama_models else 0
                        )
                    else:
                        ollama_model = st.text_input("Ollama Model", DEFAULT_OLLAMA_EMBED_MODEL)
                    
                    ollama_host = st.text_input("Ollama Host", "http://localhost:11434")
                else:
                    st.warning("Ollama not available. Make sure Ollama is installed and running.")
                    ollama_model = DEFAULT_OLLAMA_EMBED_MODEL
                    ollama_host = "http://localhost:11434"
            
            elif embedding_provider == "FastEmbed":
                if HAS_FASTEMBED:
                    if FASTEMBED_MODELS_REGISTRY:
                        # Filter models by type
                        fastembed_dense_models = [k for k, v in FASTEMBED_MODELS_REGISTRY.items() 
                                                if v.get("ndim", 0) > 0]
                        fastembed_sparse_models = [k for k, v in FASTEMBED_MODELS_REGISTRY.items() 
                                                if v.get("ndim", 1) == 0]
                        
                        if fastembed_dense_models:
                            fastembed_model = st.selectbox(
                                "FastEmbed Dense Model",
                                fastembed_dense_models,
                                index=fastembed_dense_models.index(DEFAULT_FASTEMBED_MODEL) if DEFAULT_FASTEMBED_MODEL in fastembed_dense_models else 0
                            )
                        else:
                            fastembed_model = st.text_input("FastEmbed Dense Model", DEFAULT_FASTEMBED_MODEL)
                        
                        if fastembed_sparse_models:
                            fastembed_sparse_model = st.selectbox(
                                "FastEmbed Sparse Model",
                                fastembed_sparse_models,
                                index=fastembed_sparse_models.index(DEFAULT_FASTEMBED_SPARSE_MODEL) if DEFAULT_FASTEMBED_SPARSE_MODEL in fastembed_sparse_models else 0
                            )
                        else:
                            fastembed_sparse_model = st.text_input("FastEmbed Sparse Model", DEFAULT_FASTEMBED_SPARSE_MODEL)
                    else:
                        fastembed_model = st.text_input("FastEmbed Dense Model", DEFAULT_FASTEMBED_MODEL)
                        fastembed_sparse_model = st.text_input("FastEmbed Sparse Model", DEFAULT_FASTEMBED_SPARSE_MODEL)
                        
                    fastembed_use_gpu = st.checkbox("Use GPU (requires fastembed-gpu)", value=False)
                    fastembed_cache_dir = st.text_input("Model Cache Directory", "")
                else:
                    st.warning("FastEmbed not available. You need to install fastembed.")
                    fastembed_model = DEFAULT_FASTEMBED_MODEL
                    fastembed_sparse_model = DEFAULT_FASTEMBED_SPARSE_MODEL
                    fastembed_use_gpu = False
                    fastembed_cache_dir = ""
        
        # Advanced settings
        if expandable_section("Advanced Settings", "advanced_expanded", False, "⚙️"):
            st.session_state.show_debug = st.checkbox("Show Debug Info", False)
            max_docs = st.number_input("Max Documents to Index", min_value=1, max_value=10000, value=100)
            
            # Storage management
            if connection_type == "Local Storage":
                if st.button("Clear Storage"):
                    try:
                        # Create directory path
                        collections_dir = os.path.join(storage_path, "collections")
                        if os.path.exists(collections_dir):
                            shutil.rmtree(collections_dir)
                            os.makedirs(collections_dir, exist_ok=True)
                            st.success("Storage cleared successfully!")
                        else:
                            os.makedirs(collections_dir, exist_ok=True)
                            st.info("Created storage directory.")
                    except Exception as e:
                        st.error(f"Error clearing storage: {e}")
                
                if st.button("Force Close Database Connection"):
                    try:
                        if 'db_manager' in st.session_state and st.session_state.db_manager:
                            if hasattr(st.session_state.db_manager, 'client') and hasattr(st.session_state.db_manager.client, 'close'):
                                st.session_state.db_manager.client.close()
                            del st.session_state.db_manager
                            st.session_state.db_manager = None
                            st.success("Database connection closed and removed from session.")
                            st.session_state.search_initialized = False
                    except Exception as e:
                        st.error(f"Failed to close database connection: {e}")
    
    # Check connection button
    if st.button("Check Connection"):
        try:
            # Initialize database manager for connection check
            db_args = {
                "collection_name": collection_name,
                "storage_path": storage_path if connection_type == "Local Storage" else None,
                "verbose": st.session_state.show_debug
            }
            
            # Add connection-specific parameters
            if db_type == "qdrant":
                db_args.update({
                    "host": host,
                    "port": port
                })
            elif db_type == "milvus":
                db_args.update({
                    "host": host,
                    "port": port,
                    "user": st.session_state.get("milvus_user", ""),
                    "password": st.session_state.get("milvus_password", ""),
                    "secure": st.session_state.get("milvus_secure", False)
                })
            elif db_type == "lancedb":
                db_args.update({
                    "uri": st.session_state.get("lancedb_uri", "")
                })
            elif db_type == "meilisearch":
                db_args.update({
                    "url": st.session_state.get("meilisearch_url", "http://localhost:7700"),
                    "api_key": st.session_state.get("meilisearch_api_key", "")
                })
            elif db_type == "elasticsearch":
                es_hosts_list = [h.strip() for h in st.session_state.get("es_hosts", "http://localhost:9200").split(",")]
                db_args.update({
                    "hosts": es_hosts_list,
                    "api_key": st.session_state.get("es_api_key", ""),
                    "username": st.session_state.get("es_username", ""),
                    "password": st.session_state.get("es_password", "")
                })
            
            # Create database manager
            db_manager = gpu_safe_call(DBFactory.create_db, db_type, **db_args)
            
            # Close previous connection if exists
            if 'db_manager' in st.session_state and st.session_state.db_manager:
                try:
                    if hasattr(st.session_state.db_manager, 'client') and hasattr(st.session_state.db_manager.client, 'close'):
                        st.session_state.db_manager.client.close()
                    del st.session_state.db_manager
                    st.session_state.db_manager = None
                    st.success("Closed previous database connection.")
                except Exception as e:
                    st.warning(f"Failed to close previous connection: {e}")
            
            # Store the new connection
            st.session_state.db_manager = db_manager
            
            # Get collection info
            collection_info = db_manager.get_collection_info()
            
            # Check if collection exists
            if isinstance(collection_info, dict) and "error" not in collection_info:
                st.success(f"✅ Connected to {db_type.capitalize()} successfully!")
                
                # Show collection information if available
                if "points_count" in collection_info:
                    points_count = collection_info["points_count"]
                    if points_count > 0:
                        st.info(f"Collection '{collection_name}' has {points_count} points.")
                    else:
                        st.info(f"Collection '{collection_name}' exists but is empty.")
                elif isinstance(collection_info, dict) and collection_info:
                    st.info(f"Collection information: {collection_info}")
                else:
                    st.info(f"Collection '{collection_name}' will be created during indexing.")
            else:
                error_msg = collection_info.get("error", "Unknown error") if isinstance(collection_info, dict) else "Unknown error"
                st.warning(f"Connected but couldn't get collection info: {error_msg}")
        except Exception as e:
            st.error(f"Error checking connection: {str(e)}")
            if st.session_state.show_debug:
                import traceback
                st.error(traceback.format_exc())
    
    # File upload option
    upload_option = st.radio("Upload Method", ["Upload Files", "Specify Directory"])
    
    if upload_option == "Upload Files":
        uploaded_files = st.file_uploader("Upload documents", 
                                        accept_multiple_files=True, 
                                        type=["txt", "md", "pdf", "json", "csv", "html"])
        
        if uploaded_files:
            st.info(f"Uploaded {len(uploaded_files)} files.")
            
            # Create a temp directory for uploaded files if needed
            if 'upload_dir' not in st.session_state:
                st.session_state.upload_dir = os.path.join("mlxrag_uploads")
                os.makedirs(st.session_state.upload_dir, exist_ok=True)
            
            # Button to index uploaded files
            if st.button("Index Uploaded Files"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Save uploaded files to temp directory
                saved_files = []
                for i, file in enumerate(uploaded_files):
                    progress = (i + 1) / (len(uploaded_files) * 2)  # First half for saving
                    progress_bar.progress(progress)
                    status_text.text(f"Saving file {i+1}/{len(uploaded_files)}: {file.name}")
                    
                    # Save the file
                    file_path = os.path.join(st.session_state.upload_dir, file.name)
                    with open(file_path, "wb") as f:
                        f.write(file.getbuffer())
                    saved_files.append(file_path)
                
                # Initialize database and processor
                try:
                    # Close previous connections if any
                    if 'db_manager' in st.session_state and st.session_state.db_manager:
                        try:
                            if hasattr(st.session_state.db_manager, 'client') and hasattr(st.session_state.db_manager.client, 'close'):
                                st.session_state.db_manager.client.close()
                            del st.session_state.db_manager
                            st.session_state.db_manager = None
                        except Exception as e:
                            st.warning(f"Failed to close previous connection: {e}")
                    
                    # Prepare database arguments
                    db_args = {
                        "collection_name": collection_name,
                        "storage_path": storage_path if connection_type == "Local Storage" else None,
                        "verbose": st.session_state.show_debug
                    }
                    
                    # Add database-specific arguments based on type
                    if db_type == "qdrant":
                        db_args.update({
                            "host": host,
                            "port": port
                        })
                    elif db_type == "milvus":
                        db_args.update({
                            "host": host,
                            "port": port,
                            "user": st.session_state.get("milvus_user", ""),
                            "password": st.session_state.get("milvus_password", ""),
                            "secure": st.session_state.get("milvus_secure", False)
                        })
                    elif db_type == "lancedb":
                        db_args.update({
                            "uri": st.session_state.get("lancedb_uri", "")
                        })
                    elif db_type == "meilisearch":
                        db_args.update({
                            "url": st.session_state.get("meilisearch_url", "http://localhost:7700"),
                            "api_key": st.session_state.get("meilisearch_api_key", "")
                        })
                    elif db_type == "elasticsearch":
                        es_hosts_list = [h.strip() for h in st.session_state.get("es_hosts", "http://localhost:9200").split(",")]
                        db_args.update({
                            "hosts": es_hosts_list,
                            "api_key": st.session_state.get("es_api_key", ""),
                            "username": st.session_state.get("es_username", ""),
                            "password": st.session_state.get("es_password", "")
                        })
                    
                    # Create database manager
                    db_manager = gpu_safe_call(DBFactory.create_db, db_type, **db_args)
                    
                    # Create collection
                    db_manager.create_collection(recreate=recreate_collection)
                    
                    # Initialize processor based on embedding provider
                    processor_args = {
                        "model_name": "none",
                        "weights_path": None,
                        "verbose": st.session_state.show_debug
                    }
                    
                    if embedding_provider == "MLX":
                        processor_args.update({
                            "use_mlx_embedding": True,
                            "dense_model": dense_model,
                            "sparse_model": sparse_model,
                            "top_k": top_k,
                            "custom_repo_id": custom_repo_id if custom_repo_id else None,
                            "custom_ndim": custom_ndim if custom_ndim > 0 else None,
                            "custom_pooling": custom_pooling,
                            "custom_normalize": custom_normalize,
                            "custom_max_length": custom_max_length
                        })
                    elif embedding_provider == "Ollama":
                        processor_args.update({
                            "use_ollama": True,
                            "ollama_model": ollama_model,
                            "ollama_host": ollama_host
                        })
                    elif embedding_provider == "FastEmbed":
                        processor_args.update({
                            "use_fastembed": True,
                            "fastembed_model": fastembed_model,
                            "fastembed_sparse_model": fastembed_sparse_model,
                            "fastembed_use_gpu": fastembed_use_gpu,
                            "fastembed_cache_dir": fastembed_cache_dir if fastembed_cache_dir else None
                        })
                    
                    processor = gpu_safe_call(TextProcessor, **processor_args)
                    
                    # Get vector dimension from processor
                    vector_dim = processor.vector_size
                    
                    # Update database with vector size
                    if hasattr(db_manager, 'update_vector_size'):
                        db_manager.update_vector_size(vector_dim)
                    
                    # Process and index each file
                    total_chunks = 0
                    successful_files = 0
                    
                    for i, file_path in enumerate(saved_files):
                        progress = 0.5 + (i + 1) / (len(saved_files) * 2)  # Second half for indexing
                        progress_bar.progress(progress)
                        status_text.text(f"Indexing file {i+1}/{len(saved_files)}: {os.path.basename(file_path)}")
                        
                        try:
                            # Choose processing method (sparse embeddings if available)
                            has_sparse = (embedding_provider == "MLX" and HAS_MLX_EMBEDDING_MODELS) or \
                                        (embedding_provider == "FastEmbed" and HAS_FASTEMBED) or \
                                        (embedding_provider == "Ollama" and HAS_OLLAMA)
                                        
                            if has_sparse:
                                results = gpu_safe_call(processor.process_file_with_sparse, file_path)
                                if results:
                                    gpu_safe_call(db_manager.insert_embeddings_with_sparse, results)
                                    total_chunks += len(results)
                                    successful_files += 1
                            else:
                                results = gpu_safe_call(processor.process_file, file_path)
                                if results:
                                    gpu_safe_call(db_manager.insert_embeddings, results)
                                    total_chunks += len(results)
                                    successful_files += 1
                                    
                        except Exception as e:
                            st.error(f"Error processing {file_path}: {str(e)}")
                            if st.session_state.show_debug:
                                import traceback
                                st.error(traceback.format_exc())
                            continue
                    
                    # Update session state
                    st.session_state.db_manager = db_manager
                    st.session_state.processor = processor
                    st.session_state.search_initialized = True
                    
                    # Show success message
                    progress_bar.progress(1.0)
                    status_text.text("Indexing complete!")
                    st.success(f"Successfully indexed {successful_files}/{len(saved_files)} files with {total_chunks} chunks.")
                    
                except Exception as e:
                    st.error(f"Error during indexing: {str(e)}")
                    if st.session_state.show_debug:
                        import traceback
                        st.error(traceback.format_exc())
    
    else:  # Directory indexing
        # Max document limit
        max_docs = st.number_input("Max Number of Documents to Index", min_value=1, max_value=10000, value=100)

        st.markdown("### Add Directory to Index")
        new_dir = st.text_input("Directory Path", key="add_dir_input")

        col1, col2 = st.columns([5, 1])
        with col1:
            if st.button("Add Directory"):
                if new_dir and os.path.isdir(new_dir):
                    if new_dir not in st.session_state.directories:
                        st.session_state.directories.append(new_dir)
                elif new_dir:
                    st.warning("Directory does not exist.")
        with col2:
            st.write("")

        # Display added directories
        if st.session_state.directories:
            st.markdown("### Selected Directories:")
            for i, d in enumerate(st.session_state.directories):
                col1, col2 = st.columns([10, 1])
                with col1:
                    st.write(f"{i+1}. `{d}`")
                with col2:
                    if st.button("❌", key=f"remove_dir_{i}"):
                        st.session_state.directories.pop(i)
                        st.rerun()
        else:
            st.info("No directories added yet.")

        include_patterns = st.text_input("Include Patterns (space separated)", "*.txt *.pdf *.md *.csv *.json")

        if st.button("Index Selected Directories"):
            if not st.session_state.directories:
                st.error("Please add at least one valid directory.")
            else:
                progress_bar = st.progress(0)
                status_text = st.empty()

                try:
                    include_patterns_list = include_patterns.split() if include_patterns else None
                    all_files = []

                    # Get files from all directories
                    for dir_path in st.session_state.directories:
                        if os.path.isdir(dir_path):
                            files = FileUtils.get_files_to_process(
                                dir_path,
                                include_patterns=include_patterns_list,
                                limit=None,
                                verbose=st.session_state.show_debug
                            )
                            all_files.extend(files)

                    if not all_files:
                        st.warning("No matching files found in the selected directories.")
                    else:
                        # Truncate if over max_docs
                        all_files = all_files[:max_docs]
                        status_text.text(f"Found {len(all_files)} files to process.")

                        # Close previous connections if any
                        if 'db_manager' in st.session_state and st.session_state.db_manager:
                            try:
                                if hasattr(st.session_state.db_manager, 'client') and hasattr(st.session_state.db_manager.client, 'close'):
                                    st.session_state.db_manager.client.close()
                                del st.session_state.db_manager
                                st.session_state.db_manager = None
                            except Exception as e:
                                st.warning(f"Failed to close previous connection: {e}")
                        
                        # Prepare database arguments
                        db_args = {
                            "collection_name": collection_name,
                            "storage_path": storage_path if connection_type == "Local Storage" else None,
                            "verbose": st.session_state.show_debug
                        }
                        
                        # Add database-specific arguments based on type
                        if db_type == "qdrant":
                            db_args.update({
                                "host": host,
                                "port": port
                            })
                        elif db_type == "milvus":
                            db_args.update({
                                "host": host,
                                "port": port,
                                "user": st.session_state.get("milvus_user", ""),
                                "password": st.session_state.get("milvus_password", ""),
                                "secure": st.session_state.get("milvus_secure", False)
                            })
                        elif db_type == "lancedb":
                            db_args.update({
                                "uri": st.session_state.get("lancedb_uri", "")
                            })
                        elif db_type == "meilisearch":
                            db_args.update({
                                "url": st.session_state.get("meilisearch_url", "http://localhost:7700"),
                                "api_key": st.session_state.get("meilisearch_api_key", "")
                            })
                        elif db_type == "elasticsearch":
                            es_hosts_list = [h.strip() for h in st.session_state.get("es_hosts", "http://localhost:9200").split(",")]
                            db_args.update({
                                "hosts": es_hosts_list,
                                "api_key": st.session_state.get("es_api_key", ""),
                                "username": st.session_state.get("es_username", ""),
                                "password": st.session_state.get("es_password", "")
                            })
                        
                        # Create database manager
                        db_manager = gpu_safe_call(DBFactory.create_db, db_type, **db_args)
                        
                        # Create collection
                        db_manager.create_collection(recreate=recreate_collection)
                        
                        # Initialize processor based on embedding provider
                        processor_args = {
                            "model_name": "none",
                            "weights_path": None,
                            "verbose": st.session_state.show_debug
                        }
                        
                        if embedding_provider == "MLX":
                            processor_args.update({
                                "use_mlx_embedding": True,
                                "dense_model": dense_model,
                                "sparse_model": sparse_model,
                                "top_k": top_k,
                                "custom_repo_id": custom_repo_id if custom_repo_id else None,
                                "custom_ndim": custom_ndim if custom_ndim > 0 else None,
                                "custom_pooling": custom_pooling,
                                "custom_normalize": custom_normalize,
                                "custom_max_length": custom_max_length
                            })
                        elif embedding_provider == "Ollama":
                            processor_args.update({
                                "use_ollama": True,
                                "ollama_model": ollama_model,
                                "ollama_host": ollama_host
                            })
                        elif embedding_provider == "FastEmbed":
                            processor_args.update({
                                "use_fastembed": True,
                                "fastembed_model": fastembed_model,
                                "fastembed_sparse_model": fastembed_sparse_model,
                                "fastembed_use_gpu": fastembed_use_gpu,
                                "fastembed_cache_dir": fastembed_cache_dir if fastembed_cache_dir else None
                            })
                        
                        processor = gpu_safe_call(TextProcessor, **processor_args)
                        
                        # Update vector size in database manager if needed
                        vector_dim = processor.vector_size
                        if hasattr(db_manager, 'update_vector_size'):
                            db_manager.update_vector_size(vector_dim)
                        
                        # Process and index each file
                        total_chunks = 0
                        successful_files = 0
                        
                        for i, file_path in enumerate(all_files):
                            progress = (i + 1) / len(all_files)
                            progress_bar.progress(progress)
                            status_text.text(f"Indexing file {i+1}/{len(all_files)}: {os.path.basename(file_path)}")
                            
                            try:
                                # Choose processing method (sparse embeddings if available)
                                has_sparse = (embedding_provider == "MLX" and HAS_MLX_EMBEDDING_MODELS) or \
                                            (embedding_provider == "FastEmbed" and HAS_FASTEMBED) or \
                                            (embedding_provider == "Ollama" and HAS_OLLAMA)
                                            
                                if has_sparse:
                                    results = gpu_safe_call(processor.process_file_with_sparse, file_path)
                                    if results:
                                        gpu_safe_call(db_manager.insert_embeddings_with_sparse, results)
                                        total_chunks += len(results)
                                        successful_files += 1
                                else:
                                    results = gpu_safe_call(processor.process_file, file_path)
                                    if results:
                                        gpu_safe_call(db_manager.insert_embeddings, results)
                                        total_chunks += len(results)
                                        successful_files += 1
                            except Exception as e:
                                st.error(f"Error processing {file_path}: {str(e)}")
                                if st.session_state.show_debug:
                                    import traceback
                                    st.error(traceback.format_exc())
                                continue
                        
                        # Update session state
                        st.session_state.db_manager = db_manager
                        st.session_state.processor = processor
                        st.session_state.search_initialized = True
                        
                        # Show success message
                        progress_bar.progress(1.0)
                        status_text.text("Indexing complete!")
                        st.success(f"Successfully indexed {successful_files}/{len(all_files)} files with {total_chunks} chunks.")
                
                except Exception as e:
                    st.error(f"Error during indexing: {str(e)}")
                    if st.session_state.show_debug:
                        import traceback
                        st.error(traceback.format_exc())

# Tab 2: Search
with tab2:
    st.title("Document Search")
    
    # Search settings in sidebar
    with st.sidebar:
        st.title("Search Settings")
        
        # Search configuration with expandable sections
        if expandable_section("Connection Settings", "connection_expanded", True, "🔌"):
            connection_type = st.radio("Connection Type", ["Local Storage", "Remote Server"])
            
            if connection_type == "Local Storage":
                host = "localhost"
                port = 6333
                storage_path = os.path.join(os.getcwd(), "mlxrag_storage")
                st.info(f"Using local storage at: {storage_path}")
            else:
                host = st.text_input("Host", "localhost")
                port = st.number_input("Port", value=6333, min_value=1, max_value=65535)
                storage_path = None
            
            # Database selection
            db_type = st.selectbox(
                "Database Backend", 
                ["qdrant", "milvus", "lancedb", "meilisearch", "elasticsearch", "chromadb"],
                help="Select the vector database backend to use"
            )
            
            # Add specific database settings based on selection
            if db_type == "milvus":
                st.text_input("Milvus User", key="milvus_user", value="")
                st.text_input("Milvus Password", key="milvus_password", value="", type="password")
                st.checkbox("Secure Connection", key="milvus_secure", value=False)
            elif db_type == "lancedb":
                st.text_input("LanceDB URI", key="lancedb_uri", value="")
            elif db_type == "meilisearch":
                st.text_input("Meilisearch URL", key="meilisearch_url", value="http://localhost:7700")
                st.text_input("Meilisearch API Key", key="meilisearch_api_key", value="", type="password")
            elif db_type == "elasticsearch":
                es_hosts = st.text_input("Elasticsearch Hosts (comma-separated)", key="es_hosts", value="http://localhost:9200")
                st.text_input("Elasticsearch API Key", key="es_api_key", value="", type="password")
                st.text_input("Elasticsearch Username", key="es_username", value="")
                st.text_input("Elasticsearch Password", key="es_password", value="", type="password")
            
            # Collection name
            collection_name = st.text_input("Collection Name", "documents")
        
        # Embedding provider settings
        if expandable_section("Embedding Settings", "embedding_expanded", False, "🧠"):
            # Embedding provider selection
            embedding_provider = st.radio(
                "Embedding Provider",
                ["MLX", "Ollama", "FastEmbed"], 
                help="Select the embedding provider to use"
            )
            
            if embedding_provider == "MLX":
                if HAS_MLX_EMBEDDING_MODELS:
                    # Get available models from registry
                    dense_models = [k for k, v in MLX_EMBEDDING_REGISTRY.items() if not v.get("lm_head")]
                    sparse_models = [k for k, v in MLX_EMBEDDING_REGISTRY.items() if v.get("lm_head")]
                    
                    if dense_models:
                        dense_model = st.selectbox(
                            "Dense Model", 
                            dense_models,
                            index=dense_models.index(DEFAULT_DENSE_MODEL) if DEFAULT_DENSE_MODEL in dense_models else 0
                        )
                    else:
                        dense_model = st.text_input("Dense Model", DEFAULT_DENSE_MODEL)
                    
                    if sparse_models:
                        sparse_model = st.selectbox(
                            "Sparse Model",
                            sparse_models,
                            index=sparse_models.index(DEFAULT_SPARSE_MODEL) if DEFAULT_SPARSE_MODEL in sparse_models else 0
                        )
                    else:
                        sparse_model = st.text_input("Sparse Model", DEFAULT_SPARSE_MODEL)
                    
                    custom_repo_id = st.text_input(
                        "Custom Repo ID", 
                        "",
                        help="Optional: Specify a custom HuggingFace repository"
                    )
                else:
                    st.warning("MLX Embedding Models not available.")
                    dense_model = DEFAULT_DENSE_MODEL
                    sparse_model = DEFAULT_SPARSE_MODEL
                    custom_repo_id = ""
            
            elif embedding_provider == "Ollama":
                if HAS_OLLAMA:
                    if OLLAMA_MODELS_REGISTRY:
                        ollama_models = list(OLLAMA_MODELS_REGISTRY.keys())
                        ollama_model = st.selectbox(
                            "Ollama Model",
                            ollama_models,
                            index=ollama_models.index(DEFAULT_OLLAMA_EMBED_MODEL) if DEFAULT_OLLAMA_EMBED_MODEL in ollama_models else 0
                        )
                    else:
                        ollama_model = st.text_input("Ollama Model", DEFAULT_OLLAMA_EMBED_MODEL)
                    
                    ollama_host = st.text_input("Ollama Host", "http://localhost:11434")
                else:
                    st.warning("Ollama not available.")
                    ollama_model = DEFAULT_OLLAMA_EMBED_MODEL
                    ollama_host = "http://localhost:11434"
            
            elif embedding_provider == "FastEmbed":
                if HAS_FASTEMBED:
                    if FASTEMBED_MODELS_REGISTRY:
                        # Filter models by type
                        fastembed_dense_models = [k for k, v in FASTEMBED_MODELS_REGISTRY.items() 
                                                if v.get("ndim", 0) > 0]
                        fastembed_sparse_models = [k for k, v in FASTEMBED_MODELS_REGISTRY.items() 
                                                if v.get("ndim", 1) == 0]
                        
                        if fastembed_dense_models:
                            fastembed_model = st.selectbox(
                                "FastEmbed Dense Model",
                                fastembed_dense_models,
                                index=fastembed_dense_models.index(DEFAULT_FASTEMBED_MODEL) if DEFAULT_FASTEMBED_MODEL in fastembed_dense_models else 0
                            )
                        else:
                            fastembed_model = st.text_input("FastEmbed Dense Model", DEFAULT_FASTEMBED_MODEL)
                        
                        if fastembed_sparse_models:
                            fastembed_sparse_model = st.selectbox(
                                "FastEmbed Sparse Model",
                                fastembed_sparse_models,
                                index=fastembed_sparse_models.index(DEFAULT_FASTEMBED_SPARSE_MODEL) if DEFAULT_FASTEMBED_SPARSE_MODEL in fastembed_sparse_models else 0
                            )
                        else:
                            fastembed_sparse_model = st.text_input("FastEmbed Sparse Model", DEFAULT_FASTEMBED_SPARSE_MODEL)
                    else:
                        fastembed_model = st.text_input("FastEmbed Dense Model", DEFAULT_FASTEMBED_MODEL)
                        fastembed_sparse_model = st.text_input("FastEmbed Sparse Model", DEFAULT_FASTEMBED_SPARSE_MODEL)
                else:
                    st.warning("FastEmbed not available.")
                    fastembed_model = DEFAULT_FASTEMBED_MODEL
                    fastembed_sparse_model = DEFAULT_FASTEMBED_SPARSE_MODEL
        
        # Search settings section
        if expandable_section("Search Settings", "search_settings_expanded", True, "🔍"):
            st.session_state.search_limit = st.number_input("Results Per Page", 5, 50, 10)
            max_results = st.number_input("Max Total Results", 10, 1000, 100)
            st.session_state.context_size = st.number_input("Context Size", 200, 2000, 500)
            prefetch_limit = st.number_input("Prefetch Limit for Hybrid Search", 10, 200, 50)
            fusion_type = st.selectbox("Fusion Strategy", ["rrf", "dbsf", "linear"])
            rerank = st.checkbox("Apply Reranking", True)
            reranker_type = st.selectbox("Reranker Type", ["cross", "colbert", "cohere", "jina", "rrf", "linear"], 0)
            sort_by_score = st.checkbox("Sort Results by Score", value=True)
            score_threshold = st.slider("Score Threshold", 0.0, 1.0, 0.0, 0.01)
        
        # Advanced settings
        if expandable_section("Advanced Settings", "advanced_expanded", False, "⚙️"):
            st.session_state.show_debug = st.checkbox("Show Debug Info", False)
    
    # Initialize search connection UI
    if not st.session_state.search_initialized:
        st.info("No active search connection. Please initialize search to continue.")
        
        if st.button("Initialize Search"):
            try:
                # Close previous connection if exists
                if 'db_manager' in st.session_state and st.session_state.db_manager:
                    try:
                        if hasattr(st.session_state.db_manager, 'client') and hasattr(st.session_state.db_manager.client, 'close'):
                            st.session_state.db_manager.client.close()
                        del st.session_state.db_manager
                        st.session_state.db_manager = None
                        st.success("Closed previous connection.")
                    except Exception as e:
                        st.warning(f"Failed to close previous connection: {e}")
                
                # Prepare database arguments
                db_args = {
                    "collection_name": collection_name,
                    "storage_path": storage_path if connection_type == "Local Storage" else None,
                    "verbose": st.session_state.show_debug
                }
                
                # Add database-specific arguments based on type
                if db_type == "qdrant":
                    db_args.update({
                        "host": host,
                        "port": port
                    })
                elif db_type == "milvus":
                    db_args.update({
                        "host": host,
                        "port": port,
                        "user": st.session_state.get("milvus_user", ""),
                        "password": st.session_state.get("milvus_password", ""),
                        "secure": st.session_state.get("milvus_secure", False)
                    })
                elif db_type == "lancedb":
                    db_args.update({
                        "uri": st.session_state.get("lancedb_uri", "")
                    })
                elif db_type == "meilisearch":
                    db_args.update({
                        "url": st.session_state.get("meilisearch_url", "http://localhost:7700"),
                        "api_key": st.session_state.get("meilisearch_api_key", "")
                    })
                elif db_type == "elasticsearch":
                    es_hosts_list = [h.strip() for h in st.session_state.get("es_hosts", "http://localhost:9200").split(",")]
                    db_args.update({
                        "hosts": es_hosts_list,
                        "api_key": st.session_state.get("es_api_key", ""),
                        "username": st.session_state.get("es_username", ""),
                        "password": st.session_state.get("es_password", "")
                    })
                
                # Create database manager
                db_manager = gpu_safe_call(DBFactory.create_db, db_type, **db_args)
                
                # Initialize processor based on embedding provider
                processor_args = {
                    "model_name": "none",
                    "weights_path": None,
                    "verbose": st.session_state.show_debug
                }
                
                if embedding_provider == "MLX":
                    processor_args.update({
                        "use_mlx_embedding": True,
                        "dense_model": dense_model,
                        "sparse_model": sparse_model,
                        "custom_repo_id": custom_repo_id if custom_repo_id else None
                    })
                elif embedding_provider == "Ollama":
                    processor_args.update({
                        "use_ollama": True,
                        "ollama_model": ollama_model,
                        "ollama_host": ollama_host
                    })
                elif embedding_provider == "FastEmbed":
                    processor_args.update({
                        "use_fastembed": True,
                        "fastembed_model": fastembed_model,
                        "fastembed_sparse_model": fastembed_sparse_model
                    })
                
                processor = gpu_safe_call(TextProcessor, **processor_args)
                
                # Update vector size in database manager if needed
                vector_dim = processor.vector_size
                if hasattr(db_manager, 'update_vector_size'):
                    db_manager.update_vector_size(vector_dim)
                
                # Update session state
                st.session_state.db_manager = db_manager
                st.session_state.processor = processor
                st.session_state.search_initialized = True
                
                # Get collection info
                try:
                    collection_info = db_manager.get_collection_info()
                    if isinstance(collection_info, dict) and "error" not in collection_info:
                        if "points_count" in collection_info:
                            points_count = collection_info["points_count"]
                            if points_count > 0:
                                st.success(f"✅ Connected to collection '{collection_name}' with {points_count} points.")
                            else:
                                st.warning(f"Connected to collection '{collection_name}', but it's empty. Please index documents.")
                        else:
                            st.success(f"✅ Connected to {db_type.capitalize()} successfully!")
                    else:
                        error_msg = collection_info.get("error", "Unknown error") if isinstance(collection_info, dict) else "Unknown error"
                        st.warning(f"Connected but couldn't get collection info: {error_msg}")
                except Exception as e:
                    st.warning(f"Connected but couldn't get collection info: {e}")
                
                # Force refresh to update UI
                st.rerun()
                
            except Exception as e:
                st.error(f"Error initializing search: {str(e)}")
                if st.session_state.show_debug:
                    import traceback
                    st.error(traceback.format_exc())
    
    # Active search UI
    else:
        # Display connection status
        st.markdown(f"""
        <div class="search-info">
            <div><span class="status-indicator status-green"></span> Connected to: <strong>{db_type.capitalize()}</strong></div>
            <div>Collection: <strong>{collection_name}</strong></div>
            <div>Embedding: <strong>{embedding_provider}</strong></div>
        </div>
        """, unsafe_allow_html=True)
        
        # Search interface
        st.write("### Enter your search query")
        
        # Create search form with text input and button side by side
        col1, col2 = st.columns([3, 1])

        def handle_search_submitted():
            st.session_state.search_triggered = True
            st.session_state.current_page = 1
            st.session_state.last_query = st.session_state.search_text
        
        with col1:
            search_text = st.text_input("Search Query", key="search_text", on_change=handle_search_submitted)
        
        with col2:
            search_type = st.selectbox("Search Type", ["hybrid", "vector", "sparse", "keyword"])
        
        # Search button - now we check both button click and if Enter was pressed
        search_clicked = st.button("Search")

        # This flag will only be set once when Enter is pressed
        if "search_triggered" not in st.session_state:
            st.session_state.search_triggered = False

        if search_clicked or st.session_state.search_triggered:
            st.session_state.search_triggered = False

            query = st.session_state.search_text
            if not query:
                st.warning("Please enter a search query.")
            else:
                try:
                    with st.spinner("Searching..."):
                        # Get search results - ask for more results for pagination
                        total_results_to_fetch = max_results
                        
                        # Score threshold from UI
                        threshold = score_threshold if score_threshold > 0 else None
                        
                        results = gpu_safe_call(st.session_state.db_manager.search,
                            query=query,
                            search_type=search_type,
                            limit=total_results_to_fetch,
                            processor=st.session_state.processor,
                            prefetch_limit=prefetch_limit,
                            fusion_type=fusion_type,
                            relevance_tuning=True,
                            context_size=st.session_state.context_size,
                            score_threshold=threshold,
                            rerank=rerank,
                            reranker_type=reranker_type if rerank else None
                        )

                        # Optional sorting by score
                        if sort_by_score:
                            results['results'].sort(key=lambda x: x['score'], reverse=True)

                        # Re-assign rank after sorting
                        for i, r in enumerate(results['results']):
                            r['rank'] = i + 1
                        
                        # Store results in session state
                        st.session_state.search_results = results
                        
                        # Store total results count for pagination
                        if "count" in results:
                            st.session_state.total_results = results["count"]
                        else:
                            st.session_state.total_results = 0
                        
                        # Store last successful query
                        st.session_state.last_query = query
                    
                except Exception as e:
                    st.error(f"Error during search: {str(e)}")
                    
                    # Check if it's a collection not found error
                    # Check if it's a collection not found error
                    if "Collection" in str(e) and "does not exist" in str(e):
                        st.error("The collection doesn't exist. Try initializing search again or check your collection name.")
                        
                        # Reset the initialization flag to force re-initialization
                        st.session_state.search_initialized = False
                    
                    if st.session_state.show_debug:
                        import traceback
                        st.error(traceback.format_exc())
        
        # Display search results
        if st.session_state.search_results:
            results = st.session_state.search_results
            
            if "error" in results:
                st.error(f"Search error: {results['error']}")
            else:
                # Search info
                st.markdown(f"""
                <div class="search-info">
                    <p><strong>Query:</strong> {results['query']}</p>
                    <p><strong>Search type:</strong> {results['search_type']}</p>
                    <p><strong>Using embedders:</strong> {results['embedder_info']['dense']} (dense), {results['embedder_info']['sparse']} (sparse)</p>
                    <p><strong>Total results:</strong> {results['count']}</p>
                </div>
                """, unsafe_allow_html=True)
                
                if results['count'] == 0:
                    st.warning("No results found for your query.")
                else:
                    # Pagination logic
                    total_pages = (results['count'] + st.session_state.search_limit - 1) // st.session_state.search_limit
                    
                    # Make sure current page is within bounds
                    if st.session_state.current_page < 1:
                        st.session_state.current_page = 1
                    if st.session_state.current_page > total_pages:
                        st.session_state.current_page = total_pages
                    
                    # Calculate start and end indices for current page
                    start_idx = (st.session_state.current_page - 1) * st.session_state.search_limit
                    end_idx = min(start_idx + st.session_state.search_limit, results['count'])
                    
                    # Display page info
                    st.markdown(f"""
                    <div class="page-info">
                        Showing results {start_idx + 1}-{end_idx} of {results['count']}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Get current page results
                    page_results = results['results'][start_idx:end_idx]
                    
                    # Display each result
                    for result in page_results:
                        # Format the preview by replacing markdown ** with HTML span for highlighting
                        preview = result['preview']
                        preview = preview.replace('**', '<span class="highlight">')
                        preview = preview.replace('**', '</span>')
                        
                        # Calculate score color based on value (higher = greener)
                        score = result['score']
                        score_color = f"color: rgb({int(255 * (1 - score))}, {int(255 * score)}, 0)"
                        
                        st.markdown(f"""
                        <div class="result-container">
                            <div class="filename">{result['file_name']} - Rank: {result['rank']}, <span style="{score_color}">Score: {result['score']:.4f}</span></div>
                            <div class="meta-info">Path: {result['file_path']}</div>
                            <div class="meta-info">Chunk: {result['chunk_index']} ({result['chunk_size']['characters']} chars, {result['chunk_size']['words']} words)</div>
                            <div class="meta-info">
                                Embedders: {result['embedder_info']['dense_embedder']} (dense), 
                                {result['embedder_info']['sparse_embedder']} (sparse)
                            </div>
                            <div class="preview-text">{preview}</div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Expandable sections for full context and file actions
                        with st.expander("Show Full Context"):
                            # First try big_context, fall back to text if not available
                            context_text = result.get("big_context", result.get("text", ""))
                            st.text_area("Full Context", context_text, height=400)
                            
                            # Add action buttons
                            col1, col2 = st.columns([1, 1])
                            
                            # Open file button (if file exists)
                            with col1:
                                file_path = result['file_path']
                                if os.path.exists(file_path):
                                    if st.button(f"Open File", key=f"open_{result['id']}"):
                                        try:
                                            # Determine the OS and open file accordingly
                                            system = platform.system()
                                            if system == 'Darwin':  # macOS
                                                subprocess.run(['open', file_path])
                                            elif system == 'Windows':
                                                subprocess.run(['start', file_path], shell=True)
                                            else:  # Linux and others
                                                subprocess.run(['xdg-open', file_path])
                                                
                                            st.success(f"Opening {file_path}")
                                        except Exception as e:
                                            st.error(f"Error opening file: {e}")
                            
                            # Copy content button
                            with col2:
                                if st.button(f"Copy Content", key=f"copy_{result['id']}"):
                                    try:
                                        st.write("Copied to clipboard!")
                                        # Note: In Streamlit we can't directly access the clipboard,
                                        # but we can use JavaScript via components for this
                                        st.markdown(f"""
                                        <div class="copy-text" onclick="navigator.clipboard.writeText(`{context_text.replace('`', '\`')}`)">
                                            <span style="color: green">✓ Click here to copy</span>
                                        </div>
                                        """, unsafe_allow_html=True)
                                    except Exception as e:
                                        st.error(f"Error: {e}")
                    
                    # Pagination controls
                    col1, col2, col3, col4, col5 = st.columns([1, 1, 3, 1, 1])
                    
                    with col1:
                        if st.session_state.current_page > 1:
                            if st.button("⏮️ First"):
                                st.session_state.current_page = 1
                                st.rerun()
                    
                    with col2:
                        if st.session_state.current_page > 1:
                            if st.button("◀️ Previous"):
                                st.session_state.current_page -= 1
                                st.rerun()
                    
                    with col3:
                        # Create a centered container for page numbers
                        st.markdown(f"""
                        <div class="centered">
                            <span>Page {st.session_state.current_page} of {total_pages}</span>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col4:
                        if st.session_state.current_page < total_pages:
                            if st.button("Next ▶️"):
                                st.session_state.current_page += 1
                                st.rerun()
                    
                    with col5:
                        if st.session_state.current_page < total_pages:
                            if st.button("Last ⏭️"):
                                st.session_state.current_page = total_pages
                                st.rerun()

# tab 3 for LLM inference:
with tab3:
    st.title("Chat with Documents")
    
    # Check if search is initialized
    if not st.session_state.search_initialized:
        st.info("Please initialize search connection in the Search tab first.")
    else:
        # LLM Settings sidebar
        with st.sidebar:
            st.title("LLM Settings")
            
            # LLM provider settings
            if expandable_section("LLM Provider", "llm_settings_expanded", True, "🤖"):
                llm_provider = st.radio("LLM Provider", ["Ollama", "MLX", "Transformers", "llama.cpp"])
                st.session_state.llm_provider = llm_provider
                
                # Different model options based on provider
                if llm_provider == "Ollama":
                    # Check Ollama availability
                    try:
                        response = requests.get("http://localhost:11434/api/tags")
                        if response.status_code == 200:
                            models = [model["name"] for model in response.json()["models"]]
                            ollama_model = st.selectbox("Ollama Model", models) if models else st.text_input("Ollama Model", DEFAULT_OLLAMA_MODEL)
                        else:
                            ollama_model = st.text_input("Ollama Model", DEFAULT_OLLAMA_MODEL)
                    except:
                        ollama_model = st.text_input("Ollama Model", DEFAULT_OLLAMA_MODEL)
                        
                    st.session_state.llm_model = ollama_model
                    ollama_host = st.text_input("Ollama Host", "http://localhost:11434")
                    
                elif llm_provider == "MLX":
                    # Check for MLX models
                    # Note: This is just an example, adjust based on your actual MLX model detection
                    mlx_model = st.text_input("MLX Model Path/ID", DEFAULT_MLX_MODEL)
                    st.session_state.llm_model = mlx_model
                    st.info("For local MLX models, provide the model directory path. For Hugging Face models, provide the model ID.")
                    
                elif llm_provider == "Transformers":
                    # Transformers model selection
                    transformers_model = st.text_input("Transformers Model ID", DEFAULT_TRANSFORMERS_MODEL)
                    st.session_state.llm_model = transformers_model
                    use_gpu = st.checkbox("Use GPU (if available)", True)
                    
                elif llm_provider == "llama.cpp":
                    # llama.cpp model selection
                    llamacpp_model = st.text_input("llama.cpp Model Path", DEFAULT_LLAMACPP_MODEL)
                    st.session_state.llm_model = llamacpp_model
                    n_gpu_layers = st.slider("GPU Layers", 0, 100, 0, 
                                           help="Number of layers to offload to GPU (0 = CPU only)")
                    n_threads = st.slider("CPU Threads", 1, 16, 4)

                # Add auto-search option
                st.session_state.auto_search_enabled = st.checkbox("Enable Automatic Search", True,
                                                                help="Let the LLM automatically generate search queries")
                if st.session_state.auto_search_enabled:
                    auto_search_strategy = st.radio("Auto-Search Strategy", 
                                                ["always", "when_needed", "ask_first"],
                                                format_func=lambda x: {
                                                    "always": "Always Search First",
                                                    "when_needed": "Search Only When Needed",
                                                    "ask_first": "Ask Before Searching"
                                                }.get(x))
                    st.session_state.auto_search_strategy = auto_search_strategy
                    
                    max_auto_searches = st.slider("Max Auto Searches Per Question", 1, 5, 2,
                                                help="Maximum number of different searches the LLM can perform for one question")
                    st.session_state.max_auto_searches = max_auto_searches


            # Context settings
            if expandable_section("Context Settings", "context_settings_expanded", False, "📄"):
                st.session_state.context_from_search = st.checkbox("Use Search Results as Context", True)
                
                context_strategy = st.radio("Context Strategy", 
                                          ["top_results", "most_relevant", "custom_selection"],
                                          format_func=lambda x: {
                                              "top_results": "Top N Results",
                                              "most_relevant": "Most Relevant (Above Score Threshold)",
                                              "custom_selection": "Manually Select Results"
                                          }.get(x))
                st.session_state.context_strategy = context_strategy
                
                if context_strategy == "top_results":
                    n_results = st.slider("Number of Results to Include", 1, 20, 5)
                elif context_strategy == "most_relevant":
                    score_threshold = st.slider("Minimum Score", 0.0, 1.0, 0.7, 0.05)
                
                max_context_length = st.slider("Max Context Length (tokens)", 1000, 32000, 8000, 1000)
                
                # Custom system prompt
                st.text_area("System Prompt", LLM_SYSTEM_PROMPT, key="llm_system_prompt", height=100)

            # Generation settings
            if expandable_section("Generation Settings", "generation_settings_expanded", False, "⚙️"):
                st.session_state.llm_temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.05)
                st.session_state.llm_max_tokens = st.slider("Max Tokens", 128, 4096, 1024, 128)
                st.session_state.llm_streaming = st.checkbox("Stream Output", True)
                
                # Advanced provider-specific settings
                if llm_provider == "Ollama":
                    top_p = st.slider("Top P", 0.0, 1.0, 0.9, 0.05)
                    top_k = st.slider("Top K", 1, 100, 40)
                elif llm_provider == "Transformers" or llm_provider == "MLX":
                    top_p = st.slider("Top P", 0.0, 1.0, 0.9, 0.05)
                    repetition_penalty = st.slider("Repetition Penalty", 1.0, 2.0, 1.1, 0.05)
        
        # Context display and chat interface
        context_expander = st.expander("RAG Context", expanded=False)
        
        with context_expander:
            # Check if we have search results
            if st.session_state.search_results:
                results = st.session_state.search_results.get('results', [])
                
                if not results:
                    st.info("No search results available for context. Run a search to get context.")
                else:
                    # Display auto-search information if any was performed
                    if st.session_state.auto_searched_queries and len(st.session_state.auto_searched_queries) > 0:
                        st.markdown("### Auto-generated search queries:")
                        for i, query in enumerate(st.session_state.auto_searched_queries):
                            st.markdown(f"{i+1}. `{query}`")               
                    
                    # Handle different context strategies
                    if st.session_state.context_strategy == "top_results":
                        n_results = n_results if 'n_results' in locals() else 5
                        selected_results = results[:n_results]
                        st.write(f"Using top {len(selected_results)} search results as context")
                    
                    elif st.session_state.context_strategy == "most_relevant":
                        threshold = score_threshold if 'score_threshold' in locals() else 0.7
                        selected_results = [r for r in results if r['score'] >= threshold]
                        st.write(f"Using {len(selected_results)} results with score ≥ {threshold}")
                    
                    elif st.session_state.context_strategy == "custom_selection":
                        st.write("Select context passages to include:")
                        selected_indices = []
                        
                        for i, result in enumerate(results[:10]):  # Limit to first 10 for UI clarity
                            selected = st.checkbox(
                                f"{result['file_name']} (Score: {result['score']:.2f})", 
                                value=(i < 3),  # Default select first 3
                                key=f"select_context_{i}"
                            )
                            if selected:
                                selected_indices.append(i)
                        
                        selected_results = [results[i] for i in selected_indices]
                        st.write(f"Using {len(selected_results)} manually selected results as context")
                    
                    # Display selected context
                    if selected_results:
                        context_text = "\n\n".join([
                            f"[Document: {r['file_name']}, Score: {r['score']:.2f}]\n{r['text']}"
                            for r in selected_results
                        ])
                        st.text_area("Context Content", context_text, height=250)
                        
                        # Button to save context
                        if st.button("Save Context to File"):
                            try:
                                timestamp = time.strftime("%Y%m%d-%H%M%S")
                                filename = f"context_{timestamp}.txt"
                                with open(filename, "w") as f:
                                    f.write(context_text)
                                st.success(f"Context saved to {filename}")
                            except Exception as e:
                                st.error(f"Error saving context: {e}")
                    else:
                        st.warning("No context selected.")
            else:
                st.info("No search results available. Run a search in the Search tab first.")
        
        # Function to initialize the LLM based on provider
        def initialize_llm():
            provider = st.session_state.llm_provider
            model = st.session_state.llm_model
            
            if provider == "Ollama":
                return init_ollama()
            elif provider == "MLX":
                return init_mlx()
            elif provider == "Transformers":
                return init_transformers()
            elif provider == "llama.cpp":
                return init_llamacpp()
            else:
                st.error(f"Unknown LLM provider: {provider}")
                return None
        
        # Provider-specific initialization functions
        def init_ollama():
            try:
                return {"type": "ollama", "initialized": True}
            except Exception as e:
                st.error(f"Failed to initialize Ollama: {e}")
                return {"type": "ollama", "initialized": False, "error": str(e)}
        
        def init_mlx():
            """Initialize MLX model for text generation."""
            try:
                # Check if MLX is available
                mlx_spec = importlib.util.find_spec("mlx")
                if mlx_spec is None:
                    return {"type": "mlx", "initialized": False, "error": "MLX not installed"}
                
                mlx_lm_spec = importlib.util.find_spec("mlx_lm")
                if mlx_lm_spec is None:
                    return {"type": "mlx", "initialized": False, "error": "MLX LM not installed. Install with 'pip install mlx-lm'"}
                
                # Check if model path/ID exists
                model_id = st.session_state.llm_model
                
                # For local models, check if path exists
                if os.path.exists(model_id):
                    if not os.path.isdir(model_id):
                        return {"type": "mlx", "initialized": False, "error": f"Model path {model_id} is not a directory"}
                
                # Don't actually load the model here to save memory
                # We'll load it when we need to generate
                return {"type": "mlx", "initialized": True, "model_id": model_id}
            except Exception as e:
                st.error(f"Failed to initialize MLX: {e}")
                if st.session_state.show_debug:
                    import traceback
                    st.error(traceback.format_exc())
                return {"type": "mlx", "initialized": False, "error": str(e)}

        
        def init_transformers():
            try:
                # Check if transformers is available
                transformers_spec = importlib.util.find_spec("transformers")
                if transformers_spec is None:
                    return {"type": "transformers", "initialized": False, "error": "Transformers not installed"}
                
                # We don't actually load the model here, just check availability
                return {"type": "transformers", "initialized": True}
            except Exception as e:
                st.error(f"Failed to initialize Transformers: {e}")
                return {"type": "transformers", "initialized": False, "error": str(e)}
        
        def init_llamacpp():
            try:
                # Check if llama.cpp Python bindings are available
                llamacpp_spec = importlib.util.find_spec("llama_cpp")
                if llamacpp_spec is None:
                    return {"type": "llamacpp", "initialized": False, "error": "llama.cpp Python bindings not installed"}
                
                # Check if model path exists
                model_path = os.path.expanduser(st.session_state.llm_model)
                if not os.path.exists(model_path):
                    return {"type": "llamacpp", "initialized": False, "error": f"Model file not found: {model_path}"}
                
                return {"type": "llamacpp", "initialized": True}
            except Exception as e:
                st.error(f"Failed to initialize llama.cpp: {e}")
                return {"type": "llamacpp", "initialized": False, "error": str(e)}
        
        # Function to generate responses
        def generate_response(llm_info, prompt, context=None):
            provider = st.session_state.llm_provider
            model = st.session_state.llm_model
            temperature = st.session_state.llm_temperature
            max_tokens = st.session_state.llm_max_tokens
            system_prompt = st.session_state.llm_system_prompt
            
            # Combine context and prompt
            if context and context.strip():
                # Use a different system prompt for RAG
                rag_system_prompt = RAG_SYSTEM_PROMPT
                full_prompt = f"Context information:\n\n{context}\n\nQuestion: {prompt}\n\nAnswer:"
            else:
                # Use the original system prompt if no context
                rag_system_prompt = system_prompt
                full_prompt = prompt
            
            # Call the appropriate provider-specific function
            if provider == "Ollama":
                return generate_ollama(model, rag_system_prompt, full_prompt, temperature, max_tokens)
            elif provider == "MLX":
                return generate_mlx(model, rag_system_prompt, full_prompt, temperature, max_tokens)
            elif provider == "Transformers":
                return generate_transformers(model, rag_system_prompt, full_prompt, temperature, max_tokens)
            elif provider == "llama.cpp":
                return generate_llamacpp(model, rag_system_prompt, full_prompt, temperature, max_tokens)
            else:
                return "Error: Unknown LLM provider"
        
        # Provider-specific generation functions
        def generate_ollama(model, system, prompt, temperature, max_tokens):
            try:
                # For streaming
                if st.session_state.llm_streaming:
                    url = "http://localhost:11434/api/generate"
                    data = {
                        "model": model,
                        "prompt": prompt,
                        "system": system,
                        "stream": True,
                        "temperature": temperature,
                        "max_tokens": max_tokens
                    }
                    
                    response = requests.post(url, json=data, stream=True)
                    response.raise_for_status()
                    
                    full_response = ""
                    for line in response.iter_lines():
                        if line:
                            chunk = json.loads(line)
                            response_chunk = chunk.get("response", "")
                            full_response += response_chunk
                            st.session_state.llm_response_queue.put(response_chunk)
                            
                            if chunk.get("done", False):
                                break
                    
                    return full_response
                else:
                    # Non-streaming version
                    url = "http://localhost:11434/api/generate"
                    data = {
                        "model": model,
                        "prompt": prompt,
                        "system": system,
                        "stream": False,
                        "temperature": temperature,
                        "max_tokens": max_tokens
                    }
                    
                    response = requests.post(url, json=data)
                    response.raise_for_status()
                    return response.json().get("response", "")
            except Exception as e:
                st.error(f"Error generating with Ollama: {e}")
                return f"Error: {str(e)}"
        
        def generate_mlx(model_id, system, prompt, temperature, max_tokens):
            """Generate text using MLX models."""
            try:
                # Import mlx_lm here to avoid loading on startup
                from mlx_lm import load, generate, stream_generate
                import mlx.core as mx
                
                # Format the prompt based on likely system template
                # This is tricky because different models use different templates
                # We'll try to detect if it's an instruct model and format accordingly
                
                if "instruct" in model_id.lower() or "chat" in model_id.lower():
                    # For instruction-tuned models, wrap in a chat template
                    messages = [
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt}
                    ]
                    
                    # Progress update
                    st.session_state.llm_response_queue.put("Loading model and tokenizer...\n")
                    
                    # Load the model and tokenizer
                    model, tokenizer = gpu_safe_call(load, model_id)
                    
                    st.session_state.llm_response_queue.put("Formatting prompt...\n")
                    
                    # Try to apply chat template
                    try:
                        formatted_prompt = tokenizer.apply_chat_template(
                            messages, add_generation_prompt=True
                        )
                    except Exception as e:
                        # Fallback format if chat template fails
                        if st.session_state.show_debug:
                            st.error(f"Chat template failed: {e}, using fallback format")
                        
                        formatted_prompt = f"<s>[INST] <<SYS>>\n{system}\n<</SYS>>\n\n{prompt} [/INST]"
                else:
                    # For base models, use a simpler format
                    st.session_state.llm_response_queue.put("Loading model and tokenizer...\n")
                    model, tokenizer = gpu_safe_call(load, model_id)
                    
                    # Simple prompt format for non-instruct models
                    formatted_prompt = f"{system}\n\n{prompt}"
                
                # Prepare generation config
                generation_args = {
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "prompt": formatted_prompt,
                }
                
                if "top_p" in st.session_state:
                    generation_args["top_p"] = st.session_state.top_p
                
                if "repetition_penalty" in st.session_state:
                    generation_args["repetition_penalty"] = st.session_state.repetition_penalty
                
                # Generate with streaming
                if st.session_state.llm_streaming:
                    st.session_state.llm_response_queue.put("Generating response...\n")
                    full_response = ""
                    
                    # Stream generate with MLX
                    for response in gpu_safe_call(stream_generate, model, tokenizer, **generation_args):
                        chunk = response.text
                        full_response += chunk
                        st.session_state.llm_response_queue.put(chunk)
                    
                    return full_response
                else:
                    # Non-streaming version
                    st.session_state.llm_response_queue.put("Generating response...\n")
                    response = gpu_safe_call(generate, model, tokenizer, **generation_args)
                    return response
                    
            except Exception as e:
                error_msg = f"Error generating with MLX: {e}"
                st.error(error_msg)
                if st.session_state.show_debug:
                    import traceback
                    st.error(traceback.format_exc())
                return f"Error: {str(e)}"
        
        def generate_transformers(model, system, prompt, temperature, max_tokens):
            try:
                # Import required libraries
                from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
                import torch
                
                st.info("Loading Transformers model... this may take a while.")
                
                # Prepare the prompt in the correct format for the model
                # This format works with Llama 2 and similar models
                formatted_prompt = f"<s>[INST] <<SYS>>\n{system}\n<</SYS>>\n\n{prompt} [/INST]"
                
                # Load model and tokenizer
                tokenizer = AutoTokenizer.from_pretrained(model)
                model = AutoModelForCausalLM.from_pretrained(
                    model,
                    torch_dtype=torch.float16,
                    device_map="auto"
                )
                
                # Generate
                if st.session_state.llm_streaming:
                    # Stream tokens
                    tokens = tokenizer.encode(formatted_prompt, return_tensors="pt").to(model.device)
                    streamer = TextStreamer(tokenizer)
                    
                    def stream_to_queue():
                        full_response = ""
                        with torch.no_grad():
                            generated_ids = model.generate(
                                tokens,
                                max_new_tokens=max_tokens,
                                temperature=temperature,
                                streamer=streamer
                            )
                        
                        # Convert back to text
                        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                        response_text = generated_text[len(tokenizer.decode(tokens[0], skip_special_tokens=True)):]
                        return response_text
                    
                    # Run in a separate thread
                    response_thread = Thread(target=stream_to_queue)
                    response_thread.start()
                    response_thread.join()
                    
                    # Collect streaming output here
                    full_response = ""
                    while not st.session_state.llm_response_queue.empty():
                        chunk = st.session_state.llm_response_queue.get()
                        full_response += chunk
                    
                    return full_response
                else:
                    # Non-streaming version
                    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=max_tokens,
                            temperature=temperature
                        )
                    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    # Extract only the generated part (not the prompt)
                    response_only = response[len(tokenizer.decode(inputs.input_ids[0], skip_special_tokens=True)):]
                    return response_only
            except Exception as e:
                st.error(f"Error generating with Transformers: {e}")
                if st.session_state.show_debug:
                    import traceback
                    st.error(traceback.format_exc())
                return f"Error: {str(e)}"
                
        class TextStreamer:
            def __init__(self, tokenizer):
                self.tokenizer = tokenizer
            
            def put(self, token_ids):
                text = self.tokenizer.decode(token_ids)
                st.session_state.llm_response_queue.put(text)
            
            def end(self):
                pass
        
        def generate_llamacpp(model, system, prompt, temperature, max_tokens):
            try:
                from llama_cpp import Llama
                
                # Expand user path (~/ to absolute path)
                model_path = os.path.expanduser(model)
                
                # Get parameters from session state
                n_gpu_layers = n_gpu_layers if 'n_gpu_layers' in locals() else 0
                n_threads = n_threads if 'n_threads' in locals() else 4
                
                # Initialize model
                llm = Llama(
                    model_path=model_path,
                    n_gpu_layers=n_gpu_layers,
                    n_threads=n_threads
                )
                
                # Format the prompt appropriately for llama.cpp
                # This format works for Llama 2 and similar instruction models
                formatted_prompt = f"<s>[INST] <<SYS>>\n{system}\n<</SYS>>\n\n{prompt} [/INST]"
                
                # Generate with streaming
                if st.session_state.llm_streaming:
                    full_response = ""
                    for chunk in llm(
                        formatted_prompt,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        stream=True
                    ):
                        text_chunk = chunk["choices"][0]["text"]
                        full_response += text_chunk
                        st.session_state.llm_response_queue.put(text_chunk)
                    
                    return full_response
                else:
                    # Non-streaming version
                    output = llm(
                        formatted_prompt,
                        max_tokens=max_tokens,
                        temperature=temperature
                    )
                    return output["choices"][0]["text"]
            except Exception as e:
                st.error(f"Error generating with llama.cpp: {e}")
                if st.session_state.show_debug:
                    import traceback
                    st.error(traceback.format_exc())
                return f"Error: {str(e)}"
        
        # Background thread for LLM inference
        def run_llm_inference(prompt, provided_context=None):
            try:
                # Set the generating flag
                st.session_state.llm_is_generating = True
                
                # Reset search attempt counter for this question
                st.session_state.search_attempts = 0
                st.session_state.auto_searched_queries = []
                
                # Initialize LLM if needed
                llm_info = initialize_llm()
                
                if not llm_info or not llm_info.get("initialized", False):
                    error = llm_info.get("error", "Unknown error initializing LLM") if llm_info else "Failed to initialize LLM"
                    st.session_state.llm_response_queue.put(f"Error: {error}")
                    st.session_state.llm_is_generating = False
                    return
                
                # Use existing context if provided
                context = provided_context
                
                # Auto-search logic
                if st.session_state.auto_search_enabled and (not context or context.strip() == ""):
                    strategy = st.session_state.auto_search_strategy
                    
                    if strategy in ["always", "when_needed"]:
                        # Always search or search when needed (no context)
                        st.session_state.llm_response_queue.put("🔍 Generating search query...")
                        
                        # Get search query from LLM
                        search_query = get_search_query_from_llm(prompt)
                        
                        if search_query and search_query != prompt:
                            st.session_state.llm_response_queue.put(f"\n🔎 Searching for: \"{search_query}\"...\n")
                            
                            # Perform the search
                            search_results_json = perform_vector_search(search_query)
                            search_results = json.loads(search_results_json)
                            
                            if "error" in search_results:
                                st.session_state.llm_response_queue.put(f"\n❌ Search error: {search_results['error']}")
                            else:
                                result_count = search_results.get("count", 0)
                                
                                if result_count > 0:
                                    # Format search results as context
                                    result_texts = []
                                    for result in search_results.get("results", [])[:5]:  # Use top 5 results
                                        result_texts.append(f"Document: {result['file_name']}, Score: {result['score']:.2f}\n{result['text']}")
                                    
                                    context = "\n\n".join(result_texts)
                                    st.session_state.chat_context = context
                                    
                                    # Increment search attempts
                                    st.session_state.search_attempts += 1
                                    
                                    st.session_state.llm_response_queue.put(f"📄 Found {result_count} relevant documents.\n\n")
                                else:
                                    st.session_state.llm_response_queue.put("❓ No relevant documents found in initial search.")
                                    
                                    # Try a second search if allowed
                                    if st.session_state.search_attempts < st.session_state.max_auto_searches:
                                        st.session_state.llm_response_queue.put("\n🔄 Trying an alternative search query...\n")
                                        
                                        # Generate a different search query
                                        alt_query = get_alternative_search_query(prompt, search_query)
                                        
                                        if alt_query and alt_query != search_query:
                                            st.session_state.llm_response_queue.put(f"🔎 Searching for: \"{alt_query}\"...\n")
                                            
                                            # Perform the second search
                                            alt_results_json = perform_vector_search(alt_query)
                                            alt_results = json.loads(alt_results_json)
                                            
                                            if "error" not in alt_results and alt_results.get("count", 0) > 0:
                                                # Format search results as context
                                                result_texts = []
                                                for result in alt_results.get("results", [])[:5]:
                                                    result_texts.append(f"Document: {result['file_name']}, Score: {result['score']:.2f}\n{result['text']}")
                                                
                                                context = "\n\n".join(result_texts)
                                                st.session_state.chat_context = context
                                                
                                                st.session_state.llm_response_queue.put(f"📄 Found {alt_results.get('count', 0)} relevant documents with alternative query.\n\n")
                                            else:
                                                st.session_state.llm_response_queue.put("❓ No relevant documents found with alternative query either.\n\n")
                        else:
                            st.session_state.llm_response_queue.put("⚠️ Could not generate a search query. Using original question.\n\n")
                    
                    elif strategy == "ask_first":
                        # Implementation for "ask_first" would go here
                        # This would require a separate UI flow where we first ask the user
                        # if they want to search, then do the search.
                        # For now, we'll just use the original question as the search query
                        pass
                
                # Generate response
                response = generate_response(llm_info, prompt, context)
                
                # Update chat history with the complete response
                st.session_state.llm_history.append({"role": "user", "content": prompt})
                st.session_state.llm_history.append({"role": "assistant", "content": response})
                
                # Clear the generating flag
                st.session_state.llm_is_generating = False
            except Exception as e:
                st.session_state.llm_response_queue.put(f"Error: {str(e)}")
                st.session_state.llm_is_generating = False
                if st.session_state.show_debug:
                    import traceback
                    st.error(traceback.format_exc())


        
        # Get context from selected search results
        context_text = ""
        if st.session_state.context_from_search and st.session_state.search_results:
            results = st.session_state.search_results.get('results', [])
            
            if results:
                # Use previously selected results from the context expander
                if 'selected_results' in locals() and selected_results:
                    context_text = "\n\n".join([
                        f"Document: {r['file_name']}\n{r['text']}"
                        for r in selected_results
                    ])
                # Default context selection if the expander wasn't used
                else:
                    # Default to top 3 results
                    top_results = results[:3]
                    context_text = "\n\n".join([
                        f"Document: {r['file_name']}\n{r['text']}"
                        for r in top_results
                    ])
        
        # Display chat history
        st.write("### Chat")
                
        # Add a switch to toggle between using search results as context or not ###
        col1, col2 = st.columns([3, 1])
        with col1:
            st.session_state.context_from_search = st.checkbox(
                "Use search results as context", 
                value=st.session_state.context_from_search,
                help="When enabled, answers will be based on search results"
            )
        with col2:
            if st.button("📝 View/Edit Context"):
                context_expander.expanded = not context_expander.expanded

        
        for message in st.session_state.llm_history:
            role = message["role"]
            content = message["content"]
            
            if role == "user":
                st.markdown(f"<div style='background-color: #e6f7ff; padding: 10px; border-radius: 5px; margin-bottom: 10px;'><strong>You:</strong> {content}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div style='background-color: #f0f0f0; padding: 10px; border-radius: 5px; margin-bottom: 10px;'><strong>Assistant:</strong> {content}</div>", unsafe_allow_html=True)
        
        # Show streaming output if generating
        if st.session_state.llm_is_generating:
            streaming_container = st.empty()
            streaming_text = ""
            
            # Display "Generating..." initially
            streaming_container.markdown("<div style='background-color: #f0f0f0; padding: 10px; border-radius: 5px;'><strong>Assistant:</strong> <i>Generating...</i></div>", unsafe_allow_html=True)
            
            # Update with streaming text
            while st.session_state.llm_is_generating:
                try:
                    # Try to get new text with timeout
                    try:
                        new_text = st.session_state.llm_response_queue.get(timeout=0.1)
                        streaming_text += new_text
                        streaming_container.markdown(f"<div style='background-color: #f0f0f0; padding: 10px; border-radius: 5px;'><strong>Assistant:</strong> {streaming_text}</div>", unsafe_allow_html=True)
                    except queue.Empty:
                        # No new text, just wait
                        time.sleep(0.1)
                        
                    # Rerun to update UI
                    time.sleep(0.1)
                except Exception as e:
                    if st.session_state.show_debug:
                        st.error(f"Error in streaming: {e}")
                    break
        
        # Chat input
        user_input = st.chat_input("Ask a question about your documents...")

        if user_input and not st.session_state.llm_is_generating:
            # Get context if we're using it
            context_text = ""
            if st.session_state.context_from_search:
                # Use the context from the chat_context session variable if available
                if st.session_state.chat_context:
                    context_text = st.session_state.chat_context
                # Otherwise, fall back to search results if available
                elif st.session_state.search_results and st.session_state.search_results.get('results', []):
                    results = st.session_state.search_results.get('results', [])
                    top_results = results[:5]  # Use top 5 results
                    context_text = "\n\n".join([
                        f"Document: {r['file_name']}\n{r['text']}"
                        for r in top_results
                    ])
            
            # Start an LLM thread with context
            llm_thread = Thread(target=run_llm_inference, args=(user_input, context_text))
            llm_thread.daemon = True
            llm_thread.start()
            
            # Store the thread reference
            st.session_state.llm_thread = llm_thread
            
            # Force a rerun to start showing streaming output
            st.rerun()

        
        # Add buttons for actions
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Clear Chat History"):
                st.session_state.llm_history = []
                st.rerun()
        
        with col2:
            if st.button("Try New Search"):
                # Switch to search tab
                st.switch_page("#Search Documents")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center;">
        <p>MLXrag Search - Powered by MLX, Ollama, FastEmbed</p>
        <p style="font-size: 0.8em;">Supports multiple vector databases: Qdrant, Milvus, LanceDB, Meilisearch, ElasticSearch</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Run the script if it's the main module
if __name__ == "__main__":
    pass  # The Streamlit app runs automatically