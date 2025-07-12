import os
import time
import psutil
import numpy as np
import pandas as pd
import torch # NEW: Import torch
from typing import List, Tuple, Dict
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from rapidfuzz import process, fuzz
from transformers import AutoModel, AutoTokenizer # NEW: Import from transformers

# --- LLM Embedding Wrapper and Configuration ---
# This class handles loading and embedding for large language models (LLMs)
# that are not typically loaded by SentenceTransformer.
class LLMEmbedder:
    """
    A wrapper class to make LLMs callable like SentenceTransformer models
    for embedding purposes, using Hugging Face transformers.
    """
    def __init__(self, model_name: str):
        self.model_name = model_name
        
        # Load tokenizer and model
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            # Use float16 for memory efficiency; adjust device as needed.
            # device_map="auto" is recommended for very large models to utilize multiple GPUs.
            self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.float16, device_map="auto") 
            self.model.eval() # Set to evaluation mode for inference
        except Exception as e:
            raise RuntimeError(f"Failed to load LLM '{model_name}': {e}")

        # Ensure tokenizer has a pad_token
        # Phi-3 and other LLMs might not have a pad_token by default, which is needed for batching.
        if self.tokenizer.pad_token is None:
            # A common practice is to set it to eos_token or add a new one if necessary.
            # For Phi-3, eos_token works well for padding.
            if self.tokenizer.eos_token:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                # If a new token is added, resize token embeddings.
                # Only uncomment if you actually added a *new* token, not just assigned an existing one.
                # self.model.resize_token_embeddings(len(self.tokenizer))

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # If device_map="auto" is used above, model is already on correct devices.
        # Otherwise, explicitly move it:
        if self.model.device.type != self.device:
             self.model.to(self.device)

    def encode(self, sentences: List[str], batch_size: int = 32, show_progress_bar: bool = False, convert_to_numpy: bool = True):
        """
        Generates embeddings for a list of sentences using the loaded LLM.
        """
        all_embeddings = []
        
        # Optional: Add tqdm for progress bar if available
        # from tqdm import tqdm
        # sentence_batches = [sentences[i:i + batch_size] for i in range(0, len(sentences), batch_size)]
        # if show_progress_bar and 'tqdm' in globals(): # Check if tqdm is imported
        #     sentence_batches = tqdm(sentence_batches, desc=f"Encoding with {self.model_name}")
        # else:
        #     sentence_batches = [sentences[i:i + batch_size] for i in range(0, len(sentences), batch_size)]

        for i in range(0, len(sentences), batch_size):
            batch_sentences = sentences[i:i + batch_size]
            
            # Tokenize batch, return PyTorch tensors, pad to longest in batch, truncate if too long
            # max_length is important to prevent excessively long inputs for LLMs
            inputs = self.tokenizer(batch_sentences, return_tensors="pt", padding=True, 
                                    truncation=True, max_length=512).to(self.device)
            
            with torch.no_grad(): # Disable gradient calculations for inference
                # Get hidden states. output_hidden_states=True ensures we get all layer outputs.
                outputs = self.model(**inputs, output_hidden_states=True)
            
            # Mean pooling of the last hidden state, ignoring padding tokens.
            # This is a common method to get sentence embeddings from generative LLMs.
            last_hidden_states = outputs.last_hidden_state
            attention_mask = inputs['attention_mask'].unsqueeze(-1) # Expand mask for element-wise multiplication
            
            # Sum the hidden states weighted by the attention mask, then divide by the sum of the mask
            # torch.clamp is used to prevent division by zero for cases where sum_attention_mask might be 0 (e.g., empty string after truncation)
            sum_attention_mask = attention_mask.sum(dim=1)
            embeddings = (last_hidden_states * attention_mask).sum(dim=1) / torch.clamp(sum_attention_mask, min=1e-9)
            
            # L2 Normalize embeddings to unit length (important for cosine similarity)
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            
            all_embeddings.append(embeddings.cpu()) # Move to CPU to free up GPU memory

        final_embeddings = torch.cat(all_embeddings, dim=0) # Concatenate all batch embeddings
        return final_embeddings.numpy() if convert_to_numpy else final_embeddings # Return as NumPy array or PyTorch tensor


# Define the list of models that should be loaded as LLMs for embedding.
# Ensure these are the EXACT Hugging Face IDs used for models like Phi-3, Gemma, Qwen2, etc.
# These entries should match the full IDs in your models.py where they are listed.
LLM_MODELS_FOR_EMBEDDING = [
    "microsoft/Phi-3-mini-128k-instruct",
    "microsoft/Phi-3-mini-4k-instruct", # If you also use the 4k variant
    "google/gemma-2b-it", 
    "google/gemma-7b-it", 
    "Qwen/Qwen2-7B-Instruct", # Example for Qwen2
    "Qwen/Qwen2-1.5B-Instruct", # Example for Qwen2
    "mistralai/Mistral-7B-v0.1", # Example for Mistral
    "meta-llama/Llama-2-7b-hf", # Example for Llama-2 (ensure you have access)
    "Salesforce/SFR-Embedding-Mistral", # Some "embedding" models are also loaded with AutoModel/AutoTokenizer
    "Salesforce/SFR-Embedding-2_R",
    "GritLM/GritLM-7B",
    "intfloat/e5-mistral-7b-instruct",
    "Alibaba-NLP/gte-Qwen2-7B-instruct",
    "Alibaba-NLP/gte-Qwen2-1.5B-instruct",
    # Add any other models from your models.py "high_performance" or "balanced"
    # categories that are *not* SentenceTransformer models and require transformers.AutoModel.
    # For example, if "phi3" or "gemma" are shorthand names in your models.py
    # you might need a mapping here if models.py doesn't provide the full HF ID.
    # For now, assuming you update models.py to use full HF IDs or handle mapping elsewhere.
]

def calculate_mrr(predictions: List[int], correct_indices: List[int]) -> float:
    """
    Calculates Mean Reciprocal Rank (MRR).
    This function is now largely superseded by MRR calculation directly within
    compute_top1_accuracy_cosine and compute_top1_accuracy_fuzzy,
    which have access to the full scores/ranks.
    """
    # This function is now essentially a placeholder as MRR calculation logic
    # has been moved into the accuracy computation functions for better context.
    # You can remove this function if it's not called elsewhere.
    return 0.0 # Placeholder, as MRR is calculated elsewhere

def compute_top1_accuracy_fuzzy(
    source_texts: List[str],
    target_texts: List[str],
    correct_indices: List[int]
) -> Tuple[float, List[int], float]: # Changed return type to float for MRR
    preds = []
    reciprocal_ranks = [] # Renamed from correct_ranks for clarity with MRR
    for i, entry in enumerate(source_texts):
        # Use a limit for process.extract to get top N matches if performance is an issue,
        # but for MRR we ideally need all scores for proper ranking.
        all_results = process.extract(entry, target_texts, scorer=fuzz.token_sort_ratio, limit=None) # limit=None to get all matches for robust MRR
        
        pred_index = 0
        current_reciprocal_rank = 0.0 # Initialize for current query
        
        if all_results:
            # Sort by score in descending order
            sorted_results = sorted(all_results, key=lambda x: x[1], reverse=True)
            
            # The top-1 prediction for accuracy
            pred_index = sorted_results[0][2] # Get index of best match
            
            # Find the rank of the correct answer for MRR calculation
            current_correct_idx = correct_indices[i]
            found_rank = 0
            for rank, (match, score, idx) in enumerate(sorted_results):
                if idx == current_correct_idx:
                    found_rank = rank + 1 # Rank is 1-based
                    current_reciprocal_rank = 1.0 / found_rank
                    break
            reciprocal_ranks.append(current_reciprocal_rank)
        else:
            reciprocal_ranks.append(0.0) # No results, correct rank cannot be found
            
        preds.append(pred_index)

    accuracy = np.mean(np.array(preds) == np.array(correct_indices))
    mrr_score = np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0
    return accuracy, preds, mrr_score


def compute_top1_accuracy_cosine(
    queries: List[str],
    candidates: List[str],
    correct_indices: List[int],
    model_name: str
) -> Tuple[float, List[int], np.ndarray, float]: # Added return for mrr_score
    
    model = None # Initialize model to None
    print(f"DEBUG: Attempting to load model: {model_name}")

    if model_name in LLM_MODELS_FOR_EMBEDDING:
        try:
            model = LLMEmbedder(model_name)
            print(f"Successfully loaded LLM: {model_name}")
        except RuntimeError as e: # Catch the specific RuntimeError from LLMEmbedder
            print(f"Error loading LLM {model_name}: {e}. Returning default values.")
            return 0.0, [0] * len(queries), np.zeros((len(queries), len(candidates))), 0.0
        except Exception as e: # Catch any other unexpected errors during LLM loading
            print(f"Unexpected error during LLM loading {model_name}: {e}. Returning default values.")
            return 0.0, [0] * len(queries), np.zeros((len(queries), len(candidates))), 0.0
    else:
        try:
            # Explicitly setting device for SentenceTransformer models.
            # This ensures consistency with LLMEmbedder and proper GPU utilization.
            model = SentenceTransformer(model_name, device='cuda' if torch.cuda.is_available() else 'cpu') 
            print(f"Successfully loaded SentenceTransformer model: {model_name}")
        except Exception as e:
            print(f"Error loading SentenceTransformer model {model_name}: {e}. Returning default values.")
            return 0.0, [0] * len(queries), np.zeros((len(queries), len(candidates))), 0.0

    # If model loading failed and returned None, this check handles it
    if model is None:
        print(f"Skipping computation for {model_name} due to loading error.")
        return 0.0, [0] * len(queries), np.zeros((len(queries), len(candidates))), 0.0

    if not queries or not candidates:
        print("Warning: Empty queries or candidates list for cosine similarity. Returning default values.")
        return 0.0, [0] * len(queries), np.zeros((len(queries), len(candidates))), 0.0

    try:
        # Pass convert_to_numpy=True to ensure consistency as expected by cosine_similarity
        query_emb = model.encode(queries, convert_to_numpy=True, show_progress_bar=True, batch_size=32)
        cand_emb = model.encode(candidates, convert_to_numpy=True, show_progress_bar=True, batch_size=32)
    except Exception as e:
        print(f"Error encoding embeddings for {model_name}: {e}. Returning default values.")
        return 0.0, [0] * len(queries), np.zeros((len(queries), len(candidates))), 0.0


    if query_emb.size == 0 or cand_emb.size == 0:
        print(f"Warning: Empty embeddings generated for {model_name}. Queries: {len(queries)}, Candidates: {len(candidates)}. Returning default values.")
        return 0.0, [0] * len(queries), np.zeros((len(queries), len(candidates))), 0.0


    scores = cosine_similarity(query_emb, cand_emb)

    if scores.shape[0] != len(queries) or scores.shape[1] != len(candidates):
        print(f"Warning: Score matrix shape mismatch for {model_name}. Expected ({len(queries)}, {len(candidates)}), got {scores.shape}. Returning default values.")
        return 0.0, [0] * len(queries), np.zeros((len(queries), len(candidates))), 0.0

    preds = np.argmax(scores, axis=1) # This is for Top-1 accuracy

    reciprocal_ranks = []
    for i, correct_idx in enumerate(correct_indices):
        # Get the scores for the current query against all candidates
        query_scores = scores[i, :]
        
        # Sort candidates by score in descending order and get their original indices
        # np.argsort returns indices that would sort an array
        # [::-1] reverses it for descending order
        ranked_indices = np.argsort(query_scores)[::-1]
        
        # Find the rank of the correct_idx
        try:
            # np.where returns a tuple of arrays, we need the first element of the first array
            rank = np.where(ranked_indices == correct_idx)[0][0] + 1 # +1 because rank is 1-based
            reciprocal_ranks.append(1.0 / rank)
        except IndexError:
            # correct_idx not found in ranked_indices (shouldn't happen if correct_idx is valid)
            # This can happen if correct_idx is out of bounds for candidates,
            # indicating a data mismatch (e.g., correct_indices points to a candidate not in the pool).
            print(f"WARNING: Correct index {correct_idx} not found in ranked candidates for query {i}. MRR for this query will be 0.")
            reciprocal_ranks.append(0.0)
        
    mrr_score = np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0
    accuracy = np.mean(preds == np.array(correct_indices))
    
    return accuracy, preds.tolist(), scores, mrr_score # Return MRR


def evaluate_and_save_results(
    queries: List[str],
    references: List[str],
    candidate_pool: List[str],
    model_name: str,
    use_fuzzy: bool = False
) -> Tuple[Dict[str, float], pd.DataFrame]:
    start = time.perf_counter()
    process_info = psutil.Process(os.getpid())

    correct_indices = []
    num_references_found = 0
    # Create a mapping for faster lookup in candidate_pool
    candidate_to_index = {candidate: i for i, candidate in enumerate(candidate_pool)}

    for i, ref in enumerate(references):
        if ref in candidate_to_index:
            idx = candidate_to_index[ref]
            correct_indices.append(idx)
            num_references_found += 1
        else:
            correct_indices.append(0) # Default to 0 or handle as an error case for missing reference

    if len(references) > 0:
        print(f"--- DEBUG: {model_name} (Data Alignment) ---")
        print(f"Total references: {len(references)}, References found in candidate_pool: {num_references_found}")
        if num_references_found == 0 and len(references) > 0:
            print("WARNING: Zero references found in candidate_pool. This indicates a significant data preparation issue.")
            if queries and candidate_pool:
                print(f"Sample reference: '{references[0][:100]}...'")
                print(f"Sample candidate: '{candidate_pool[0][:100]}...'")

    predictions_data = []
    accuracy = 0.0
    mrr_score = 0.0
    pred_ids = []
    score_matrix = np.array([]) # Initialize score_matrix for the cosine case

    if use_fuzzy:
        accuracy, pred_ids, mrr_score = compute_top1_accuracy_fuzzy(queries, candidate_pool, correct_indices)
        for i, pred_id in enumerate(pred_ids):
            # Ensure candidate_pool[pred_id] is a valid index, otherwise handle
            predicted_candidate = candidate_pool[pred_id] if 0 <= pred_id < len(candidate_pool) else "N/A_Invalid_Index"
            individual_score = fuzz.token_sort_ratio(queries[i], predicted_candidate)
            predictions_data.append({
                "query": queries[i],
                "actual": references[i],
                "prediction": predicted_candidate,
                "score": individual_score,
                "model": "rapidfuzz",
                "method": "fuzzy",
                "correct": int(pred_id == correct_indices[i])
            })
    else: # This is the cosine similarity block
        if not queries or not candidate_pool:
            print(f"Warning for model {model_name}: Skipping cosine evaluation due to empty queries or candidate pool.")
            accuracy = 0.0
            pred_ids = [0] * len(queries)
            score_matrix = np.zeros((len(queries), len(candidate_pool))) if queries and candidate_pool else np.array([])
            mrr_score = 0.0 # Also set MRR to 0.0 in this case
        else:
            accuracy, pred_ids, score_matrix, mrr_score = compute_top1_accuracy_cosine(queries, candidate_pool, correct_indices, model_name)

        for i, pred_id in enumerate(pred_ids):
            score_value = 0.0 # Initialize, will be updated.
            predicted_candidate = "N/A_Invalid_Index" # Initialize for safety

            try:
                # Validate pred_id before accessing score_matrix and candidate_pool
                if 0 <= pred_id < len(candidate_pool) and i < score_matrix.shape[0] and pred_id < score_matrix.shape[1]:
                    retrieved_score = score_matrix[i, pred_id]
                    score_value = float(retrieved_score)
                    if not np.isscalar(retrieved_score):
                        print(f"WARNING: score_matrix[{i},{pred_id}] returned non-scalar after direct access: {retrieved_score}. This should not happen. Setting score to 0.0.")
                        score_value = 0.0
                    predicted_candidate = candidate_pool[pred_id]
                else:
                    print(f"WARNING: Invalid pred_id={pred_id} or query index={i} for model {model_name}. Skipping score retrieval for this entry. score_matrix shape: {score_matrix.shape}, candidate_pool size: {len(candidate_pool)}. Setting score to 0.0.")
                    score_value = 0.0
                    # If pred_id is invalid, we can't reliably get the candidate,
                    # but if it was 0 from pred_ids initialization due to empty queries/candidates,
                    # we need to ensure candidate_pool is not empty for candidate_pool[0].
                    if candidate_pool: # Only assign if candidate_pool is not empty
                        predicted_candidate = candidate_pool[0] # Fallback to first candidate or "N/A"
                    else:
                        predicted_candidate = "N/A_Empty_Candidate_Pool"

            except IndexError:
                print(f"CRITICAL ERROR (IndexError): {model_name} at i={i}, pred_id={pred_id}. Matrix shape={score_matrix.shape}. Setting score to 0.0.")
                score_value = 0.0
                if candidate_pool: predicted_candidate = candidate_pool[0]
                else: predicted_candidate = "N/A_Empty_Candidate_Pool"
            except Exception as e:
                print(f"UNEXPECTED ERROR: {model_name} at i={i}, pred_id={pred_id}. Error: {e}. Setting score to 0.0.")
                score_value = 0.0
                if candidate_pool: predicted_candidate = candidate_pool[0]
                else: predicted_candidate = "N/A_Empty_Candidate_Pool"

            predictions_data.append({
                "query": queries[i],
                "actual": references[i],
                "prediction": predicted_candidate,
                "score": score_value,
                "model": model_name,
                "method": "cosine",
                "correct": int(pred_id == correct_indices[i])
            })

    end = time.perf_counter()
    avg_time = (end - start) / len(queries) if len(queries) > 0 else 0.0
    mem_usage = process_info.memory_info().rss / 1024 / 1024

    df_preds = pd.DataFrame(predictions_data)

    metrics = {
        "model_name": model_name,
        "top1_accuracy": round(accuracy, 4),
        "mrr_score": round(mrr_score, 4), # ADDED MRR HERE
        "avg_inference_time": round(avg_time, 6),
        "memory_usage_mb": round(mem_usage, 2),
        "method": "fuzzy" if use_fuzzy else "cosine"
    }

    return metrics, df_preds