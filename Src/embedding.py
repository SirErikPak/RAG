import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F

# Load the model directly from the library
local_path = "/Users/sir/Downloads/HuggingFace/sentence_transformer/all-mpnet-base-v2"

# Global device and model initialization for performance
# This runs once when the module is imported, avoiding repeated checks
# --- OPTIMIZATION: Prioritize MPS, then CUDA, then CPU ---
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")
print(f"Using device: {DEVICE}")

# --- 'model' and 'tokenizer' are defined ---
try:
    # Example: Load a common small embedding model (replace with your actual model)
    tokenizer = AutoTokenizer.from_pretrained(local_path)
    
    # Load model and move it to the determined device
    model = AutoModel.from_pretrained(local_path).to(DEVICE)
        
except Exception as e:
    print(f"Warning: Could not load dummy model/tokenizer for demo: {e}. Define them before use.")
    # imports fail in the execution environment
    tokenizer = None 
    model = None


def mean_pooling_with_mask(model_output, attention_mask):
    # model_output[0] is the last_hidden_state (batch_size, sequence_length, hidden_size)
    token_embeddings = model_output.last_hidden_state
    
    # 1. Expand attention mask to match the embedding dimension
    # (batch_size, sequence_length, 1) -> (batch_size, sequence_length, hidden_size)
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    
    # 2. Sum the real tokens (zeros out padding tokens)
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    
    # 3. Get the number of real tokens per sentence (must be at least 1 to avoid division by zero)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    # 4. Divide the sum by the number of real tokens to get the average
    return sum_embeddings / sum_mask


def embed_texts(texts: list[str]) -> np.ndarray:
    """
    Generates embeddings for a list of texts in an optimized, batch-aware manner.

    Optimizations:
    1. Processes a batch (list) of texts at once.
    2. Uses torch.float16 (half-precision) for faster GPU calculation.
    3. Manages tensors on the detected DEVICE (CUDA or CPU).

    Args:
        texts: A list of strings to embed.

    Returns:
        A NumPy array where each row is the embedding vector for a corresponding text.
    """
    if not tokenizer or not model:
        raise RuntimeError("Model and tokenizer must be initialized before calling embed_texts.")

    # 1. Tokenization and Device Transfer (Batch Processing)
    inputs = tokenizer(
        texts, 
        return_tensors="pt", 
        truncation=True, 
        padding=True,
        max_length=512 # Good practice to specify max length
    ).to(DEVICE)
    
    # 2. Model Inference
    with torch.no_grad():
        outputs = model(**inputs)

    # 3. Mean Pooling and Conversion to NumPy
    # Use mean pooling of the last hidden state over the sequence length dimension (dim=1)
    embeddings = mean_pooling_with_mask(outputs, inputs['attention_mask'])
    # embeddings = outputs.last_hidden_state.mean(dim=1)

    # 4. L2 NORMALIZATION
    # Normalize the vectors to have a length of 1 along the feature dimension (dim=1)
    normalized_embeddings = F.normalize(embeddings, p=2, dim=1)
    
    # 5. Conversion to NumPy
    return normalized_embeddings.cpu().numpy()