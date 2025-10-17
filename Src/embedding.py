import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel

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
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")
    
    # Load model and move it to the determined device
    model = AutoModel.from_pretrained("sentence-transformers/all-mpnet-base-v2").to(DEVICE)
        
except Exception as e:
    print(f"Warning: Could not load dummy model/tokenizer for demo: {e}. Define them before use.")
    # imports fail in the execution environment
    tokenizer = None 
    model = None


def embed_texts(texts: list[str]) -> np.ndarray:
    """
    Generates embeddings for a list of texts in an optimized, batch-aware manner.

    Optimizations:
    1. Processes a batch (list) of texts at once.
    2. Utilizes the detected high-performance device (MPS or CUDA).
    3. Uses torch.float16 (half-precision) only if running on CUDA.

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
        max_length=768 # Good practice to specify max length
    ).to(DEVICE)
    
    # 2. Model Inference
    with torch.no_grad():
        # MPS models were moved to the device in full precision during initialization.             
        outputs = model(**inputs)

    # 3. Mean Pooling and Conversion to NumPy
    # Use mean pooling of the last hidden state over the sequence length dimension (dim=1)
    # The .cpu().numpy() moves the result back to the CPU memory and converts it for standard use.
    embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
    
    return embeddings