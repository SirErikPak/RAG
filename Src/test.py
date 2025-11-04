# import torch
# import torch.nn.functional as F
# from transformers import AutoTokenizer, AutoModel

# tokenizer = AutoTokenizer.from_pretrained("/Users/sir/Downloads/HuggingFace/sentence_transformer/nomic-embed-code")
# model = AutoModel.from_pretrained("/Users/sir/Downloads/HuggingFace/sentence_transformer/nomic-embed-code")

# def last_token_pooling(hidden_states, attention_mask):
#     sequence_lengths = attention_mask.sum(-1) - 1
#     return hidden_states[torch.arange(hidden_states.shape[0]), sequence_lengths]

# queries = ['Represent this query for searching relevant code: Calculate the n-th factorial']
# codes = ['def fact(n):\n if n < 0:\n  raise ValueError\n return 1 if n == 0 else n * fact(n - 1)']
# code_snippets = queries + codes

# encoded_input = tokenizer(code_snippets, padding=True, truncation=True, return_tensors='pt')
# model.eval()
# with torch.no_grad():
#     model_output = model(**encoded_input)[0]

# embeddings = last_token_pooling(model_output, encoded_input['attention_mask'])
# embeddings = F.normalize(embeddings, p=2, dim=1)
# print(embeddings.shape)

# similarity = F.cosine_similarity(embeddings[0], embeddings[1], dim=0)
# print(similarity)


from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

queries = ['Calculate the n-th factorial']
code_snippets = ['Do not calculate the n-th factorial']

model = SentenceTransformer("/Users/sir/Downloads/HuggingFace/sentence_transformer/intfloat_e5-large-v2")
query_emb = model.encode(queries, prompt_name="query")
code_emb = model.encode(code_snippets)

similarity = model.similarity(query_emb[0], code_emb[0])
print(similarity)
print(query_emb.shape)