import faiss

index = faiss.read_index("/home/dbd5616/rag/corpus/pubmed/index/facebook/contriever/faiss.index")  # adjust path
print("Number of embeddings:", index.ntotal)
print("Embedding dimension:", index.d)