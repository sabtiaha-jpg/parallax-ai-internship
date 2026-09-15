import sys

print("=== Environment Verification ===")
print("Python version:", sys.version)

print("\nChecking imports...")

try:
    import pandas as pd
    print("pandas: OK")
except Exception as e:
    print("pandas: FAILED", e)

try:
    import chromadb
    print("ChromaDB: OK")
except Exception as e:
    print("ChromaDB: FAILED", e)

try:
    import spacy
    print("spaCy: OK")
except Exception as e:
    print("spaCy: FAILED", e)

try:
    from sentence_transformers import SentenceTransformer
    print("sentence-transformers: OK")
except Exception as e:
    print("sentence-transformers: FAILED", e)

try:
    import torch

    print("PyTorch: OK")
    print("CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    else:
        print("GPU: No CUDA-compatible GPU detected")

except Exception as e:
    print("PyTorch/CUDA check: FAILED", e)

print("\n=== Verification Complete ===")