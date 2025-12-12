# build_db.py
import chromadb
from chromadb.utils import embedding_functions
from datasets import load_dataset
import pandas as pd
import shutil
import os

# --- Clean Start ---
# We wipe the old database to ensure a clean merge of both datasets.
if os.path.exists("./chroma_db"):
    print("🧹 Removing old database for a fresh build...")
    shutil.rmtree("./chroma_db")

# 1. Load Dataset A: Qualifire (Hugging Face)
print("⏳ Loading Dataset 1: qualifire/prompt-injections-benchmark...")
hf_dataset = load_dataset("qualifire/prompt-injections-benchmark", split="test") # or "train"
hf_df = pd.DataFrame(hf_dataset)

# Filter for malicious labels
qualifire_prompts = hf_df[hf_df['label'] == 'jailbreak']['text'].tolist()
print(f"   ✅ Found {len(qualifire_prompts)} prompts in Qualifire.")

# 2. Load Dataset B: Local CSV
print("⏳ Loading Dataset 2: jailbreak_prompts.csv...")
try:
    csv_df = pd.read_csv("jailbreak_prompts.csv")
    
    # Check if 'Prompt' column exists
    if "Prompt" in csv_df.columns:
        # Drop empty rows and convert to list
        csv_prompts = csv_df["Prompt"].dropna().tolist()
        print(f"   ✅ Found {len(csv_prompts)} prompts in CSV.")
    else:
        print("   ⚠️ Column 'Prompt' not found in CSV! Skipping CSV data.")
        csv_prompts = []
        
except FileNotFoundError:
    print("   ⚠️ 'jailbreak_prompts.csv' not found in current directory. Skipping.")
    csv_prompts = []

# 3. Merge Datasets
all_malicious_prompts = qualifire_prompts + csv_prompts
# Optional: Remove duplicates to save space
all_malicious_prompts = list(set(all_malicious_prompts))

print(f"📊 Total Unique Malicious Prompts to Index: {len(all_malicious_prompts)}")

# --- Setup ChromaDB ---
print("⏳ Initializing Vector Database...")
client = chromadb.PersistentClient(path="./chroma_db")

emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

collection = client.create_collection(
    name="jailbreak_signatures",
    embedding_function=emb_fn,
    metadata={"hnsw:space": "cosine"}
)

# --- Add to DB ---
print("⏳ Generating embeddings and storing... (This takes a moment)")

batch_size = 200
total_batches = (len(all_malicious_prompts) // batch_size) + 1

for i in range(0, len(all_malicious_prompts), batch_size):
    batch_texts = all_malicious_prompts[i : i + batch_size]
    # Create simple IDs: id_0, id_1, etc.
    batch_ids = [f"id_{j}" for j in range(i, i + len(batch_texts))]
    
    collection.add(
        documents=batch_texts,
        ids=batch_ids
    )
    print(f"   Processed batch {i // batch_size + 1}/{total_batches}")

print(f"🎉 Success! Database built at './chroma_db' with {collection.count()} items.")