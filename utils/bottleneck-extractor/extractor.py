# extractor.py
import chromadb
from chromadb.utils import embedding_functions
import re
import os

# --- Setup ---
client = chromadb.PersistentClient(path="./chroma_db")
emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)
collection = client.get_collection(name="jailbreak_signatures", embedding_function=emb_fn)

def split_into_sentences(text):
    sentences = re.split(r'(?<=[.!?]) +', text)
    final_chunks = []
    for s in sentences:
        if len(s.split()) > 200: 
            words = s.split()
            for i in range(0, len(words), 200):
                final_chunks.append(" ".join(words[i:i+200]))
        else:
            final_chunks.append(s)
    return [s.strip() for s in final_chunks if s.strip()]

def analyze_prompt_with_score(user_prompt):
    print(f"\n🔍 Analyzing ({len(user_prompt)} chars)...")
    
    # Track the "Worst" (Min) Distance found across all checks
    # Start with 1.0 (Safe/No match)
    min_distance_found = 1.0
    detected_details = []

    # --- CHECK 1: Whole Prompt ---
    whole_results = collection.query(query_texts=[user_prompt], n_results=1)
    if whole_results['distances'][0]:
        dist = whole_results['distances'][0][0]
        # Check against "Whole Prompt" Threshold
        if dist < 0.45:
            min_distance_found = min(min_distance_found, dist)
            detected_details.append({
                "text": "ENTIRE PROMPT",
                "distance": dist,
                "score": 1 - dist
            })

    # --- CHECK 2: Sentences ---
    sentences = split_into_sentences(user_prompt)
    if sentences:
        results = collection.query(query_texts=sentences, n_results=1)
        SENTENCE_THRESHOLD = 0.65
        
        for i, sentence in enumerate(sentences):
            if not results['distances'][i]: continue
            dist = results['distances'][i][0]
            
            if dist < SENTENCE_THRESHOLD:
                # Update our "Worst Case" tracker
                min_distance_found = min(min_distance_found, dist)
                detected_details.append({
                    "text": sentence,
                    "distance": dist,
                    "score": 1 - dist
                })

    # --- Calculate Final Score ---
    # If no triggers were found, score is 0.
    # If triggers found, score is (1 - lowest_distance)
    if not detected_details:
        final_score = 0.0
    else:
        # Clamp score between 0 and 1 just in case
        final_score = max(0.0, min(1.0, 1.0 - min_distance_found))

    return final_score, detected_details

# --- MAIN LOOP ---
if __name__ == "__main__":
    print("--- Qualifire Jailbreak Scorer ---")
    print("Commands: [paste text] | 'file' | 'exit'")
    
    while True:
        try:
            command = input("\n>> Enter Command or Prompt: ").strip()
            if command.lower() in ['exit', 'quit']: break
            
            user_text = ""
            if command.lower() == "file":
                if os.path.exists("input.txt"):
                    with open("input.txt", "r") as f: user_text = f.read()
                    print(f"📄 Read 'input.txt'")
                else:
                    print("❌ 'input.txt' missing.")
                    continue
            else:
                user_text = command
            
            if user_text:
                score, details = analyze_prompt_with_score(user_text)
                
                # --- VISUAL OUTPUT FOR HACKATHON JUDGES ---
                print("\n" + "="*40)
                print(f"🛑 RISK SCORE: {score:.4f} / 1.0")
                
                # Visual Bar Graph
                bar_len = int(score * 20)
                bar = "█" * bar_len + "░" * (20 - bar_len)
                
                # Color coding (ANSI escape codes)
                if score > 0.7:
                    color = "\033[91m" # Red
                    status = "CRITICAL"
                elif score > 0.4:
                    color = "\033[93m" # Yellow
                    status = "SUSPICIOUS"
                else:
                    color = "\033[92m" # Green
                    status = "SAFE"
                reset = "\033[0m"

                print(f"LEVEL: {color}{status} [{bar}]{reset}")
                print("="*40)

                if details:
                    print(f"\nfound {len(details)} malicious segments:")
                    for d in details:
                        # Print the specific phrase and its individual score
                        print(f" - [{d['score']:.2f}] '{d['text'][:50]}...'")
                else:
                    print("✅ No malicious signatures detected.")

        except KeyboardInterrupt:
            break