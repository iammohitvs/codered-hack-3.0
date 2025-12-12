import sys
import json
import os
import argparse
from typing import Dict, List, Tuple

try:
    import spacy
    from rapidfuzz import process, fuzz
except ImportError:
    print("[!] Error: Missing libraries. Run: pip install spacy rapidfuzz")
    sys.exit(1)

DEFAULT_KEYWORDS = {
    """ We have filled this with words that consist of profanity and vulgarity and can seem to be dangerous """
}

class SemanticSanitizer:
    def __init__(self, merge_file: str = None, replace_file: str = None):
        if not hasattr(self, 'nlp'):
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                print("[!] Error: Model 'en_core_web_sm' not found. Run: python -m spacy download en_core_web_sm")
                sys.exit(1)
            
        self.keywords = self._load_vocabulary(merge_file, replace_file)

    def _load_vocabulary(self, merge_file, replace_file) -> Dict[str, float]:
        keywords = {}

        if replace_file and os.path.exists(replace_file):
            keywords = self._read_json(replace_file)
            return keywords

        keywords = DEFAULT_KEYWORDS.copy()

        if merge_file and os.path.exists(merge_file):
            custom_words = self._read_json(merge_file)
            keywords.update(custom_words)
            
        return keywords

    def _read_json(self, filepath: str) -> Dict[str, float]:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"[!] Warning: Could not read '{filepath}'. Error: {e}")
            return {}

    def analyze(self, text: str) -> Tuple[float, List[str]]:
        leetspeak_map = str.maketrans({
            '0': 'o', '1': 'i', '3': 'e', '4': 'a', 
            '5': 's', '@': 'a', '$': 's', '!': 'i'
        })
        normalized_text = text.translate(leetspeak_map)
        
        cleaned_text = normalized_text.replace("-", "").replace(".", "").replace("_", "")
        
        combined_text = f"{text} {normalized_text} {cleaned_text}"
        
        doc = self.nlp(combined_text)
        
        tokens = [token.lemma_.lower() for token in doc]
        
        total_score = 0.0
        triggered_words = set()

        for bad_word, weight in self.keywords.items():
            
            if bad_word in tokens:
                triggered_words.add(bad_word)
                total_score += weight
                continue 

            match = process.extractOne(bad_word, tokens, scorer=fuzz.ratio)
            if match:
                matched_token, score, index = match
                if score >= 85: 
                    triggered_words.add(f"{bad_word} (matched: '{matched_token}')")
                    total_score += weight

        return min(total_score, 1.0), sorted(list(triggered_words))

def main():
    parser = argparse.ArgumentParser(description="AI Semantic Sanitizer")
    parser.add_argument("filename", help="Path to the input text file")
    parser.add_argument("--merge", help="JSON file to ADD to defaults", default=None)
    parser.add_argument("--replace", help="JSON file to REPLACE defaults", default=None)
    
    args = parser.parse_args()

    sanitizer = SemanticSanitizer(merge_file=args.merge, replace_file=args.replace)

    try:
        with open(args.filename, 'r', encoding='utf-8') as f:
            content = f.read()
    except FileNotFoundError:
        print(f"[!] Error: File '{args.filename}' not found.")
        sys.exit(1)

    print("-" * 40)
    print(f"Scanning file: {args.filename}")
    score, triggers = sanitizer.analyze(content)

    print("-" * 40)
    print(f"RISK SCORE:      {score:.2f} / 1.0")
    print(f"TRIGGERS FOUND:  {triggers}")
    print("-" * 40)

    if score > 0.7:
        print("🔴 STATUS: HIGH RISK - BLOCK CONTENT")
    elif score > 0.3:
        print("🟠 STATUS: MODERATE RISK - FLAGGED FOR REVIEW")
    else:
        print("🟢 STATUS: SAFE")

if __name__ == "__main__":
    main()