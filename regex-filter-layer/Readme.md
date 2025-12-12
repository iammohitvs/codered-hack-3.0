
# AI Semantic Sanitizer

A robust, semantic-based text sanitizer designed to detect dangerous, harmful, or prohibited content in AI prompts and outputs. Unlike simple keyword filters, this tool uses **Natural Language Processing (NLP)** and **Fuzzy Matching** to catch obfuscated threats, leetspeak, and grammatical variations.

## Features

* **Context Aware:** Uses **SpaCy** lemmatization to catch grammatical variations (e.g., detects "killing" as "kill").
* **Obfuscation Proof:** Uses **RapidFuzz** to detect fuzzy matches (e.g., "h4ck", "murd3r", "assshole").
* **Normalization Layer:** Automatically strips separation characters (`k-i-l-l`) and translates leetspeak (`byp4ss` -> `bypass`).
* **Fully Configurable:** Supports merging custom blocklists or completely replacing the default vocabulary via JSON.

## Installation

1. **Install Python Dependencies:**
   ```bash
   pip install spacy rapidfuzz
````

2.  **Download the NLP Model:**
    This script requires the English language model for SpaCy.
    ```bash
    python -m spacy download en_core_web_sm
    ```

## Usage

Run the script from the command line by providing a text file to scan.

### Basic Scan

Runs the sanitizer using the built-in default vocabulary.

```bash
python reg.py input.txt
```

### Merge Custom Words

Adds specific words to the default list (good for domain-specific blocking).

```bash
python reg.py input.txt --merge my_custom_words.json
```

### Replace Vocabulary

Ignores the built-in defaults entirely and uses *only* the provided JSON file (good for strict or non-English use cases).

```bash
python reg.py input.txt --replace full_policy.json
```

## Configuration (JSON)

Configuration files must be valid JSON dictionaries where:

  * **Key:** The dangerous word (lowercase).
  * **Value:** A risk score between `0.0` (safe) and `1.0` (dangerous).

**Example `words.json`:**

```json
{
    "malware": 0.9,
    "ransomware": 1.0,
    "competitor_name": 0.5
}
```

## How It Works

The sanitizer processes text in three stages to ensure robustness:

1.  **Normalization & Cleaning:**
      * Translates leetspeak (e.g., `4` -\> `a`, `$` -\> `s`).
      * Strips punctuation to detect separated attacks (e.g., `k.n.i.f.e` -\> `knife`).
2.  **Lemmatization (SpaCy):**
      * Converts words to their root form.
      * *Example:* "The server is **dying**" -\> checks for "**die**".
3.  **Fuzzy Matching (RapidFuzz):**
      * Calculates a similarity ratio for every word.
      * catches typos and evasive spellings (e.g., "att@ck" or "h.a.c.k").
      * *Threshold:* Matches must be \>85% similar to trigger a flag.

## Output Example

```text
----------------------------------------
Scanning file: prompt_injection.txt
----------------------------------------
RISK SCORE:      1.00 / 1.0
TRIGGERS FOUND:  ['bypass', "exploit (matched: 'expl0it')"]
----------------------------------------
STATUS: HIGH RISK - BLOCK CONTENT
```

## License

[Insert your license here, e.g., MIT, Proprietary, etc.]

```
```