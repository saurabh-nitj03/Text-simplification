"""
Hybrid Lexical Simplification Pipeline

Steps:
1. Generate semantic neighbors using CBOW (Word2Vec)
2. Filter candidates using word frequency (simplicity filter)
3. Re-rank top candidates using DistilBERT MLM probability
4. Replace best candidate in sentence
"""

import torch
import torch.nn.functional as F
from transformers import DistilBertTokenizer, DistilBertForMaskedLM
from gensim.models import KeyedVectors
from wordfreq import zipf_frequency

# ===============================
# CONFIGURATION
# ===============================

TOP_CANDIDATES_CBow = 20
TOP_CANDIDATES_AFTER_FILTER = 5
ALPHA = 0.7  # weight for contextual score
BETA = 0.3   # weight for simplicity score

# ===============================
# LOAD MODELS
# ===============================

print("\n🔹 Loading Word2Vec (CBOW) model...")
# Download GoogleNews-vectors-negative300.bin beforehand
# https://code.google.com/archive/p/word2vec/
w2v_model = KeyedVectors.load_word2vec_format(
    "GoogleNews-vectors-negative300.bin",
    binary=True
)

print("✔ Word2Vec loaded")

print("\n🔹 Loading DistilBERT MLM...")
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
mlm_model = DistilBertForMaskedLM.from_pretrained("distilbert-base-uncased")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mlm_model.to(device)
mlm_model.eval()

print("✔ DistilBERT loaded\n")


# ===============================
# STEP 1: CBOW Candidate Generation
# ===============================

def generate_cbow_candidates(word):
    print(f"\n🔹 Generating CBOW neighbors for: '{word}'")

    if word not in w2v_model:
        print("⚠ Word not in vocabulary")
        return []

    neighbors = w2v_model.most_similar(word, topn=TOP_CANDIDATES_CBow)

    candidates = [w for w, _ in neighbors]

    print("Top CBOW candidates:")
    print(candidates)

    return candidates


# ===============================
# STEP 2: Frequency Filtering
# ===============================

def filter_by_frequency(original_word, candidates):
    print("\n🔹 Filtering by simplicity (frequency)...")

    original_freq = zipf_frequency(original_word, "en")
    print(f"Original word frequency: {original_freq:.2f}")

    simpler_candidates = []

    for word in candidates:
        freq = zipf_frequency(word, "en")

        # Keep only words that are more frequent (simpler)
        if freq > original_freq:
            simpler_candidates.append((word, freq))

    # Sort by frequency descending
    simpler_candidates.sort(key=lambda x: x[1], reverse=True)

    # Keep top 5
    simpler_candidates = simpler_candidates[:TOP_CANDIDATES_AFTER_FILTER]

    print("Filtered simpler candidates:")
    for word, freq in simpler_candidates:
        print(f"{word} (freq={freq:.2f})")

    return [word for word, _ in simpler_candidates]


# ===============================
# STEP 3: DistilBERT Re-ranking
# ===============================

def rerank_with_mlm(sentence, target_word, candidates):
    print("\n🔹 Re-ranking with DistilBERT MLM...")

    masked_sentence = sentence.replace(target_word, tokenizer.mask_token, 1)
    inputs = tokenizer(masked_sentence, return_tensors="pt").to(device)

    mask_index = torch.where(
        inputs["input_ids"] == tokenizer.mask_token_id
    )[1]

    with torch.no_grad():
        outputs = mlm_model(**inputs)

    logits = outputs.logits
    mask_logits = logits[0, mask_index, :]
    probs = F.softmax(mask_logits, dim=-1)

    ranked = []

    for word in candidates:
        token_id = tokenizer.convert_tokens_to_ids(word)

        # Skip multi-token words
        if token_id == tokenizer.unk_token_id:
            continue

        lm_score = probs[0, token_id].item()
        simplicity_score = zipf_frequency(word, "en")

        final_score = ALPHA * lm_score + BETA * simplicity_score

        ranked.append((word, lm_score, simplicity_score, final_score))

    ranked.sort(key=lambda x: x[3], reverse=True)

    print("\nRe-ranked candidates:")
    for word, lm, simp, final in ranked:
        print(f"{word} | LM={lm:.4f} | Simplicity={simp:.2f} | Final={final:.4f}")

    return ranked


# ===============================
# STEP 4: Replace Best Candidate
# ===============================

def simplify_sentence(sentence, target_word):
    print("\n====================================")
    print("Original sentence:", sentence)

    cbow_candidates = generate_cbow_candidates(target_word)

    if not cbow_candidates:
        print("⚠ No CBOW candidates found")
        return sentence

    filtered_candidates = filter_by_frequency(target_word, cbow_candidates)

    if not filtered_candidates:
        print("⚠ No simpler candidates found")
        return sentence

    ranked_candidates = rerank_with_mlm(sentence, target_word, filtered_candidates)

    if not ranked_candidates:
        print("⚠ No valid MLM-ranked candidates")
        return sentence

    best_word = ranked_candidates[0][0]

    simplified = sentence.replace(target_word, best_word, 1)

    print("\n✅ Final Simplified Sentence:")
    print(simplified)

    return simplified


# ===============================
# TEST EXAMPLE
# ===============================

# sentence = "The culprit was apprehended."
# target_word = "apprehended"

sentences = [
    ("The culprit was apprehended.", "apprehended"),
    ("The project was abruptly terminated.", "terminated"),
    ("She exhibited remarkable resilience.", "remarkable"),
    ("The corporation initiated a comprehensive restructuring process.","comprehensive"),
    ("His intentions were ambiguous.","His intentions were ambiguous."),
    ("We must evacuate the premises immediately.","evacuate"),
    ("We must evacuate the premises immediately.","premises"),
    ("That explanation is completely absurd.","absurd"),
    ("I cannot tolerate this outrageous behavior.","tolerate"),
    ("The scientist formulated a groundbreaking hypothesis.","groundbreaking"),
    ("The scientist formulated a groundbreaking hypothesis.","hypothesis"),
    ("The project was abruptly terminated due to financial constraints.","abruptly"),
    ("The committee unanimously endorsed the controversial legislation.","endorsed"),
    ("The committee unanimously endorsed the controversial legislation.","legislation")
]

for sentence, target_word in sentences:
    simplify_sentence(sentence,target_word)

# simplify_sentence(sentence, target_word)
