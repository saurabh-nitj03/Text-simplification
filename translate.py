"""
Back Translation using SMALL100
English → German → English
"""

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# ===============================
# CONFIG
# ===============================

MODEL_NAME = "alirezamsh/small100"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===============================
# LOAD MODEL
# ===============================

print("🔹 Loading SMALL100 model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device)
model.eval()

print("✔ Model Loaded\n")


# ===============================
# TRANSLATION FUNCTION
# ===============================

def translate(texts, src_lang, tgt_lang):
    """
    Translate batch of texts from src_lang to tgt_lang
    """
    tokenizer.src_lang = src_lang

    encoded = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(device)

    forced_bos_token_id = tokenizer.get_lang_id(tgt_lang)

    with torch.no_grad():
        generated_tokens = model.generate(
            **encoded,
            forced_bos_token_id=forced_bos_token_id,
            max_length=128
        )

    return tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)


# ===============================
# BACK TRANSLATION PIPELINE
# ===============================

def back_translate(sentences):
    print("🔹 Step 1: English → German\n")
    german_texts = translate(sentences, src_lang="en", tgt_lang="de")

    for en, de in zip(sentences, german_texts):
        print(f"EN: {en}")
        print(f"DE: {de}\n")

    print("\n🔹 Step 2: German → English\n")
    back_translated = translate(german_texts, src_lang="de", tgt_lang="en")

    return back_translated


# ===============================
# TEST SENTENCES
# ===============================

complex_sentences = [
    "The culprit was apprehended after an extensive investigation.",
    "The project was abruptly terminated due to financial constraints.",
    "She exhibited remarkable resilience during adversity.",
    "The corporation initiated a comprehensive restructuring process.",
    "His intentions were ambiguous."
]

# ===============================
# RUN
# ===============================

results = back_translate(complex_sentences)

print("\n==============================")
print("🔹 Back Translated Results")
print("==============================")

for original, simplified in zip(complex_sentences, results):
    print(f"\nOriginal: {original}")
    print(f"Back-Translated: {simplified}")