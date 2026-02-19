# from transformers import DistilBertTokenizer, DistilBertForMaskedLM
# import torch

# tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
# model = DistilBertForMaskedLM.from_pretrained("distilbert-base-uncased")

# model.eval()

# def mask_word(sentence, target_word):
#     return sentence.replace(target_word, tokenizer.mask_token)
# #
# def generate_substitutes(sentence, top_k=10):
#     inputs = tokenizer(sentence, return_tensors="pt")
    
#     mask_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]
    
#     with torch.no_grad():
#         outputs = model(**inputs)
    
#     logits = outputs.logits
#     mask_token_logits = logits[0, mask_token_index, :]
    
#     top_tokens = torch.topk(mask_token_logits, top_k, dim=1).indices[0].tolist()
#     print(top_tokens)
#     return [tokenizer.decode([token]).strip() for token in top_tokens]


# sentence = "The culprit was apprehended."
# masked = mask_word(sentence, "apprehended")

# candidates = generate_substitutes(masked)
# print(candidates)


from transformers import DistilBertTokenizer, DistilBertForMaskedLM
import torch
import torch.nn.functional as F

# Load model
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertForMaskedLM.from_pretrained("distilbert-base-uncased")
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


def mask_word(sentence, target_word):
    # Replace only first occurrence (safer)
    return sentence.replace(target_word, tokenizer.mask_token, 1)


def generate_substitutes(sentence, top_k=10):
    inputs = tokenizer(sentence, return_tensors="pt").to(device)

    mask_token_index = torch.where(
        inputs["input_ids"] == tokenizer.mask_token_id
    )[1]

    if len(mask_token_index) == 0:
        print("⚠ No [MASK] token found in sentence!")
        return []

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits

    # Extract logits for masked token
    mask_token_logits = logits[0, mask_token_index, :]

    # Convert logits to probabilities
    probs = F.softmax(mask_token_logits, dim=-1)

    # Get top-k tokens
    top_tokens = torch.topk(probs, top_k, dim=1)

    substitutes = []

    for score, token_id in zip(top_tokens.values[0], top_tokens.indices[0]):
        word = tokenizer.decode([token_id]).strip()
        substitutes.append((word, score.item()))

    return substitutes


# ✅ Pass array of sentences
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
    print("\nOriginal:", sentence)

    masked_sentence = mask_word(sentence, target_word)
    print("Masked:", masked_sentence)

    candidates = generate_substitutes(masked_sentence, top_k=10)

    print("Top Substitutes with Scores:")
    for word, score in candidates:
        print(f"{word:<15} | Score: {score:.4f}")


from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
import torch.nn.functional as F

# Load TinyBERT 4-layer model
MODEL_NAME = "prajjwal1/bert-mini"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()


def mask_word(sentence, target_word):
    return sentence.replace(target_word, tokenizer.mask_token, 1)


def generate_substitutes(sentence, top_k=10):
    inputs = tokenizer(sentence, return_tensors="pt").to(device)

    mask_token_index = torch.where(
        inputs["input_ids"] == tokenizer.mask_token_id
    )[1]

    if len(mask_token_index) == 0:
        print("⚠ No [MASK] token found!")
        return []

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    mask_logits = logits[0, mask_token_index, :]

    probs = F.softmax(mask_logits, dim=-1)

    top_tokens = torch.topk(probs, top_k, dim=1)

    substitutes = []

    for score, token_id in zip(top_tokens.values[0], top_tokens.indices[0]):
        word = tokenizer.decode([token_id]).strip()

        # Optional: skip subword fragments
        if word.startswith("##"):
            continue

        substitutes.append((word, score.item()))

    return substitutes


# 🔹 Test sentences
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
    print("\nOriginal:", sentence)

    masked = mask_word(sentence, target_word)
    print("Masked:", masked)

    candidates = generate_substitutes(masked, top_k=10)

    print("Substitutes with Scores:")
    for word, score in candidates:
        print(f"{word:<15} | Score: {score:.4f}")
