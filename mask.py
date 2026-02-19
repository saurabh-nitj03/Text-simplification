# pip install transformers torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForMaskedLM.from_pretrained("bert-base-uncased")
model.eval()

def generate_candidates(sentence, index, topk=10):
    toks = tokenizer.tokenize(sentence)
    # replace token at index with [MASK] (index uses tokenized wordpiece index)
    toks[index] = '[MASK]'
    inp = tokenizer.convert_tokens_to_ids(toks)
    input_ids = torch.tensor([tokenizer.build_inputs_with_special_tokens(inp)])
    mask_pos = toks.index('[MASK]')
    with torch.no_grad():
        logits = model(input_ids)[0]
    probs = torch.softmax(logits[0, mask_pos], dim=-1)
    top = torch.topk(probs, topk).indices.tolist()
    return [tokenizer.decode([t]).strip() for t in top]

# Example (note: index must be tokenized index — for a full system you'd map word -> token indices)
sent = "The *culprit* was arrested."
# you would locate token index for "culprit" in the tokenized sentence, then call generate_candidates(...)