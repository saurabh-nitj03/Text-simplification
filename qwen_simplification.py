
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import evaluate

# --- Configuration ---
# <<< IMPORTANT >>>
# TODO: Replace with the absolute path to your downloaded Qwen model directory.
# For example: "d:/models/qwen/Qwen-1_8B-Chat"
MODEL_PATH = "YOUR_LOCAL_QWEN_MODEL_PATH"

# <<< IMPORTANT >>>
# TODO: Replace with the absolute path to your local dataset file.
# For example: "d:/data/simplification_dataset.jsonl"
DATASET_PATH = "YOUR_LOCAL_DATASET_PATH"

# TODO: Adjust these based on your dataset's structure.
# The names of the columns in your dataset that contain the complex and reference simple sentences.
COMPLEX_COLUMN = "complex"
SIMPLE_COLUMN = "simple"

# --- 1. Load Model and Tokenizer ---
print(f"Loading model from: {MODEL_PATH}")
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map="auto", trust_remote_code=True).eval()
    print("Model and tokenizer loaded successfully.")
except Exception as e:
    print(f"Error loading the model: {e}")
    print("Please ensure that MODEL_PATH is set correctly to your local Qwen model directory.")
    exit()

# --- 2. Load Dataset ---
print(f"Loading dataset from: {DATASET_PATH}")
try:
    # Assuming the dataset is a jsonl file. If it's a different format like csv,
    # change "json" to "csv".
    dataset = load_dataset("json", data_files=DATASET_PATH)
    # Using the 'train' split as an example. Change if your split is named differently.
    test_dataset = dataset['train']
    print("Dataset loaded successfully.")
except Exception as e:
    print(f"Error loading the dataset: {e}")
    print("Please ensure that DATASET_PATH is set correctly and the file format is correct.")
    exit()

# --- 3. Text Simplification Function ---
def simplify_text(complex_text):
    """
    Generates a simplified version of the text using the Qwen model.
    """
    prompt = f"Simplify the following sentence: {complex_text}"
    
    # The model.chat() method is specific to Qwen models and handles the prompt formatting.
    try:
        response, history = model.chat(tokenizer, prompt, history=None)
        return response
    except Exception as e:
        print(f"An error occurred during text generation: {e}")
        return f"Error: Could not generate simplification. Details: {e}"


# --- 4. Perform Simplification and Evaluation ---
predictions = []
references = []

print("\n--- Starting Text Simplification ---")
# Limiting to the first 10 examples for a quick test.
# Remove or adjust `[:10]` to process the entire dataset.
for i, example in enumerate(test_dataset.select(range(min(10, len(test_dataset))))):
    complex_sentence = example[COMPLEX_COLUMN]
    reference_sentence = example[SIMPLE_COLUMN]

    print(f"\nExample {i+1}:")
    print(f"  Original : {complex_sentence}")
    
    simplified_sentence = simplify_text(complex_sentence)
    print(f"  Simplified: {simplified_sentence}")
    print(f"  Reference: {reference_sentence}")

    predictions.append(simplified_sentence)
    references.append(reference_sentence)

# --- 5. Evaluate Performance ---
if predictions and references:
    print("\n--- Evaluating Performance ---")
    try:
        rouge = evaluate.load('rouge')
        results = rouge.compute(predictions=predictions, references=references)
        
        print("\nROUGE Scores:")
        print(f"  rouge1: {results['rouge1']:.4f}")
        print(f"  rouge2: {results['rouge2']:.4f}")
        print(f"  rougeL: {results['rougeL']:.4f}")
        print(f"  rougeLsum: {results['rougeLsum']:.4f}")

    except Exception as e:
        print(f"\nCould not compute ROUGE scores. Error: {e}")
else:
    print("\nNo predictions were generated, skipping evaluation.")

print("\n--- Script Finished ---")
