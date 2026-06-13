from transformers import AutoTokenizer
from pathlib import Path
from app.config import Config

def test_tokenizer():
    model_path = Path(Config.RELATION_MODEL_DIR)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    t1 = "claim text"
    t2 = "evidence text"
    
    # Method 1: Manual string with [SEP]
    manual_text = f"{t1} [SEP] {t2}"
    tokens_manual = tokenizer.tokenize(manual_text)
    ids_manual = tokenizer.convert_tokens_to_ids(tokens_manual)
    
    # Method 2: Tokenizer pair
    encoded = tokenizer(t1, t2)
    tokens_pair = tokenizer.convert_ids_to_tokens(encoded["input_ids"])
    
    print("Manual concatenation tokens:")
    print(tokens_manual)
    print("Manual concatenation IDs:")
    print(ids_manual)
    
    print("\nProper pair encoding tokens:")
    print(tokens_pair)
    print("Proper pair encoding IDs:")
    print(encoded["input_ids"])
    
    # Check if [SEP] is tokenized as special token
    sep_id = tokenizer.sep_token_id
    print(f"\nSEP token ID: {sep_id}")
    print(f"Is SEP ID in manual IDs? {sep_id in ids_manual}")
    print(f"Is SEP ID in proper IDs? {sep_id in encoded['input_ids']}")

if __name__ == "__main__":
    test_tokenizer()
