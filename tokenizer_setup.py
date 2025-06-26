from transformers import AutoTokenizer

def get_tokenizer(model_name="distilbert-base-uncased"):
    return AutoTokenizer.from_pretrained(model_name)

