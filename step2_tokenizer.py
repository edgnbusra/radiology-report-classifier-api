from transformers import AutoTokenizer

MODEL_ID = "dmis-lab/biobert-v1.1"
REVISION = "551ca18efd7f052c8dfa0b01c94c2a8e68bc5488"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)

text = "Small right pneumothorax. No pleural effusion or cardiomegaly."

tokens = tokenizer.tokenize(text)
encoded = tokenizer(text)

print("Metin   :", text)
print("Token   :", tokens)
print("ID'ler  :", encoded["input_ids"])
print("Geri    :", tokenizer.decode(encoded["input_ids"]))
print("Sozluk boyutu:", tokenizer.vocab_size)
