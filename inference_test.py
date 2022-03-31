from transformers import T5Tokenizer, T5ForConditionalGeneration

tokenizer = T5Tokenizer.from_pretrained("./results/t5-base/tokenizer/best-f1")
# /checkpoint-31
model = T5ForConditionalGeneration.from_pretrained("./results/t5-base/model/best-f1")

context = """"""
input_ids = tokenizer(f"question: What else reaches the headquarters along with them??  context: {context}", return_tensors="pt").input_ids
outputs = model.generate(input_ids)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
