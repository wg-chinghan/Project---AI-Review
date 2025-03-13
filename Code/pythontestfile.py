import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Use CPU (since CUDA is not available)
device = torch.device("cpu")

# Load Qwen model and tokenizer
model_name = "Qwen/Qwen2.5-0.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

# Test inference
input_text = "My Favourite color is blue"
inputs = tokenizer(input_text, return_tensors="pt").to(device)
outputs = model.generate(**inputs)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("Model Response:", response)
