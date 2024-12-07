from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time

# Sample prompts.
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

# Load the model and tokenizer.
model_name = "/WORK/PUBLIC/zhaijd_work/dataset/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Set the sampling parameters.
temperature = 0.8
top_p = 0.95

# Generate texts from the prompts.
outputs = []
start_time = time.perf_counter()
for prompt in prompts:
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        output = model.generate(
            inputs.input_ids,
            max_length=50,  # Adjust as needed
            temperature=temperature,
            top_p=top_p,
            do_sample=True
        )
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    outputs.append((prompt, generated_text))
end_time = time.perf_counter()
print(f"inference time: {end_time-start_time}")

# Print the outputs.
for prompt, generated_text in outputs:
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")