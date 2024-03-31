from transformers import (
    AutoModelForCausalLM,
    #    BitsAndBytesConfig,
    AutoTokenizer,
)


model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-v0.1",
    device_map="cpu",  # "auto" for GPU
    #    quantization_config=BitsAndBytesConfig(
    #        load_in_4bit=True,
    #        load_in_4bit_fp32_cpu_offload=True,
    #    )
)


tokenizer = AutoTokenizer.from_pretrained(
    "mistralai/Mistral-7B-v0.1", padding_side="left"
)
model_inputs = tokenizer(
    ["A list of colors: red, blue"],
    return_tensors="pt"
).to("cpu")  # "cuda:0" for CPU

generated_ids = model.generate(**model_inputs)
print(tokenizer.batch_decode(generated_ids, skip_special_tokens=True))


tokenizer.pad_token = tokenizer.eos_token  # Most LLMs don't have a pad token by default
model_inputs = tokenizer(
    [
        "A list of colors: red, blue",
        "Portugal is"
    ],
    return_tensors="pt",
    padding=True,
).to("cpu")
generated_ids = model.generate(**model_inputs)
print(tokenizer.batch_decode(generated_ids, skip_special_tokens=True))
