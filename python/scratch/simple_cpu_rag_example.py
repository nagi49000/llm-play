from transformers import (
    AutoTokenizer,
    RagRetriever,
    RagModel
)
import torch

tokenizer = AutoTokenizer.from_pretrained("facebook/rag-token-base")
retriever = RagRetriever.from_pretrained(
    "facebook/rag-token-base",
    index_name="exact",
    use_dummy_dataset=True,
)
# initialize with RagRetriever to do everything in one forward call
model = RagModel.from_pretrained(
    "facebook/rag-token-base",
    retriever=retriever,
    device_map="cpu",
)

inputs = tokenizer(
    "How many people live in Paris?",
    return_tensors="pt"
).to("cpu")
outputs = model(input_ids=inputs["input_ids"])
