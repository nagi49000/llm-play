from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
)
from langchain_core.prompts import PromptTemplate


model_id = "gpt2"
max_new_tokens = 64

pipe = pipeline(
    "text-generation",
    model=AutoModelForCausalLM.from_pretrained(model_id),
    tokenizer=AutoTokenizer.from_pretrained(model_id),
    max_new_tokens=max_new_tokens,
    device_map="cpu",
)
hf = HuggingFacePipeline(
    pipeline=pipe,
    model_kwargs={
        "temperature": 0.1,
        "device_map": "cpu",
    },
)


template = """Question: {question}

Answer: """
prompt = PromptTemplate.from_template(template)

chain = prompt | hf

question = "What is electroencephalography?"

print(chain.invoke({"question": question}))
