
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

tuned_model_path = "qlora-mistral-law"
base_model_name = "mistralai/Mistral-7B-v0.1"

tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(tuned_model_path, device_map="auto")

generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=200,
    do_sample=True,
    temperature=0.7
)

def ask(question: str):
    prompt = f"### Question:\n{question}\n\n### Answer:\n"
    output = generator(prompt)
    return output[0]['generated_text'].split('### Answer:')[-1].strip()

if __name__ == '__main__':
    q = "What is India according to the Union and its Territory?"
    print("Q:", q)
    print("A:", ask(q))
