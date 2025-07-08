import json

INPUT_JSON = "merged.json"      
OUTPUT_JSONL = "law_qa_formatted.jsonl"

def format_example(example):
    prompt = f"### Question:\n{example['question']}\n\n### Answer:\n"
    response = example['answer']
    return {"prompt": prompt, "response": response}


def main():
    with open(INPUT_JSON, 'r', encoding='utf-8') as f:
        data = json.load(f)

    with open(OUTPUT_JSONL, 'w', encoding='utf-8') as fout:
        for entry in data:
            formatted = format_example(entry)
            fout.write(json.dumps(formatted, ensure_ascii=False) + "\n")

    print(f"Wrote formatted data to {OUTPUT_JSONL}")


if __name__ == '__main__':
    main()