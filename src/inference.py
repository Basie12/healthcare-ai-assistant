from unsloth import FastLanguageModel

# Load final DPO model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="../saved_models/final_medical_assistant_dpo",
    max_seq_length=2048,
    load_in_4bit=True,
)

FastLanguageModel.for_inference(model)


def generate(question):
    prompt = f"""Below is an instruction that describes a task.

### Instruction:
You are a knowledgeable medical AI assistant.

### Question:
{question}

### Response:
"""

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    outputs = model.generate(
        **inputs,
        max_new_tokens=300,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    if "### Response:" in response:
        response = response.split("### Response:")[-1].strip()

    return response


if __name__ == "__main__":
    while True:
        question = input("\nAsk a medical question (type 'exit' to quit): ")

        if question.lower() == "exit":
            break

        print("\nAssistant:")
        print(generate(question))