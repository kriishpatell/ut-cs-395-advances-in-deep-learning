def generate_dataset(output_json: str, oversample: int = 10, temperature: float = 0.6):
    import json

    import torch

    from .cot import CoTModel
    from .data import Dataset, is_answer_valid

    model = CoTModel(checkpoint="HuggingFaceTB/SmolLM2-1.7B-Instruct")

    if model.device == "cuda":
        model.model = model.model.half()

    trainset = Dataset("train")

    questions = [item[0] for item in trainset]
    correct_answers = [item[1] for item in trainset]

    prompts = [model.format_prompt(q) for q in questions]

    generations = model.batched_generate(
        prompts, num_return_sequences=oversample, temperature=temperature
    )

    dataset = []
    max_per_question = 3  
    covered = 0
    for question, correct, candidates in zip(questions, correct_answers, generations):
        kept = 0
        for candidate in candidates:
            parsed = model.parse_answer(candidate)
            if parsed == parsed and is_answer_valid(parsed, correct):  
                dataset.append([question, correct, candidate])
                kept += 1
                if kept >= max_per_question:
                    break
        if kept > 0:
            covered += 1

    with open(output_json, "w") as f:
        json.dump(dataset, f, indent=2)

    print(
        f"Generated {len(dataset)} examples covering {covered} / {len(questions)} questions."
    )


if __name__ == "__main__":
    from fire import Fire

    Fire(generate_dataset)