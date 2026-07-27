import json
from pathlib import Path

from .base_llm import BaseLLM
from .sft import TokenizedDataset, test_model


def load() -> BaseLLM:
    from pathlib import Path

    from peft import PeftModel

    model_name = "rft_model"
    model_path = Path(__file__).parent / model_name

    llm = BaseLLM()
    llm.model = PeftModel.from_pretrained(llm.model, model_path).to(llm.device)
    llm.model.eval()

    return llm

class RFTDataset:
    """Loads the RFT json file: entries of [question, answer, reasoning]."""

    def __init__(self, path: str):
        with open(path) as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def rft_format_example(question: str, answer: float, reasoning: str) -> dict[str, str]:
    """
    Train the model to reproduce the chain-of-thought reasoning that ends in an
    <answer></answer> tag.
    """
    return {
        "question": question,
        "answer": reasoning,
    }


def train_model(
    output_dir: str,
    **kwargs,
):
    # Reuse much of the SFT code here
    from peft import LoraConfig, get_peft_model
    from transformers import Trainer, TrainingArguments

    rft_path = Path(__file__).parent.parent / "data" / "rft.json"

    llm = BaseLLM()

    # Slightly larger LoRA adapter is allowed for RFT (submission < 50MB).
    lora_config = LoraConfig(
        r=16,
        lora_alpha=64,
        target_modules="all-linear",
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(llm.model, lora_config)

    if llm.device == "cuda":
        model.enable_input_require_grads()

    model.print_trainable_parameters()

    train_dataset = TokenizedDataset(llm.tokenizer, RFTDataset(str(rft_path)), rft_format_example)

    training_args = TrainingArguments(
        output_dir=output_dir,
        logging_dir=output_dir,
        report_to="tensorboard",
        num_train_epochs=5,
        per_device_train_batch_size=32,
        learning_rate=2e-4,
        gradient_checkpointing=True,
        save_strategy="epoch",
        save_total_limit=1,
        logging_steps=10,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )

    trainer.train()

    trainer.save_model(output_dir)
    model.save_pretrained(output_dir)

    test_model(output_dir)


if __name__ == "__main__":
    from fire import Fire

    Fire({"train": train_model, "test": test_model, "load": load})