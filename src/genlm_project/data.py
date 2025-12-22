from datasets import load_dataset
from genlm.eval import Instance, Dataset
from tqdm import tqdm

class TruthfulQAInstance(Instance):
    """Schema for a TruthfulQA instance."""
    question: str
    best_answer: str
    correct_answers: list[str]
    incorrect_answers: list[str]
    instance_id: int 

    def __repr__(self):
        return f"[{self.instance_id}] Q: {self.question} (Best A: {self.best_answer})"

class TruthfulQADataset(Dataset[TruthfulQAInstance]):
    """Dataset for TruthfulQA generation evaluation."""

    def __init__(self, split="validation"):
        self.data = load_dataset("truthful_qa", "generation", split=split)

    def __iter__(self):
        for i, row in enumerate(tqdm(self.data, desc="Evaluating TruthfulQA")):
            yield TruthfulQAInstance(
                question=row["question"],
                best_answer=row["best_answer"],
                correct_answers=row["correct_answers"],
                incorrect_answers=row["incorrect_answers"],
                instance_id=i
            )

    def __len__(self):
        return len(self.data)

    @property
    def schema(self):
        return TruthfulQAInstance