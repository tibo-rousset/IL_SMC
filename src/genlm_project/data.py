import pandas as pd
import logging
from datasets import load_dataset, Dataset as HFDataset
from genlm.eval import Instance, Dataset
from tqdm import tqdm

logger = logging.getLogger(__name__)
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

    def __init__(self, split="validation", offline=False, csv_path=None):
        if offline:
            if csv_path is None:
                raise ValueError("`csv_path` must be provided when `offline=True`.")

            logger.info(f"Loading TruthfulQA dataset from local CSV: {csv_path}")
            df = pd.read_csv(csv_path)

            logger.info("Processing dataset columns...")
            column_map = {
                "Question": "question",
                "Best Answer": "best_answer",
                "Correct Answers": "correct_answers",
                "Incorrect Answers": "incorrect_answers"
            }
            df = df.rename(columns=column_map)

            def split_semicolon(val):
                if isinstance(val, str):
                    # Split by semicolon and strip whitespace
                    return [x.strip() for x in val.split(';') if x.strip()]
                return []

            if "correct_answers" in df.columns:
                df["correct_answers"] = df["correct_answers"].apply(split_semicolon)
            if "incorrect_answers" in df.columns:
                df["incorrect_answers"] = df["incorrect_answers"].apply(split_semicolon)

            self.data = HFDataset.from_pandas(df)
            
        else:
            # Online loading (uses standard HF structure)
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