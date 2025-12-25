import pandas as pd
import logging
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
    """Dataset for TruthfulQA generation evaluation (CSV ONLY)."""

    def __init__(self, split="validation", offline=False, csv_path=None):
        """
        Loads the dataset directly into a Python list, bypassing HuggingFace datasets entirely.
        """
        if csv_path is None:
             raise ValueError("You must provide `csv_path` to run this dataset version.")

        logger.info(f"Loading TruthfulQA dataset directly from CSV: {csv_path}")
        
        df = pd.read_csv(csv_path)

        df = df.fillna("")

        column_map = {
            "Question": "question",
            "Best Answer": "best_answer",
            "Correct Answers": "correct_answers",
            "Incorrect Answers": "incorrect_answers"
        }
        df = df.rename(columns=column_map)

        def split_semicolon(val):
            if isinstance(val, str):
                return [x.strip() for x in val.split(';') if x.strip()]
            return []

        self.data = []
        logger.info("Processing rows...")
        
        for i, row in df.iterrows():
            q = row.get("question", "")
            ba = row.get("best_answer", "")
            
            ca_raw = row.get("correct_answers", "")
            ia_raw = row.get("incorrect_answers", "")
            
            # Store in our list
            self.data.append(TruthfulQAInstance(
                question=str(q),
                best_answer=str(ba),
                correct_answers=split_semicolon(ca_raw),
                incorrect_answers=split_semicolon(ia_raw),
                instance_id=i
            ))
            
        logger.info(f"Successfully loaded {len(self.data)} items into memory.")

    def __iter__(self):
        for item in tqdm(self.data, desc="Evaluating TruthfulQA"):
            yield item

    def __len__(self):
        return len(self.data)

    @property
    def schema(self):
        return TruthfulQAInstance