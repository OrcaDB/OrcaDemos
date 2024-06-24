from typing import Any, List, Optional

import dspy
from datasets import Dataset
from ragas import evaluate
from ragas.evaluation import Result
from ragas.metrics import (  # TODO: Extend this
    answer_correctness,
    answer_relevancy,
    answer_similarity,
    context_precision,
    context_recall,
    faithfulness,
)


class AssessQA(dspy.Signature):
    """Assess if the answer is derived from the input context given an input question."""

    question = dspy.InputField(desc="Input Question")
    context = dspy.InputField(desc="Input Context")
    answer = dspy.InputField(desc="Input Answer")
    assessment = dspy.OutputField(desc="Return 1 or 0. 1 if answer is derived from context else 0")


class EvalSuite:  # TODO: may become an abstract class in future as we evolve the evaluation technique
    def __init__(self) -> None:
        """Initializes the evaluation suite with an empty list for metrics."""
        self.mem_metrics: List = []
        self.ragas_res: Optional[Result] = None

    def eval(self, dataset: Dataset, eval_memory: bool = False, eval_gen: bool = False, **kwargs):
        """Evaluate the performance of a model using RAGAS library on a given dataset.

        Args:
            dataset (Dataset): HuggingFace Dataset of the type [question: list[str], contexts: list[list[str]], answer: list[str], ground_truth: list[str]]
            eval_memory (bool, optional): If True, evaluate memory-related metrics. Defaults to True.
            eval_gen (bool, optional): If True, evaluate generation-related metrics. Defaults to False.

        Raises:
            TypeError: If the provided dataset is not an instance of HuggingFace's Dataset.
            ValueError: If neither eval_memory nor eval_gen is set to True.

        Returns:
            The result of the evaluation as returned by the RAGAS evaluate function.
        """
        if not isinstance(dataset, Dataset):
            raise TypeError(
                "Expected the dataset to be an instance of HuggingFace's Dataset. "
                "Refer to: https://huggingface.co/docs/datasets/loading_datasets.html"
            )

        if not eval_memory and not eval_gen:
            raise ValueError("At least one of eval_memory or eval_gen must be True.")

        # Setup metrics based on evaluation flags
        self.mem_metrics.clear()

        if not eval_memory and not eval_gen:
            raise ValueError("Either eval_memory or eval_gen needs to be set to True")

        if eval_memory:
            self.mem_metrics.extend([faithfulness, context_precision, context_recall])
        if eval_gen:
            self.mem_metrics.extend([answer_similarity, answer_correctness, answer_relevancy])

        self.ragas_res = evaluate(dataset, metrics=self.mem_metrics, **kwargs)

        return self.ragas_res

    def convert(self, data: Any, **kwargs):
        """Transform data to different format

        Raises:
            TypeError: If the provided data to export is not compatible

        Returns:
            data export to another format

        Args:
            data (Any): result to export
        """

        if isinstance(data, Result):
            return data.to_pandas(**kwargs)
        else:
            raise TypeError("Data type not supported for export.")


assess_qa = dspy.ChainOfThought(AssessQA)
