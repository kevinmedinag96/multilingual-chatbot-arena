from dataclasses import dataclass
from typing import Optional


@dataclass
class DataParams:
    """
    Relevant parameters related to loading and processing the target dataset 
    """
    comet_dataset_name : str
    comet_datset_description : str
    train_validation_test_split : list[float]
    seed : int

@dataclass
class LLMParams:
    model_id : str
    max_seq_length : Optional[int]

@dataclass
class TrainingParams:
    seed : int


llm_params = LLMParams(
    model_id= "Qwen/Qwen2.5-0.5B-Instruct",
    max_seq_length=5000
)

data_params = DataParams(
    comet_dataset_name="multilingual-chatbot-arena-v8-train-1",
    comet_datset_description="Challenge WSDM CUP. Curated-small-dataset-v8 - Training set 1. - Training set 1.",
    train_validation_test_split=[0.7,0.2],
    seed=142
)
