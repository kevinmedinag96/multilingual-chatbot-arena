from typing import Union,Optional
from dataclasses import dataclass

dict_vals_datatype = Union[list[int],int]
@dataclass
class InferenceArgs:
    comet_dataset_name : str
    comet_datset_description : str
    comet_prompt_template_name : str
    model_name : str
    max_new_tokens : int
    batch_size : Optional[int] = None
    cache : Optional[str] = None

config = InferenceArgs(
    comet_dataset_name="multilingual-chatbot-arena-v7-train",
    comet_datset_description="Challenge: WSDM CUP. Curated-smal-dataset - v7 - Training set 1.",
    comet_prompt_template_name = 'Prompt_template_wsdm_cup_1',
    model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    cache="quantized KV cache: quanto",
    max_new_tokens= 512,
    batch_size=2)

