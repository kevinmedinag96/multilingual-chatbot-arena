from dataclasses import dataclass
from typing import Optional
import torch
from typing import Union

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
    attn_implementation : Union[bool,str]
    torch_dtype : str = "auto"
    use_cache : bool =False
    device_map : str = "auto"

@dataclass
class TrainingParams:
    output_dir : str
    overwrite_output_dir: bool = True
    do_eval: bool =True
    eval_strategy: str = "steps"
    eval_steps: int =2
    per_device_eval_batch_size: int =2
    per_device_train_batch_size: int =2
    gradient_accumulation_steps: int =1
    gradient_checkpointing: bool =False
    gradient_checkpointing_kwargs: Optional[dict]=None#{"use_reentrant": False},
    torch_empty_cache_steps: Optional[int] = 50
    learning_rate: float = 2.0e-05
    weight_decay: float =0.001
    num_train_epochs: int = 1
    max_steps : Optional[int] = None
    lr_scheduler_type: str = "linear"
    logging_strategy: str ="steps"
    logging_steps: int = 50
    log_level: str ="info"
    save_strategy: str ="steps"
    save_steps: int = 500
    save_total_limit: int =2
    seed: int = 155
    data_seed: int = 142
    fp16: bool =False
    bf16 : bool = False
    load_best_model_at_end: bool =True
    auto_find_batch_size: bool =False
    dataset_text_field: str ='conversation'
    packing: bool =False



@dataclass
class QuantizationParams:
    load_in_4bit : bool = False
    bnb_4bit_quant_type : str = "nf4"
    bnb_4bit_compute_dtype = torch.float16

@dataclass
class LoraParams:
    r: int = 64
    lora_alpha: int = 16
    lora_dropout:float = 0.1
    bias: str = "none"
    task_type : str = "CAUSAL_LM"
    target_modules  = ["q_proj", "k_proj", "v_proj", "o_proj"]


# INIT DATA CLASSES...

llm_params = LLMParams(
    model_id= "Qwen/Qwen2.5-0.5B-Instruct",
    max_seq_length=5000,
    attn_implementation="flash_attention_2"
)

data_params = DataParams(
    comet_dataset_name="multilingual-chatbot-arena-v8-train-1",
    comet_datset_description="Challenge WSDM CUP. Curated-small-dataset-v8 - Training set 1. - Training set 1.",
    train_validation_test_split=[0.7,0.2],
    seed=142
)

quantization_params = QuantizationParams(
    load_in_4bit=True
)


experiment_basename = f"{llm_params.model_id.replace('/','-')}/{data_params.comet_dataset_name}/exp-1"
training_params = TrainingParams(
    output_dir= f"./chkpts/{experiment_basename}",
    bf16=False,
    fp16=True,
    num_train_epochs=1,
    eval_steps=2,
    logging_steps=2,
    #gradient_accumulation_steps=10,
    save_steps=2,
    #gradient_checkpointing=True,
    per_device_eval_batch_size=2,
    per_device_train_batch_size=2,
    #auto_find_batch_size=True,
    #gradient_checkpointing_kwargs={"use_reentrant": False},
    torch_empty_cache_steps=30
    
)

lora_params = LoraParams()