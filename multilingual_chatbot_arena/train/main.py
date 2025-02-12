
import pathlib
import sys
from fire import Fire
root_repo_directory = pathlib.Path().resolve().__str__() + "/multilingual-chatbot-arena"
sys.path.append(root_repo_directory)
from multilingual_chatbot_arena import initialize

import data_preparation
from config import data_params, llm_params

from trl import SFTTrainer,SFTConfig

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, AutoConfig, BitsAndBytesConfig
import torch
from peft import LoraConfig

from pynvml import *
import metrics

def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")


def print_summary(result):
    print(f"Time: {result.metrics['train_runtime']:.2f}")
    print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
    print_gpu_utilization()

def run():
    

    initialize()
    
    #Datasets preparation
    data_obj = data_preparation.Data(data_params,llm_params)

    #quantization config
    quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
    )


    
    #model config
    model_kwargs = dict(
        attn_implementation="flash_attention_2", # set this to True if your GPU supports it (Flash Attention drastically speeds up model computations)
        torch_dtype="auto",
        use_cache=False, # set to False as we're going to use gradient checkpointing
        device_map="auto",
        quantization_config=quantization_config,
    )

       

    #trainer set up

    #training arguments
    training_config = SFTConfig(
        output_dir=f"./chkpts/{llm_params.model_id}-v8",
        overwrite_output_dir=True,
        do_eval=True,
        eval_strategy="steps",
        eval_steps=5,
        per_device_eval_batch_size=2,
        per_device_train_batch_size=2,
        #gradient_accumulation_steps=2,
        #gradient_checkpointing=True,
        #gradient_checkpointing_kwargs={"use_reentrant": False},
        torch_empty_cache_steps=50,
        learning_rate=2.0e-05,
        weight_decay=.001,
        num_train_epochs=3,
        #max_steps=300,
        lr_scheduler_type="cosine",
        logging_strategy="steps",
        logging_steps=50,
        log_level="info",
        save_strategy="steps",
        save_total_limit=2,
        seed = 142,
        data_seed= 142,
        fp16=True,
        load_best_model_at_end=True,
        #auto_find_batch_size=True,
        model_init_kwargs=model_kwargs,
        dataset_text_field='conversation',
        packing=False,
        max_seq_length=llm_params.max_seq_length
    )

    #include lora adapters
    peft_config = LoraConfig(
        r=64,
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    trainer = SFTTrainer(
        model=llm_params.model_id,
        args=training_config,
        train_dataset=data_obj.hf_datasets['train'],
        eval_dataset=data_obj.hf_datasets['validation'],
        processing_class= data_obj.tokenizer,
        peft_config=peft_config,
        compute_metrics=metrics.compute_metrics
    )

    result = trainer.train()

    print_summary(result)
    



if __name__ == "__main__":
    Fire(run)