
import pathlib
import sys
from fire import Fire
root_repo_directory = pathlib.Path().resolve().__str__() 
sys.path.append(root_repo_directory)

from multilingual_chatbot_arena import initialize

import data_preparation
from config import data_params, llm_params,quantization_params,training_params,lora_params

from trl import SFTConfig

from transformers import BitsAndBytesConfig
import comet_ml
import torch
from peft import LoraConfig


import metrics

import trainer

from trainer_utils import (
    preprocess_logits_for_metrics,
    print_summary
)

from transformers.integrations.integration_utils import (
    CometCallback
)

def run():   

    initialize()
    
    #Datasets preparation
    data_obj = data_preparation.Data(data_params,llm_params)

    #quantization config
    quantization_config = BitsAndBytesConfig(
            load_in_4bit=quantization_params.load_in_4bit,
            bnb_4bit_quant_type=quantization_params.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=quantization_params.bnb_4bit_compute_dtype,
    )
    
    #model config
    model_kwargs = dict(
        attn_implementation=llm_params.attn_implementation, # set this to True if your GPU supports it (Flash Attention drastically speeds up model computations)
        torch_dtype=llm_params.torch_dtype,
        use_cache=llm_params.use_cache, # set to False as we're going to use gradient checkpointing
        device_map=llm_params.device_map,
        quantization_config=quantization_config
    )       

    #trainer set up

    #training arguments
    training_config = SFTConfig(
        output_dir=training_params.output_dir,
        overwrite_output_dir=training_params.overwrite_output_dir,
        do_eval=training_params.do_eval,
        eval_strategy=training_params.eval_strategy,
        eval_steps=training_params.eval_steps,
        per_device_eval_batch_size=training_params.per_device_eval_batch_size,
        per_device_train_batch_size=training_params.per_device_train_batch_size,
        #gradient_accumulation_steps=training_params.gradient_accumulation_steps,
        #gradient_checkpointing=training_params.checkpointing,
        #gradient_checkpointing_kwargs=training_params.gradient_checkpointing_kwargs,
        torch_empty_cache_steps=training_params.torch_empty_cache_steps,
        learning_rate=training_params.learning_rate,
        weight_decay=training_params.weight_decay,
        num_train_epochs=training_params.num_train_epochs,
        #max_steps=training_params.max_steps,
        #lr_scheduler_type=training_params.lr_scheduler_type,
        logging_strategy=training_params.logging_strategy,
        logging_steps=training_params.logging_steps,
        log_level=training_params.log_level,
        save_strategy=training_params.save_strategy,
        save_steps= training_params.save_steps,       
        save_total_limit=training_params.save_total_limit,
        seed = training_params.seed,
        data_seed= training_params.data_seed,
        fp16=training_params.fp16,
        load_best_model_at_end=training_params.load_best_model_at_end,
        #auto_find_batch_size=training_params.auto_find_batch_size,
        model_init_kwargs=model_kwargs,
        dataset_text_field=training_params.dataset_text_field,
        packing=training_params.packing,
        max_seq_length=llm_params.max_seq_length,
        run_name=f"{llm_params.model_id}/{data_params.comet_dataset_name}/1"
        #include_for_metrics=["loss"]
    )


    #include lora adapters
    peft_config = LoraConfig(
        r=lora_params.r,
        lora_alpha=lora_params.lora_alpha,
        lora_dropout=lora_params.lora_dropout,
        bias= lora_params.bias,
        task_type=lora_params.task_type,
        target_modules=lora_params.target_modules,
    )

    trainer_kwargs = {
        "model" : llm_params.model_id,
        "args" : training_config,
        "train_dataset" :data_obj.hf_datasets['train'],
        "eval_dataset" : data_obj.hf_datasets['validation'],
        "test_dataset" : data_obj.hf_datasets['test'],
        "processing_class" : data_obj.tokenizer,
        "peft_config" :peft_config,
        "compute_metrics" : metrics.compute_metrics,
        "preprocess_logits_for_metrics" : preprocess_logits_for_metrics,
        "formatting_func" : None,
        "callbacks" : [CometCallback()]
    }

    #training loop with validation as evaluation steps
    trainer_obj = trainer.CustomSFTTrainer(**trainer_kwargs)   

    result = trainer_obj.train()

    print_summary(result)   

    #Evaluating model's performance on a test dataset 
     
    outputs_test = trainer_obj.predict()

     


    



if __name__ == "__main__":
    Fire(run)