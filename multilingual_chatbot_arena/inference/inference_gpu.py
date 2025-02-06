"""
This function contains the workloads to run inference pipelines using rental GPU services such as Beam Cloud.
"""
from beam import function,Image
from beam import endpoint
import re
from typing import Any,Union,Optional
from tqdm import tqdm
from loguru import logger
from config import InferenceArgs,config
import torch
import utils

from transformers import AutoTokenizer,AutoModelForCausalLM
import opik
import os
from subprocess import call



   
def postprocess_generated_output_deepseek(pattern : str,outputs : list[str]) -> list[Any]:
    processed = []
    for output in outputs:
        match = re.search(pattern, output)
        if match:
            processed.append(output[match.end():])
        else:
            processed.append(output)
    return processed

dict_vals_datatype = Union[list[int],int]
@torch.inference_mode()
@torch.no_grad()
def model_inference(dataset_name,model,dataloader,config : InferenceArgs,
        resume : Optional[dict[str,dict_vals_datatype]] = None) -> dict[str,dict_vals_datatype]:
    """
    Retrieves two lists, the first list specifies the LLM's decisions per record, on which response was more humanly
    seen. The other specifies the challenge's ground truth.

    Args:
        model : HuggingFace Pretrained LLM.
    """
    

    """     if resume:
        global_output_winners = resume['predictions']
        global_answers = resume['answers']
        resume_idx = resume['last_idx'] + 1 """
    
    global_output = []

    deepseek_model_pattern = r'DeepSeek'
    try:

        i = 0
        for i,batch in enumerate(tqdm(dataloader,desc=f"Dataset : {dataset_name} - Model Inference")):
            #if i < resume_idx:
            #    continue

            # Let's send current batch into model device

            inputs= batch["inputs"].to(model.device)

            logger.info(f"Batch: {i}. Max Batch Input tokens size : {inputs['input_ids'].shape[1]}")


            #forward batch of input tokens into the model, get output token ids
            output_token_ids  = model.generate(
                **inputs,
                max_new_tokens=config.max_new_tokens,
                do_sample = False,
                cache_implementation = "quantized",
                #cache_config= {"nbits" : 4, "backend" : "quanto"}
            )

            output_token_ids = output_token_ids.detach().cpu()

            #Remove prompt from generated response
            
            output_token_ids = [output_token_ids[i,batch["longest_seq"]:]  for i in range(
                output_token_ids.shape[0])]

            #Decode batch's output
            #list[config.batch_size]
            batch_decoded_responses = dataloader.tokenizer.batch_decode(output_token_ids, skip_special_tokens=True)

            if re.search(deepseek_model_pattern, config.model_name):
                think_pattern = "</think>"
                batch_decoded_responses = postprocess_generated_output_deepseek(think_pattern,batch_decoded_responses)
                

            
            #store batch predictions and dataset's metadata

            batch.update({
                'predictions' : batch_decoded_responses,
                'dataset_name' : [dataset_name] * config.batch_size,
                'comet_prompt_template_name' : [config.comet_prompt_template_name] * config.batch_size
            })




            #transform batch dict, dict[collumn para, list] to store a DS where each element is a record,
            #i.e. list[record in batch]

            def from_batch_to_records_struct(batch):
                """
                Args:
                    batch: dict of parameters, where each param is constituted of a data struct of size batch
                returns:
                    list of records inside batch
                """
                return [
                    {
                       'dataset_name' : batch['dataset_name'][i],
                       'record_id' : batch['records_id'][i],
                       'prompt_template_name' : batch['comet_prompt_template_name'][i],
                       'prompt' : batch['prompts'][i],
                       'prediction' : batch['predictions'][i],
                       'label' : batch['labels'][i],
                       'language' : batch['languages'][i]
                    }
                    for i in range(config.batch_size)
                ]
            
            #store record outputs in global container
            global_output.extend(from_batch_to_records_struct(batch))


            #clear GPU cache
            torch.cuda.empty_cache()                
    except KeyboardInterrupt as k:
        print(k)
    except Exception as e:
        print(e)
    finally:
        return {
            'output' : global_output,
            'last_idx' : i
        }


def inference_pipeline(opik_client,model, tokenizer, num_datasets : int,config : InferenceArgs,
                       resume : Optional[list[dict[str,dict_vals_datatype]]] = None):
    
    global_ouput = []

    """     resume_dataset_id = 0
    resume_last_dict = None
    if resume:
        global_ouput = resume
        resume_dataset_id = len(resume)
        resume_last_dict = resume.pop() """



    
    for dataset_id in range(1,num_datasets+1):
        """         if dataset_id < resume_dataset_id:
            continue """

        #get dataset from commet ML
        dataset_name = f"{config.comet_dataset_name}-{dataset_id}"
        dataset = opik_client.get_or_create_dataset(dataset_name).to_pandas()

        #construct Dataset and Dataloader
        dataset = utils.ChatbotDataset(dataset)
        dataloader = utils.ChatbotDataloader(tokenizer=tokenizer,dataset=dataset,batch_size=config.batch_size)

        #run inference per dataset inside function...
        output = model_inference(dataset_name,model,dataloader,config)

        #store outputs from current dataset in the specified project from comet ML

        def store_results_in_project_comet_ml(output):
            for i,record in enumerate(output):
                trace_dict = {
                    "comet_dataset_name" : record['dataset_name'],
                    "comet_prompt_template_name" : record['prompt_template_name']
                }
                
                trace = opik_client.trace(
                    name=f"record_results:{record['record_id']}",
                    metadata=trace_dict
                )

                # Add llm call
                trace.span(
                    name="llm call",
                    input={'prompt' : record['prompt']},
                    output={'response' : record['prediction']},
                    metadata={'model' : config.model_name, 'label' : record['label'], 'language' : record['language']}
                )

                trace.end()

        store_results_in_project_comet_ml(output['output'])
        
        """         n = len(dataset)

        if resume_last_dict:
            outputs = model_inference(dataset_id,dataset,config,server_client,
                                      resume_last_dict)
            resume_last_dict = None
        else:
            outputs = model_inference(dataset_id,dataset,config,server_client) """

        global_ouput.append(output)

        """         if outputs['last_idx'] < n - 1:
            print(f"Error during batch datasets inferencing...")
            return global_ouput """
    return global_ouput

image = (
    Image(python_version="python3.10")
    .micromamba()
    .add_python_packages(packages=["torch"])
    #.add_python_packages(packages=["flash-attn"])
    .add_python_packages(packages=["optimum-quanto","transformers","accelerate","tqdm","loguru","pandas","opik"])
    #.add_python_packages(packages=["tqdm","loguru","pandas","opik"])
)

@function(gpu="A100-40",
          cpu=8,
          memory="32Gi",
          secrets=[
              "COMET_API_KEY",
              "COMET_PROJECT_NAME",
              "COMET_WORKSPACE"
          ],image=image)
def run():
    #initialize()
    #SET CUDA HOME
    with open('./load_constants.sh', 'rb') as file:
        script = file.read()
        rc = call(script, shell=True)

    opik_client = opik.Opik(project_name=os.environ['COMET_PROJECT_NAME'],
        workspace=os.environ['COMET_WORKSPACE'],api_key=os.environ['COMET_API_KEY'])

    tokenizer = AutoTokenizer.from_pretrained(
    config.model_name,padding_side="left",legacy=False)
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        device_map="auto"
    )
    model.eval()

    output  = inference_pipeline(opik_client,model,tokenizer,1,config)
        
if __name__ == "__main__":
    run()

