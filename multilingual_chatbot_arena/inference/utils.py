

import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizer,PreTrainedTokenizerFast


import pathlib
import sys
root_repo_directory = pathlib.Path().resolve().parent.parent.__str__()
sys.path.append(root_repo_directory)

SYSTEM_TEMPLATE = 'You are a specialist in evaluating multilingual chat responses, with a focus on comparing and ranking outputs from different LLMs. Your primary goal is to determine which response is more likely to be preferred by humans based on factors such as clarity, relevance, tone, and overall quality.\n'


class ChatbotDataset(Dataset):
    def __init__(self,data : pd.DataFrame):
        """

        Args:
            data pd.DataFrame : data from dataset,
            comet_id Optional[str] : dataset's name id from comet ML
            prompt_id Optional[str] : prompt's template id from comet ML
        """
        self.data = data
          


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        #get either a single data point or a pandas Dataframe window of data points
        data_window = self.data.iloc[idx]    

        return data_window.to_dict()
    
class ChatbotDataloader(DataLoader):
    def __init__(self, tokenizer :  PreTrainedTokenizer | PreTrainedTokenizerFast, **kwargs):
        self.tokenizer = tokenizer
        
        kwargs["collate_fn"] = self.chatbot_collate
        super().__init__(**kwargs)

    
    def chatbot_collate(self,batch):
        """Custom collate function to teach the Dataloader class how to parse the batches into an llm friendly format
        Args:
            original_batch : List of batch elements with len -> batch_size. Each list's element strictly follows 
            the format inside __getitem__ from Dataset class. 
        
        """
        prompts,answers,languages,records_id = [],[],[],[]

        
        for dic in batch:
            if self.tokenizer.chat_template: #tokenizer has chat template
                
                prompt_messages = [
                    {"role": "system", "content": SYSTEM_TEMPLATE},
                    {"role" : "user", "content" : dic["prompt"]}
                ]

                try:
                    prompt_text  = self.tokenizer.apply_chat_template(
                        prompt_messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                except Exception:
                    # chat template does not support system role

                    prompt_messages = [
                    {"role": "user", "content": SYSTEM_TEMPLATE},
                    {"role" : "assistant" , "content" : "Ok"},
                    {"role" : "user", "content" : dic["prompt"]}
                    ]

                    prompt_text  = self.tokenizer.apply_chat_template(
                        prompt_messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )              

            else:
                prompt_text = """
                {system}{prompt}
                """.format(system=SYSTEM_TEMPLATE,prompt=dic['prompt'])

                
            answers.append(dic['answer'])
            prompts.append(prompt_text)
            languages.append(dic['language'])
            records_id.append(dic['id'])


        #tokenize batch of prompts and answers
        prompt_tokenize = self.tokenizer(prompts,
                padding='longest',truncation=True,return_tensors="pt")

        return {
            "inputs" : prompt_tokenize, #Dict[str,torch.Tensor]
            "prompts" : prompts, #list[str],
            "labels" : answers, #list[str]
            "languages" : languages, #list[str]
            "records_id" : records_id, #list[str]
            "longest_seq" : prompt_tokenize["input_ids"].shape[1] #int
        }
          