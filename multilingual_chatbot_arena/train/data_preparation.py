from config import DataParams
import pathlib
import sys
import os
from fire import Fire
root_repo_directory = pathlib.Path().resolve().__str__() + "/multilingual-chatbot-arena"
sys.path.append(root_repo_directory)
from multilingual_chatbot_arena import initialize
import datasets_creator.src.constants as c

import opik
from config import data_params,DataParams, llm_params, LLMParams
import pandas as pd

from datasets import Dataset

from datasets import DatasetDict

from transformers import AutoTokenizer

from sklearn.model_selection import train_test_split


class Data:

    def __init__(self,data_params : DataParams, llm_params : LLMParams,tokenizer):
        self.data_params = data_params
        self.ll_params = llm_params
        self.opik_client = opik.Opik(project_name=os.environ['COMET_PROJECT_NAME'],
        workspace=os.environ['COMET_WORKSPACE'],api_key=os.environ['COMET_API_KEY'])

        self.tokenizer = tokenizer
        self.process_data()
        

    def process_data(self):

        train_original_size = self.data_params.train_validation_test_split[0]
        valid_original_size = self.data_params.train_validation_test_split[1]

        dataset_pd: pd.DataFrame = self.ingest_from_comet()

        labels = dataset_pd['answer']
        dataset_pd.drop("answer",axis=1,inplace=True)   

        # split dataset into train-validation datasets based on train size and remainder
        dataset_train,dataset_valid,y_train,y_valid = train_test_split(
            dataset_pd,labels, train_size=train_original_size,
                         random_state=self.data_params.seed, stratify=labels )
        
        dataset_train['answer'] = y_train

        # split validation dataset into validation-test datasets based on validation size and remainder

        #compute validation set proportion for the validation-test split
        valid_test_num_saples = dataset_valid.shape[0]
        valid_original_num_samples = int(dataset_pd.shape[0] * valid_original_size)
        validset_proportion = valid_original_num_samples / valid_test_num_saples


        dataset_valid,dataset_test,y_valid,y_test = train_test_split(
            dataset_valid,y_valid, train_size=validset_proportion,
                         random_state=self.data_params.seed, stratify=y_valid )
        
        dataset_valid['answer'] = y_valid
        dataset_test['answer'] = y_test

        # get train, validation test split
        dict_datasets_pd = {
            "train" : dataset_train,
            "validation" :  dataset_valid,
            "test" : dataset_test
        }   

        #convert from pandas datasets to hf datasets
        self.hf_datasets = DatasetDict({k : Dataset.from_pandas(v) for k,v in dict_datasets_pd.items()})
        
        #prepare dataset's examples with the chat template 

        def apply_chat_template(example,tokenizer):
            messages = [
                {'role' : 'system' , 'content' : c.SYSTEM_TEMPLATE},
                {'role' : 'user' , 'content' : example['prompt']},
                {'role' : 'assistant', 'content' : example['answer']}
            ]

            example["conversation"] = tokenizer.apply_chat_template(messages,tokenize=False)

            return example

        
        column_names = self.hf_datasets['train'].column_names
        self.hf_datasets = self.hf_datasets.map(apply_chat_template,
                                                fn_kwargs={"tokenizer": self.tokenizer},
                                remove_columns=column_names,
                                desc="Applying chat template")       


    def ingest_from_comet(self) ->pd.DataFrame:
        """
        Connect to Comet ML's target project and load desired dataset
        """

        return self.opik_client.get_or_create_dataset(data_params.comet_dataset_name).to_pandas()





def run():
    

    initialize()

    #data_obj = Data(data_params,llm_params)

    



if __name__ == "__main__":
    Fire(run)
