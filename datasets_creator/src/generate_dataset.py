
import sys
import pathlib
root_repo_directory = pathlib.Path().resolve().__str__()
sys.path.append(root_repo_directory)
from multilingual_chatbot_arena import initialize
import datasets_creator.src.constants as c
import datasets_creator.src.utils as utils
import pandas as pd
from fire import Fire
from pydantic import BaseModel
from typing import List,Optional,Dict,Union
import pathlib
import numpy as np
import pickle
import json

import os
import opik
from loguru import logger
from opik import track, opik_context
import time


from unstructured.cleaners.core import (
    clean,
    clean_extra_whitespace,
    clean_non_ascii_chars,
    group_broken_paragraphs,
    replace_unicode_quotes,
)



class DataSpecs(BaseModel):
    rel_input_path : str
    rel_output_path : Optional[str] = None
    train_size : float
    batch_size : int = 1
    num_val_sets : int = 1


class ChatbotDatasetsConstruct:
    """
    This class loads the given challenge data (in .Parquet format) for the competition, processes it to be 
    LLM digestable, and writes the set of datasets into .parquet files inside */data/datasets/* folder.

    Attributes:
    -----------
    data : Data
        Dataclass populated with attributes relevant to the input data challenge.

    """
    def __init__(self,data : DataSpecs,debug = False) -> None:
        self._data = data
        self.debug = debug
        self._training_data = None
        self._validation_data = None

        self._data_status = self.extract()

    @property
    def data_status(self) -> bool:
        return self._data_status
    
    @property
    def training_data(self) -> Optional[List]:
        return self._training_data

    @property
    def validation_data(self) -> Optional[List]:
        return self._validation_data


    def extract(self) -> bool:
        if self.debug:
            print(f"Current path: {c.CURR_PATH} + Input relative data path: {self._data.rel_input_path}")
        
        
        try:
            self._df: pd.DataFrame = pd.read_parquet(c.CURR_PATH + c.SLASH + self._data.rel_input_path)
        except FileNotFoundError as e:
            print(f"{e}->\nThe provided data path is non existent, please verify this parameter.")
            return False
        else:
            if self.debug:
                print("#------------------Observe the input file in DataFrame ----------------------------#")
                print(self._df.head(5))
        return True



    def transform(self) -> np.ndarray:
        """
        Transforms the input challenge data into prompt/answer format for each record.

        Returns:
            List[dict]: A list of prompts with its corresponding answer.
        """

        def construct_prompt(x):
            return c.PROMPT_TEMPLATE.format(prompt=x.prompt,response_a=x.response_a,response_b = x.response_b)

        
        self._df["custom_prompt"] = self._df.apply(construct_prompt,axis=1)

        def construct_answer(x):
            return f"Best model is {x.winner} based on its human preferability response for the input prompt."
        self._df["custom_answer"] = self._df.apply(construct_answer,axis=1)

        if self.debug:
            print("---------- New appended cols to DF ----------------")
            print(self._df[["custom_prompt","custom_answer"]].head())

        # extract from df prompt-answer and store it in a list
        def construct_dict(x) -> dict[str, str]:
            return {"prompt" : x.custom_prompt, "answer" : x.custom_answer, "language" : x.language}
        
    
        return self._df.apply(construct_dict,axis=1).values        


    def train_validation_split(self) -> tuple:
        """
        Splits the full competition data into training/validation sets. 
        This method allows to retrieve *num_val_sets* of validation sets instead of a single one.
        """
        # get data indices for each train/validation set
        generator = np.random.default_rng(seed=142)
        
        n = self._df.shape[0]
        train_num_indices = int(n * self._data.train_size)

        
        indices = np.arange(0,n) 
        generator.shuffle(indices)

        train_indices  = indices[:train_num_indices]        
        validation_indices = [indices[train_num_indices:]]

        if self.debug:
            print(f"training set indices size: {train_indices.shape}")
            print(f"validation set indices size: {validation_indices[0].shape}")

        if self._data.num_val_sets > 1:          

            m = validation_indices[0].shape[0] // self._data.num_val_sets            
            res = validation_indices[0].shape[0] % self._data.num_val_sets

            
            if self.debug:
                print(f"separating validation into {self._data.num_val_sets} sets...")
                print(f"elements per validation set : {m}")
                print(f"num of elements that will be appended to the last indices set : {res}")

            generator.shuffle(validation_indices[0]) 

            validation_indices_new = [
                validation_indices[0][k*m:(k+1)*m] for k in range(self._data.num_val_sets)]
            
            if res != 0:
                rem_indices = validation_indices[0][self._data.num_val_sets * m:]
                validation_indices_new[-1] = np.concatenate([validation_indices_new[-1],rem_indices])

            
            validation_indices = validation_indices_new 


        return train_indices, validation_indices       


    
    def set_data_structs(self) -> None:
        """
        Executes the algorithm to extract-transform the input data into a prompt/answer format.
        """
        logger.info("Creating training/validation datasets following prompt/answer format.")
        # get train validation split indices
        train_indices,validation_indices = self.train_validation_split()

        #transform input data
        data_arr_dict = self.transform()
        
        self._training_data = [data_arr_dict[train_indices].tolist()]

        validation_sets = []

        for val_idxs_arr in validation_indices:
            validation_sets.append(data_arr_dict[val_idxs_arr].tolist())

        self._validation_data = validation_sets
        
        logger.success("Sucessfully created training/validation datasets.")

    def save(self):
        """
        Serializes the train/validation datasets into .parquet files and saves them in the output_path or
        uploads datasets into Comet ML's project.
        """
        
        if self._data.rel_output_path: #output path given, save data into json files in path.
            
            #training data
            parent_folder = c.CURR_PATH + c.SLASH + self._data.rel_output_path
            logger.info(f"Loading training/validation sets in {parent_folder}")

            utils.to_parquet(self._training_data,parent_folder + c.SLASH + "train","train_data")

            #validation data
            utils.to_parquet(self._validation_data,parent_folder + c.SLASH + "validation","validation_data")

        else:
            logger.info("Loading training/validation sets into my workspace in Comet ML")
            self.upload_datasets_to_comet()
        logger.success("Training/validation sets loaded sucessfully.")

    
    def upload_datasets_to_comet(self):
        """
        Uploads the generated train/validation datasets into my *COMET_WORKSPACE* datasets.
        """ 
        #set up client with workspace name and api key  
        client = opik.Opik(workspace=os.environ['COMET_WORKSPACE'],api_key=os.environ['COMET_API_KEY'])

        #create/retrieve training sets in my workspace
        for i,dataset in enumerate(self._training_data):
            dataset_comet = client.get_or_create_dataset(name=f"multilingual-chatbot-arena-train-{i+1}",
            description=f"Challenge: Multilingual Chatbot Arena. Training set {i+1}.")

            for _,batch in enumerate(utils.batch_generator(dataset,self._data.batch_size,True,True)):
                dataset_comet.insert(batch)
                time.sleep(60.0)

        time.sleep(60.0)

        #create/retrieve validation sets in my workspace
        for i,dataset in enumerate(self._validation_data):
            dataset_comet = client.get_or_create_dataset(name=f"multilingual-chatbot-arena-validation-{i+1}",
            description=f"Challenge: Multilingual Chatbot Arena. Validation set {i+1}.")

            for batch in utils.batch_generator(dataset,self._data.batch_size,True,True):
                dataset_comet.insert(batch)
                time.sleep(60.0)
       

def run(
    rel_input_path : str,
    train_size : float,
    num_val_sets : int = 1,
    batch_size : int = 1,
    rel_output_path : Optional[str] = None
):
    
    #initialize .env file...
    initialize()

    data = DataSpecs(rel_input_path=rel_input_path,rel_output_path = rel_output_path,train_size=train_size,
                     batch_size=batch_size, num_val_sets=num_val_sets)

    dataset_construct_obj = ChatbotDatasetsConstruct(data,debug=False)
    if dataset_construct_obj.data_status:
        dataset_construct_obj.set_data_structs()
        dataset_construct_obj.save()

    
if __name__ == "__main__":
    Fire(run)