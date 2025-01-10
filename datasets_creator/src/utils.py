from typing import List,Optional,Dict,Union
import pickle
import pathlib
import datasets_creator.src.constants as c
import numpy as np
def to_pickle(data : List[List[Dict[str,str]]],file_path : str, file_name : str):
    """
    Creates pickle files (based on the number of lists inside list) in the desired file path.

    Args:
        data :
        file_path (str): Desired path to store the files.
        file_name (str): Designated prefix name present in every file.
    """
    pathlib.Path(file_path).mkdir(parents=True, exist_ok=True)
    for i,dataset in enumerate(data):
        file = file_path + c.SLASH + file_name + c.UNDERSCORE + f"{i+1}.pkl"
        with open(file, mode="wb") as f:
            pickle.dump(dataset,f)
       
              
def batch_generator(data : List[Dict[str,str]], batch_size : int):
    """
    Generator which plits input data into batches 

    Args:
        data (List[Dict[str,str]]) : Input data to be parsed
        batch_size (int) : Potentially the size of each batch.
    """
    n = len(data)
    m = n // batch_size

    for k in range(m-1):
        yield data[k*batch_size:(k+1)*batch_size]
        
    yield data[(m-1)*batch_size:]


""" data = [np.random.randint(0,20,size=(np.random.randint(1,25),)).tolist() for _ in range(3)]
print(data)

gen =batch_generator(data[0],3)

print(next(gen))
print(next(gen))
print(next(gen))
print(next(gen)) """