from typing import List,Optional,Dict,Union
import pickle
import pathlib
import datasets_creator.src.constants as c
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
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


def to_parquet(data : List[List[Dict[str,str]]],file_path : str, file_name : str):
    """
    Creates parquet files (based on the number of lists inside list) in the desired file path.

    Args:
        data :
        file_path (str): Desired path to store the files.
        file_name (str): Designated prefix name present in every file.
    """
    pathlib.Path(file_path).mkdir(parents=True, exist_ok=True)
    for i,dataset in enumerate(data):


        file = file_path + c.SLASH + file_name + c.UNDERSCORE + f"{i+1}.parquet"
        df = pd.DataFrame(dataset)
        # Convert DataFrame to Arrow Table
        table = pa.Table.from_pandas(df)

        # Write Arrow Table to Parquet file
        pq.write_table(table, file)
        


    
       
              
def batch_generator(data : List[Dict[str,str]], batch_size : int,reverse : bool = False,start_last : bool = False):
    """
    Generator which plits input data into batches 

    Args:
        data (List[Dict[str,str]]) : Input data to be parsed
        batch_size (int) : Potentially the size of each batch.
    """
    n = len(data)
    m = n // batch_size

    stop = m-1
    start = 0
    step = 1

    if start_last:
        start = m - 1
        stop = 0
        step = - 1

    
    for k in range(start,stop,step):

        if not reverse:

            batch = data[k*batch_size:(k+1)*batch_size]

            if start_last and k == m - 1:
                batch = data[k*batch_size:]
        else: #reverse flag
            batch = data[(k+1)*batch_size - 1:k*batch_size-1:-1]
            if not start_last and k== 0:
                batch = data[(k+1)*batch_size - 1::-1]
            
            if start_last and k == m-1:
                batch = data[:k*batch_size-1:-1]


        yield batch
    
    if not reverse:
        batch = data[(m-1)*batch_size:]

        if start_last:
            window = batch_size
            if m <= 1:
                window = n

            batch = data[:window]

    else:
        batch = data[:(m-1)*batch_size-1:-1]

        if start_last:
            batch = data[batch_size - 1::-1]

        if m <= 1:
            batch = data[::-1]


    yield batch


"""
Quick tests functionaloty of batch_generator:

"""

""" 
generator = np.random.default_rng(seed=142)
size= 20
nums = generator.integers(0,20,size = (size,))
print(f"nums : {nums}")
batch_size = 7
print(f"batch_size : {batch_size}")
def tests(nums,batch_size,reverse : bool = False, start_last : bool = False):
    ans = []
    for batch_nums in batch_generator(nums,batch_size,reverse,start_last):
        ans.append(batch_nums.tolist())
    return ans

print(f"batch gen, reverse : {False}, start_last : {False}, ans : \n{tests(nums,batch_size)}")
print(f"batch gen, reverse : {False}, start_last : {True}, ans : \n{tests(nums,batch_size,start_last=True)}")
print(f"batch gen, reverse : {True}, start_last : {False}, ans : \n{tests(nums,batch_size,reverse=True)}")
print(f"batch gen, reverse : {True}, start_last : {True}, ans : \n{tests(nums,batch_size,reverse=True,start_last=True)}")


print("-" * 40)

generator = np.random.default_rng(seed=142)
size= 20
nums = generator.integers(0,20,size = (size,))
print(f"nums : {nums}")
batch_size = 11 #fix 20 and 19-18
print(f"batch_size : {batch_size}")

print(f"batch gen, reverse : {False}, start_last : {False}, ans : \n{tests(nums,batch_size)}")
print(f"batch gen, reverse : {False}, start_last : {True}, ans : \n{tests(nums,batch_size,start_last=True)}")
print(f"batch gen, reverse : {True}, start_last : {False}, ans : \n{tests(nums,batch_size,reverse=True)}")
print(f"batch gen, reverse : {True}, start_last : {True}, ans : \n{tests(nums,batch_size,reverse=True,start_last=True)}")

print("-" * 40)

generator = np.random.default_rng(seed=142)
size= 20
nums = generator.integers(0,20,size = (size,))
print(f"nums : {nums}")
batch_size = 25 #fix 20 and 19-18
print(f"batch_size : {batch_size}")

print(f"batch gen, reverse : {False}, start_last : {False}, ans : \n{tests(nums,batch_size)}")
print(f"batch gen, reverse : {False}, start_last : {True}, ans : \n{tests(nums,batch_size,start_last=True)}")
print(f"batch gen, reverse : {True}, start_last : {False}, ans : \n{tests(nums,batch_size,reverse=True)}")
print(f"batch gen, reverse : {True}, start_last : {True}, ans : \n{tests(nums,batch_size,reverse=True,start_last=True)}")

 """