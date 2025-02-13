import torch
import torch.nn.functional as F
from pynvml import *

def preprocess_logits_for_metrics(logits : torch.Tensor, labels : torch.Tensor):
    # apply softmax layer to logits
    probs = F.softmax(logits,dim=-1)

    #greedy implementation: next token is the id that has the highest probability from vocabulary distribution.
    prediction_ids = probs.argmax(dim=-1)

    return prediction_ids

def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")


def print_summary(result):
    print(f"Time: {result.metrics['train_runtime']:.2f}")
    print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
    print_gpu_utilization()

