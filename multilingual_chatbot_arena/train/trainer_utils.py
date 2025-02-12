import torch
import torch.nn.functional as F
def preprocess_logits_for_metrics(logits : torch.Tensor, labels : torch.Tensor):
    # apply softmax layer to logits
    probs = F.softmax(logits,dim=-1)

    #greedy implementation: next token is the id that has the highest probability from vocabulary distribution.
    prediction_ids = probs.argmax(dim=-1)

    return prediction_ids

