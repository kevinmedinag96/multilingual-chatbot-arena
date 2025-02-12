
from transformers.trainer_utils import (
    EvalPrediction
)
import re

from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix,roc_auc_score

import sys
import pathlib
root_repo_directory = pathlib.Path().resolve().__str__() + "/multilingual-chatbot-arena"
sys.path.append(root_repo_directory)
from multilingual_chatbot_arena import initialize

import data_preparation
from config import data_params, llm_params

def extract_model_winners(texts : list[str]):
    """
    Current method works only for LLMs that were instruct fine tuned with the same chat template as Gwen's family
    """
    global_winners = []
    for text in texts:

        pattern = r"<\|im_start\|>[\s\S]*?<\|im_end\|>[\s]*?(<\|endoftext\|>)?"#r"<\|im_start\|>[\s\S]*?<\|im_end\|>\n<\|endoftext\|>"
        matches = re.findall(pattern,text)

        if matches:

            for match in matches:
                match = match.strip()
                
                def get_class(match):
                    if 'a' in match:
                        return 1
                    elif 'b' in match:
                        return 0
                    return 2
                
                cls = get_class(match)
                if cls != 2:
                    global_winners.append(cls)
                    break
        else:
                #guess it is model_a
                global_winners.append(1)
                
    
    return global_winners


def compute_metrics(eval_pred : EvalPrediction , processing_class):
    prediction_ids,label_ids  = eval_pred

    prediction_ids[prediction_ids == -100] = processing_class.pad_token_id
    label_ids[label_ids == -100] = processing_class.pad_token_id

    #decode predictions and labels
    
    predictions_text = processing_class.batch_decode(prediction_ids)
    labels_text = processing_class.batch_decode(label_ids)

    #parse text to extract the best model
    predictions_winners = extract_model_winners(predictions_text)
    labels_winners = extract_model_winners(labels_text)

    #compute metrics...
    tn,fp,fn,tp = confusion_matrix(labels_winners,predictions_winners).ravel()
    negative_predictive_value = tn /(tn  + fn)
    specificity = tn / (tn + fp)
    return {
        "accuracy" : accuracy_score(labels_winners,predictions_winners),
        "precision" : precision_score(labels_winners,predictions_winners),
        "recall" : recall_score(labels_winners,predictions_winners),
        "negative_predictive_value" : negative_predictive_value,
        "specificity" : specificity,
        "roc_auc" : roc_auc_score(labels_winners,predictions_winners)
    }

if __name__ == "__main__":
    initialize()
    data_obj = data_preparation.Data(data_params,llm_params)

    extract_model_winners(data_obj.hf_datasets["validation"]["conversation"])