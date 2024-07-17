from peft import LoraConfig, get_peft_model
from transformers import AlbertForQuestionAnswering , AutoTokenizer
import pandas as pd
from peft import LoraConfig, TaskType
from transformers import TrainingArguments , Trainer 
from sklearn.model_selection import train_test_split
import numpy as np
#import os
#from huggingface_hub import HfApi

def load_model(model_name):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AlbertForQuestionAnswering.from_pretrained(model_name)

        return tokenizer, model

def lora_config():
     
    config = LoraConfig(
          task_type = TaskType.QUESTION_ANS, 
          inference_mode= False,
          r = 32,
          lora_alpha= 32,
          lora_dropout= 0.1,
          target_modules= "all-linear"
    )
    return config

def training_config():


    training_args = TrainingArguments(
    output_dir="outputs",
    learning_rate=5e-4,
    num_train_epochs=50,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    weight_decay= 0.01,
    load_best_model_at_end= True,
)
    return training_args

    
def trainer_definition(lora_model, training_args, train_ds,test_ds, computed_metrics):
    trainer = Trainer(
    model=lora_model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    compute_metrics=computed_metrics,
)
    return trainer


def create_ds(ds):

    train_ds, test_ds = train_test_split(ds, test_size=0.2, random_state=42)

    print("train is:",len(train_ds), "test is :", len(test_ds))

    return train_ds, test_ds

def tokenize_dataset(train_ds, test_ds):
     train_ds_tokenize = []
     test_ds_tokenize = []

     for index, row in train_ds.iterrows():
          l =[]
          l.append(row["text"][:512])
          tk = tokenizer(l)
          train_ds_tokenize.append(tk)


     for index, row in test_ds.iterrows():
          l =[]
          l.append(row["text"][:509])
          tk = tokenizer(l)
          test_ds_tokenize.append(tk)
    

     return train_ds_tokenize,test_ds_tokenize
              

if __name__ == "__main__":
    
    """
    token = "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"
    print(token)
    api = HfApi()
    api.create_repo("repo_name", token = token)
    """

    model_name = "twmkn9/albert-base-v2-squad2"
    tokenizer, model = load_model(model_name)
    config = lora_config()

    train = training_config()

    lora_model = get_peft_model(model, config)

    lora_model.print_trainable_parameters()

    train_args = training_config()
    
    splits = {'train': 'data/train-00000-of-00001-b42a775f407cee45.parquet', 'validation': 'data/validation-00000-of-00001-134b8fd0c89408b6.parquet'}
    df = pd.read_parquet("hf://datasets/OpenAssistant/oasst1/" + splits["train"])
    df = df[['text','role']]
    train_ds, test_ds = create_ds(df)
    #tokenizzare il training ed il test set
    train_ds_tokenize, test_ds_tokenize = tokenize_dataset(train_ds,test_ds)
    """  
    computed_metrics = compute_metrics() 

    trainer = trainer_definition(lora_model=lora_model,training_args=train_args,train_ds= train_ds_tokenize, test_ds= test_ds_tokenize, compute_metrics= computed_metrics)
    trainer.train()
    """