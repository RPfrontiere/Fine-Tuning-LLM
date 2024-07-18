from peft import LoraConfig, TaskType, PeftModel, get_peft_model, PeftConfig
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, TrainingArguments, Trainer, DefaultDataCollator
from datasets import load_dataset, DatasetDict, Dataset
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import evaluate
import torch

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
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    weight_decay= 0.01,
    load_best_model_at_end= True,
)
    return training_args

    
def trainer_definition(lora_model, training_args, tokenized_squad, tokenizer, data_collator):
    trainer = Trainer(
    model=lora_model,
    args=training_args,
    train_dataset=tokenized_squad["train"],
    eval_dataset=tokenized_squad["test"],
    tokenizer= tokenizer,
    data_collator= data_collator,
)
    return trainer

def preprocess_function(examples):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=384,
        truncation="only_second",
        return_offsets_mapping=True,
        padding="max_length",
    )

    offset_mapping = inputs.pop("offset_mapping")
    answers = examples["answers"]
    start_positions = []
    end_positions = []

    for i, offset in enumerate(offset_mapping):
        answer = answers[i]
        start_char = answer["answer_start"][0]
        end_char = answer["answer_start"][0] + len(answer["text"][0])
        sequence_ids = inputs.sequence_ids(i)

        # Find the start and end of the context
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        # If the answer is not fully inside the context, label it (0, 0)
        if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            # Otherwise it's the start and end token positions
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs                

if __name__ == "__main__":

    #Define model and Dataset   
    model_name = "distilbert/distilbert-base-uncased"
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    squad = load_dataset("squad", split = "train[:5000]")
    squad = squad.train_test_split(test_size=0.2)
    print(squad["train"][0])
    #Define tokenizer and preprocess function
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenized_squad = squad.map(preprocess_function, batched= True, remove_columns=squad["train"].column_names)

    #Define an example batch
    data_collator = DefaultDataCollator()

    config = lora_config()

    training_args = training_config()

    lora_model = get_peft_model(model, config)

    lora_model.print_trainable_parameters()

    trainer = trainer_definition( lora_model, training_args, tokenized_squad, tokenizer, data_collator )

    trainer.train()
    
   