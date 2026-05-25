import sys
import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import torch
from torch.utils.data import Dataset
from transformers import(
DistilBertTokenizerFast,
DistilBertForSequenceClassification,
TrainingArguments,
Trainer,
EarlyStoppingCallback,
DataCollatorWithPadding
)

from sklearn.metrics import(
accuracy_score,precision_recall_fscore_support,
roc_auc_score,confusion_matrix,classification_report
)

from loguru import logger

sys.path.insert(0,str(Path(__file__).resolve().parent.parent))

from config import(
PRETRAINED_MODEL,SAVED_MODEL_PATH,MAX_TOKEN_LENGTH,
NUM_LABELS,TRAIN_BATCH_SIZE,EVAL_BATCH_SIZE,
NUM_EPOCHS,LEARNING_RATE,WARMUP_RATIO,WEIGHT_DECAY,
SEED,DATASET_DIR,MODEL_DIR
)

torch.manual_seed(SEED)
np.random.seed(SEED)


class JobPostingDataset(Dataset):

    def __init__(self,texts:list[str],labels:list[int],
                 tokenizer:DistilBertTokenizerFast,max_length:int):
        self.encodings=tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt"
        )
        self.labels=labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self,idx):
        item={k:v[idx] for k,v in self.encodings.items()}
        item["labels"]=torch.tensor(self.labels[idx],dtype=torch.long)
        return item


def compute_metrics(eval_pred):
    logits,labels=eval_pred
    probs=torch.softmax(torch.tensor(logits),dim=-1).numpy()
    preds=np.argmax(logits,axis=-1)
    fake_probs=probs[:,1]

    acc=accuracy_score(labels,preds)
    prec,rec,f1,_=precision_recall_fscore_support(
        labels,preds,average="binary",zero_division=0
    )

    try:
        auc=roc_auc_score(labels,fake_probs)
    except ValueError:
        auc=0.0

    return{
        "accuracy":round(acc,4),
        "precision":round(prec,4),
        "recall":round(rec,4),
        "f1":round(f1,4),
        "roc_auc":round(auc,4)
    }


class FakeJobTrainer:

    def __init__(self,model_name:str=PRETRAINED_MODEL):
        self.model_name=model_name
        self.device="cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer=None
        self.model=None
        self.trainer=None
        logger.info(f"Device:{self.device.upper()}")

    def load_data(self):
        train_df=pd.read_csv(DATASET_DIR/"train.csv")
        val_df=pd.read_csv(DATASET_DIR/"val.csv")
        test_df=pd.read_csv(DATASET_DIR/"test.csv")

        self.tokenizer=DistilBertTokenizerFast.from_pretrained(self.model_name)

        def make_ds(df):
            return JobPostingDataset(
                texts=df["text"].tolist(),
                labels=df["label"].tolist(),
                tokenizer=self.tokenizer,
                max_length=MAX_TOKEN_LENGTH
            )

        return make_ds(train_df),make_ds(val_df),make_ds(test_df)

    def build_model(self):
        self.model=DistilBertForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=NUM_LABELS,
            id2label={0:"Real",1:"Fake"},
            label2id={"Real":0,"Fake":1}
        )

    def configure_trainer(self,train_ds,val_ds):

        training_args=TrainingArguments(
            output_dir=str(MODEL_DIR/"checkpoints"),
            num_train_epochs=NUM_EPOCHS,
            per_device_train_batch_size=TRAIN_BATCH_SIZE,
            per_device_eval_batch_size=EVAL_BATCH_SIZE,
            learning_rate=LEARNING_RATE,
            warmup_ratio=WARMUP_RATIO,
            weight_decay=WEIGHT_DECAY,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            logging_dir=str(MODEL_DIR/"logs"),
            logging_steps=50,
            report_to="none",
            seed=SEED,
            fp16=torch.cuda.is_available()
        )

        self.trainer=Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            compute_metrics=compute_metrics,
            data_collator=DataCollatorWithPadding(self.tokenizer),
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
        )

        return self.trainer

    def train(self):
        logger.info("Starting training")
        self.trainer.train()
        logger.info("Training complete")

    def evaluate(self,test_ds):
        predictions=self.trainer.predict(test_ds)
        logits,labels=predictions.predictions,predictions.label_ids

        probs=torch.softmax(torch.tensor(logits),dim=-1).numpy()
        preds=np.argmax(logits,axis=-1)

        acc=accuracy_score(labels,preds)
        prec,rec,f1,_=precision_recall_fscore_support(
            labels,preds,average="binary",zero_division=0
        )

        auc=roc_auc_score(labels,probs[:,1])
        cm=confusion_matrix(labels,preds).tolist()
        cr=classification_report(labels,preds,target_names=["Real","Fake"])

        metrics={
            "accuracy":round(float(acc),4),
            "precision":round(float(prec),4),
            "recall":round(float(rec),4),
            "f1":round(float(f1),4),
            "roc_auc":round(float(auc),4),
            "confusion_matrix":cm
        }

        metrics_path=MODEL_DIR/"test_metrics.json"
        with open(metrics_path,"w") as f:
            json.dump(metrics,f,indent=2)

        return metrics

    def save(self,path:str=SAVED_MODEL_PATH):
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)


def train_pipeline():
    from dataset.prepare_data import prepare_dataset

    prepare_dataset()

    trainer=FakeJobTrainer()
    train_ds,val_ds,test_ds=trainer.load_data()
    trainer.build_model()
    trainer.configure_trainer(train_ds,val_ds)
    trainer.train()

    metrics=trainer.evaluate(test_ds)
    trainer.save()

    print("\nTraining complete")
    print(metrics)

    return metrics


if __name__=="__main__":
    train_pipeline()