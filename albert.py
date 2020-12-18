import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AlbertTokenizer, AlbertModel, AlbertConfig, AlbertForPreTraining, AdamW, \
    get_linear_schedule_with_warmup, AlbertForSequenceClassification, AutoModel, AutoTokenizer
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from collections import defaultdict
import pandas as pd
import random
import matplotlib as plt

RANDOM_SEED = 778
BATCH_SIZE = 1
MAX_LEN = 150
EPOCHS = 30
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.cuda.current_device()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed(seed)


class GPReviewDataset(Dataset):
    def __init__(self, reviews, targets, tokenizer, max_len):
        self.reviews = reviews
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, item):
        review = str(self.reviews[item])
        target = self.targets[item]
        encoding = self.tokenizer.encode_plus(
              review,
              add_special_tokens=True,
              max_length=self.max_len,
              return_token_type_ids=False,
              padding='max_length',
              return_attention_mask=True,
              return_tensors='pt',
        )

        return{
            'review_text': review,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'targets': torch.tensor(target, dtype=torch.long)
        }


def create_data_loader(df, tokenizer, max_len, batch_size):
    ds = GPReviewDataset(
        reviews=df.text.to_numpy(),
        targets=df.is_humor.to_numpy(),
        tokenizer=tokenizer,
        max_len=max_len
    )

    return DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=4
    )


class MyModel(nn.Module):
    def __init__(self, freeze_bert=False, n_classes=2):
        super(MyModel, self).__init__()
        albert_xxlarge_configuration = AlbertConfig(output_hidden_states=True, output_attentions=True)
        # self.model = AlbertModel.from_pretrained(pretrained_model_name_or_path=MODEL_PATH)
        self.model = AutoModel.from_pretrained(pretrained_model_name_or_path=MODEL_PATH)
        if freeze_bert:
            for p in self.model.parameters():
                p.requires_grad = False

        self.drop = nn.Dropout(p=0.1)
        self.classifier = nn.Linear(self.model.config.hidden_size, n_classes)
        self.softmax = nn.Softmax(dim=1)


    def forward(self, input_ids, attention_mask):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled_output = outputs[1]

        output = self.drop(pooled_output)
        output = self.classifier(output)
        return self.softmax(output)



def train_epoch(
    model,
    data_loader,
    loss_fn,
    optimizer,
    device,
    scheduler,
    n_examples
):
    model = model.train()
    losses = []
    correct_predictions = 0
    for d in data_loader:
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        targets = d["targets"].to(device)
        outputs = model(
          input_ids=input_ids,
          attention_mask=attention_mask
        )
        _, preds = torch.max(outputs, dim=1)
        loss = loss_fn(outputs, targets)
        correct_predictions += torch.sum(preds == targets)
        losses.append(loss.item())
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
    return correct_predictions.double() / n_examples, np.mean(losses)

def eval_model(model, data_loader, loss_fn, device, n_examples):
    model = model.eval()
    losses = []
    correct_predictions = 0
    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["targets"].to(device)
            outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
            )
            _, preds = torch.max(outputs, dim=1)
            loss = loss_fn(outputs, targets)
            correct_predictions += torch.sum(preds == targets)
            losses.append(loss.item())
    return correct_predictions.double() / n_examples, np.mean(losses)

def get_predictions(model, data_loader):
    model = model.eval()
    review_texts = []
    predictions = []
    prediction_probs = []
    real_values = []
    with torch.no_grad():
        for d in data_loader:
            texts = d["review_text"]
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["targets"].to(device)
            outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
            )
            _, preds = torch.max(outputs, dim=1)
            review_texts.extend(texts)
            predictions.extend(preds)
            prediction_probs.extend(outputs)
            real_values.extend(targets)
            predictions = torch.stack(predictions).cpu()
            prediction_probs = torch.stack(prediction_probs).cpu()
            real_values = torch.stack(real_values).cpu()
    return review_texts, predictions, prediction_probs, real_values

if __name__ == '__main__':

    set_seed(RANDOM_SEED)

    MODEL_PATH = 'bert-base-cased'
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, output_hidden_states=True, return_dict=True)

    df = pd.read_csv("./datas/task1/train/train.csv")
    df = df[['text', 'is_humor']]
    class_names = ['is_humor', 'not_humor']
    df_train, df_test = train_test_split(
        df,
        test_size=0.1,
        random_state=RANDOM_SEED
    )
    df_val, df_test = train_test_split(
        df_test,
        test_size=0.5,
        random_state=RANDOM_SEED
    )

    train_data_loader = create_data_loader(df_train, tokenizer, MAX_LEN, BATCH_SIZE)
    val_data_loader = create_data_loader(df_val, tokenizer, MAX_LEN, BATCH_SIZE)
    test_data_loader = create_data_loader(df_test, tokenizer, MAX_LEN, BATCH_SIZE)
    data = next(iter(train_data_loader))

    model = MyModel(n_classes=2)
    # model.load_state_dict(torch.load('./best_model_state.bin'))
    model = model.to(device)
    input_ids = data['input_ids'].to(device)
    attention_mask = data['attention_mask'].to(device)

    optimizer = AdamW(model.parameters(), lr=5e-5, correct_bias=False)
    total_steps = len(train_data_loader) * EPOCHS

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )

    loss_fn = nn.CrossEntropyLoss().to(device)

    history = defaultdict(list)
    best_accuracy = 0
    for epoch in range(EPOCHS):
        print(f'Epoch {epoch + 1}/{EPOCHS}')
        print('-' * 10)
        train_acc, train_loss = train_epoch(
            model,
            train_data_loader,
            loss_fn,
            optimizer,
            device,
            scheduler,
            len(df_train)
        )
        print(f'Train loss {train_loss} accuracy {train_acc}')
        val_acc, val_loss = eval_model(
            model,
            val_data_loader,
            loss_fn,
            device,
            len(df_val)
        )
        print(f'Val   loss {val_loss} accuracy {val_acc}')
        print()
        history['train_acc'].append(train_acc)
        history['train_loss'].append(train_loss)
        history['val_acc'].append(val_acc)
        history['val_loss'].append(val_loss)
        if val_acc > best_accuracy:
            torch.save(model.state_dict(), 'best_model_state.bin')
            best_accuracy = val_acc



    test_acc, _ = eval_model(
        model,
        test_data_loader,
        loss_fn,
        device,
        len(df_test)
    )
    test_acc.item()
    print(test_acc.item())

    y_review_texts, y_pred, y_pred_probs, y_test = get_predictions(
        model,
        test_data_loader
    )

    print(classification_report(y_test, y_pred, target_names=class_names))

