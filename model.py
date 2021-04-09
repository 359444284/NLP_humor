"""
this is all the code for my individual project
some of the structure of my code is adapted from Venelin's blog: https://curiousily.com/posts/sentiment-analysis-with-bert-and-hugging-face-using-pytorch-and-python/

Implement by CHAOYU DENG
"""
import argparse
import ast
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AdamW, get_linear_schedule_with_warmup, AutoModel, AutoTokenizer
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, mean_squared_error
from collections import defaultdict
import pandas as pd
import random
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Model HyperParameter')
parser.add_argument('--batch_size', type=int, default=8, metavar='N',
                    help='input batch size for training (default: 8)')
parser.add_argument('--epochs', type=int, default=15, metavar='E',
                    help='number of epochs to train (default: 15)')
parser.add_argument('--lr', type=float, default=2e-6, metavar='LR',
                    help='learning rate (default: 2e-6)')
parser.add_argument('--seed', type=int, default=70, metavar='S',
                    help='random seed (default: 70)')
parser.add_argument('--cuda', type=int, nargs='+', default=[0, 1, 2], metavar='C',
                    help='which GPU use to train (default: 0, 1, 2)')
parser.add_argument('--uncertainty', type=ast.literal_eval, default=False,
                    help='weighting with uncertainty (defalut: True)')
parser.add_argument('--all_layer', type=ast.literal_eval, default=True,
                    help="use all layer trick (defalut: True)")
parser.add_argument("--weights", type=float, nargs='+', default=[0.4, 0, 0], metavar='W',
                    help='the loss weight for subtask 1a, 1b, 1c(default: 0.4, 0, 0)')
parser.add_argument("--dropout", type=float, nargs='+', default=[0.3, 0.3, 0.3, 0.3], metavar='W',
                    help='the dropout for subtask 1a, 1b, 1c(default: 0.3, 0.3, 0.3, 0.3)')
parser.add_argument('--model', type=str, default='roberta-large',
                    help='the name of pre-trained model using in experimrnt like albert-xxlarge-v2 (default: roberta-large)')
args = parser.parse_args()

RANDOM_SEED = args.seed
BATCH_SIZE = args.batch_size
LEARNING_RATE = args.lr
MAX_LEN = 150
EPOCHS = args.epochs
WEIGHT_1A = args.weights[0]
WEIGHT_1B = args.weights[1]
WEIGHT_1C = args.weights[2]
WEIGHT_2A = 1.0 - (WEIGHT_1A + WEIGHT_1B + WEIGHT_1C)
DROP_1A = args.dropout[0]
DROP_1B = args.dropout[1]
DROP_1C = args.dropout[2]
DROP_2A = args.dropout[3]

USE_ALL_LAYER = args.all_layer
Weight_By_Uncertainty = args.uncertainty
MODEL_PATH = args.model
DEVIDE_IDS = args.cuda

print('batch_size', BATCH_SIZE)
print('epochs', EPOCHS)
print('lr', LEARNING_RATE)
print('seed', RANDOM_SEED)
print('cuda', DEVIDE_IDS)
print('uncertainty', Weight_By_Uncertainty)
print('all_layer', USE_ALL_LAYER)
print('weights', args.weights)
print('model', MODEL_PATH)
print('dropout', args.dropout)

torch.cuda.current_device()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed(seed)


# Adapted from Venelin's blog
# this function use different tockenizers to embedding the input sentenses to tokens
class GPReviewDataset(Dataset):
    def __init__(self, dataframe, with_label, tokenizer, max_len):
        self.reviews = dataframe.text.to_numpy()
        self.with_label = with_label
        if with_label:
            self.targets = dataframe.list.to_numpy()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, item):
        review = str(self.reviews[item])
        encoding = self.tokenizer.encode_plus(
            review,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
        )

        if self.with_label:
            return {
                'review_text': review,
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'targets': torch.tensor(self.targets[item], dtype=torch.float)
            }
        else:
            return {
                'review_text': review,
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
            }


# Adapted from Venelin's blog
# create a troch data loader
def create_data_loader(df, with_label, tokenizer, max_len, batch_size):
    ds = GPReviewDataset(
        df,
        with_label,
        tokenizer=tokenizer,
        max_len=max_len
    )

    return DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=4
    )


# all the code about my model
class MyModel(nn.Module):
    def __init__(self, freeze_bert=False, use_all_layer=True):
        super(MyModel, self).__init__()
        self.use_all_layer = use_all_layer
        self.model = AutoModel.from_pretrained(pretrained_model_name_or_path=MODEL_PATH, output_hidden_states=True,
                                               output_attentions=True, return_dict=True)
        if freeze_bert:
            for p in self.model.parameters():
                p.requires_grad = False

        # share layer
        self.softmax_all_layer = nn.Softmax(-1)
        self.nn_dense = nn.Linear(self.model.config.hidden_size, 1)
        # use a truncated_normalizer to initialize the α.
        self.truncated_normal_(self.nn_dense.weight)
        self.act = nn.ReLU()
        self.pooler = nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size)
        self.pooler_activation = nn.Tanh()

        # is_humour
        self.subtask_1a = nn.Sequential(
            nn.Dropout(p=DROP_1A),
            nn.Linear(self.model.config.hidden_size, 2),
            nn.Softmax(dim=1)
        )

        # humor_rating
        self.subtask_1b = nn.Sequential(
            nn.Dropout(p=DROP_1B),
            nn.Linear(self.model.config.hidden_size, 1)
        )

        # humor_controversy
        self.subtask_1c = nn.Sequential(
            nn.Dropout(p=DROP_1C),
            nn.Linear(self.model.config.hidden_size, 2),
            nn.Softmax(dim=1)
        )

        # offense_rating
        self.subtask_2a = nn.Sequential(
            nn.Dropout(p=DROP_2A),
            nn.Linear(self.model.config.hidden_size, 1)
        )

    # this function is adapted form https://zhuanlan.zhihu.com/p/83609874
    def truncated_normal_(self, tensor, mean=0, std=0.02):
        with torch.no_grad():
            size = tensor.shape
            tmp = tensor.new_empty(size + (4,)).normal_()
            valid = (tmp < 2) & (tmp > -2)
            ind = valid.max(-1, keepdim=True)[1]
            tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
            tensor.data.mul_(std).add_(mean)
            return tensor

    def forward(self, input_ids, attention_mask):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        # calculate α_i
        layer_logits = []
        for layer in outputs.hidden_states[1:]:
            out = self.nn_dense(layer)
            layer_logits.append(self.act(out))

        # sum up layers by weighting
        layer_logits = torch.cat(layer_logits, axis=2)
        layer_dist = self.softmax_all_layer(layer_logits)
        seq_out = torch.cat([torch.unsqueeze(x, axis=2) for x in outputs.hidden_states[1:]], axis=2)
        all_layer_output = torch.matmul(torch.unsqueeze(layer_dist, axis=2), seq_out)
        all_layer_output = torch.squeeze(all_layer_output, axis=2)
        # take the [CLS] token output
        all_layer_output = self.pooler_activation(
            self.pooler(all_layer_output[:, 0])) if self.pooler is not None else None

        if not self.use_all_layer:
            # use [CLS] tokken output for the last layer encoder
            pooled_output = outputs.pooler_output
        else:
            pooled_output = all_layer_output

        output1 = self.subtask_1a(pooled_output)
        output2 = self.subtask_1b(pooled_output).clamp(0, 5)
        output3 = self.subtask_1c(pooled_output)
        output4 = self.subtask_2a(pooled_output).clamp(0, 5)
        return output1, output2, output3, output4


# Adapted from Venelin's blog: training step
def train_epoch(
        model,
        mtl,
        data_loader,
        loss_fn_CE,
        loss_fn_MSE,
        optimizer,
        device,
        scheduler,
        n_examples
):
    model = model.train()
    losses = []
    correct_predictions1 = 0
    correct_predictions2 = 0

    for d in data_loader:
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        targets = d["targets"].to(device)
        output1, output2, output3, output4 = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        output2 = output2[:, 0]
        output4 = output4[:, 0]

        _, preds1 = torch.max(output1, dim=1)
        mes1 = (output2 - targets[:, 1]).norm(2).pow(2)
        _, preds3 = torch.max(output3, dim=1)
        mes2 = (output4 - targets[:, 3]).norm(2).pow(2)

        if Weight_By_Uncertainty:
            loss = mtl(output1,
                       output2[preds1 == 1],
                       output3[preds1 == 1],
                       output4,
                       [targets[:, 0].type(torch.cuda.LongTensor), targets[:, 1][preds1 == 1],
                        targets[:, 2][preds1 == 1].type(torch.cuda.LongTensor), targets[:, 3]]
                       )
        else:
            loss = 0
            loss1 = loss_fn_CE(output1, targets[:, 0].type(torch.cuda.LongTensor))
            loss4 = loss_fn_MSE(output4, targets[:, 3])
            loss += WEIGHT_1A * loss1 + WEIGHT_2A * loss4
            if output2[preds1 == 1].numel():
                loss2 = loss_fn_MSE(output2[preds1 == 1], targets[:, 1][preds1 == 1])
                loss3 = loss_fn_CE(output3[preds1 == 1], targets[:, 2][preds1 == 1].type(torch.cuda.LongTensor))
                loss += WEIGHT_1B * loss2 + WEIGHT_1C * loss3

        correct_predictions1 += torch.sum(preds1 == targets[:, 0])
        acc1 = correct_predictions1.double() / n_examples
        correct_predictions2 += torch.sum(preds3 == targets[:, 2])
        acc2 = correct_predictions2.double() / n_examples

        losses.append(loss.item())
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
    return acc1, mes1, acc2, mes2, np.mean(losses)


# Adapted from Venelin's blog: eval step
def eval_model(model, mtl, data_loader, loss_fn_CE, loss_fn_MSE, device, n_examples):
    model = model.eval()
    losses = []
    correct_predictions1 = 0
    correct_predictions2 = 0

    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["targets"].to(device)
            output1, output2, output3, output4 = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            output2 = output2[:, 0]
            output4 = output4[:, 0]

            _, preds1 = torch.max(output1, dim=1)
            mes1 = (output2 - targets[:, 1]).norm(2).pow(2)
            _, preds3 = torch.max(output3, dim=1)
            mes2 = (output4 - targets[:, 3]).norm(2).pow(2)

            if Weight_By_Uncertainty:
                loss = mtl(output1,
                           output2[preds1 == 1],
                           output3[preds1 == 1],
                           output4,
                           [targets[:, 0].type(torch.cuda.LongTensor), targets[:, 1][preds1 == 1],
                            targets[:, 2][preds1 == 1].type(torch.cuda.LongTensor), targets[:, 3]]
                           )
            else:
                loss = 0
                loss1 = loss_fn_CE(output1, targets[:, 0].type(torch.cuda.LongTensor))
                loss4 = loss_fn_MSE(output4, targets[:, 3])
                loss += WEIGHT_1A * loss1 + WEIGHT_2A * loss4
                if output2[preds1 == 1].numel():
                    loss2 = loss_fn_MSE(output2[preds1 == 1], targets[:, 1][preds1 == 1])
                    loss3 = loss_fn_CE(output3[preds1 == 1], targets[:, 2][preds1 == 1].type(torch.cuda.LongTensor))
                    loss += WEIGHT_1B * loss2 + WEIGHT_1C * loss3

            correct_predictions1 += torch.sum(preds1 == targets[:, 0])
            acc1 = correct_predictions1.double() / n_examples
            correct_predictions2 += torch.sum(preds3 == targets[:, 2])
            acc2 = correct_predictions2.double() / n_examples
            losses.append(loss.item())
    return acc1, mes1, acc2, mes2, np.mean(losses)


# Adapted from Venelin's blog: predict the labels
def get_predictions(model, with_label, data_loader):
    model = model.eval()
    review_texts = []
    predictions = []
    p1 = []
    p2 = []
    p3 = []
    p4 = []
    if with_label:
        r1 = []
        r2 = []
        r3 = []
        r4 = []
        real_values = []

    with torch.no_grad():
        for d in data_loader:
            texts = d["review_text"]
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            output1, output2, output3, output4 = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            output2 = output2[:, 0]
            output4 = output4[:, 0]

            _, preds1 = torch.max(output1, dim=1)
            _, preds3 = torch.max(output3, dim=1)

            review_texts.extend(texts)
            p1.extend(preds1)
            p2.extend(output2)
            p3.extend(preds3)
            p4.extend(output4)
            if with_label:
                targets = d["targets"].to(device)
                r1.extend(targets[:, 0])
                r2.extend(targets[:, 1])
                r3.extend(targets[:, 2])
                r4.extend(targets[:, 3])

    p1 = torch.stack(p1).cpu()
    p2 = torch.stack(p2).cpu()
    p3 = torch.stack(p3).cpu()
    p4 = torch.stack(p4).cpu()
    predictions = [p1, p2, p3, p4]
    if with_label:
        r1 = torch.stack(r1).cpu()
        r2 = torch.stack(r2).cpu()
        r3 = torch.stack(r3).cpu()
        r4 = torch.stack(r4).cpu()
        real_values = [r1, r2, r3, r4]

        return review_texts, predictions, real_values
    else:
        return review_texts, predictions


class MultiTaskLossWrapper(nn.Module):
    def __init__(self, task_num, loss_function_CE):
        super(MultiTaskLossWrapper, self).__init__()
        self.task_num = task_num
        self.log_vars = nn.Parameter(torch.zeros((task_num)))
        self.loss_function_CE = loss_function_CE

    def forward(self, output1, output2, output3, output4, targets):

        precision1 = torch.exp(-self.log_vars[0])
        loss = torch.sum(precision1 / 2 * self.loss_function_CE(output1, targets[0]) + self.log_vars[0], -1)

        if output2.numel():
            precision2 = torch.exp(-2 * self.log_vars[1])
            loss += torch.sum(precision2 / 2 * (targets[1] - output2) ** 2. + self.log_vars[1], -1)

        if output2.numel():
            precision3 = torch.exp(-self.log_vars[1])
            loss += torch.sum(precision3 / 2 * self.loss_function_CE(output3, targets[2]) + self.log_vars[2], -1)

        precision4 = torch.exp(-2 * self.log_vars[3])
        loss += torch.sum(precision4 / 2 * (targets[3] - output4) ** 2. + self.log_vars[3], -1)

        loss = torch.mean(loss)

        return loss


if __name__ == '__main__':

    set_seed(RANDOM_SEED)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, output_hidden_states=True, return_dict=True)

    # load data
    df = pd.read_csv("./datas/task1/train/train.csv")
    df = df.fillna(0.0)
    df['list'] = df[df.columns[2:]].values.tolist()
    df = df[['text', 'list']]

    # Separat data to training set(0.9) validation set(0.05) and test set(0.05)
    class_names_1 = ['is_humor', 'not_humor']
    class_names_2 = ['is_CON', 'not_CON']
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

    train_data_loader = create_data_loader(df_train, True, tokenizer, MAX_LEN, BATCH_SIZE)
    val_data_loader = create_data_loader(df_val, True, tokenizer, MAX_LEN, BATCH_SIZE)
    test_data_loader = create_data_loader(df_test, True, tokenizer, MAX_LEN, BATCH_SIZE)
    data = next(iter(train_data_loader))

    # initialize model
    model = MyModel(use_all_layer=USE_ALL_LAYER)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model, device_ids=DEVIDE_IDS)

    model = model.to(device)
    input_ids = data['input_ids'].to(device)
    attention_mask = data['attention_mask'].to(device)

    # initialize optimizer
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, correct_bias=False)
    total_steps = len(train_data_loader) * EPOCHS

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )

    # initialize loss functions
    loss_fn_CE = nn.CrossEntropyLoss().to(device)
    loss_fn_MSE = nn.MSELoss().to(device)
    mtl = MultiTaskLossWrapper(4, loss_fn_CE).to(device)

    # training
    history = defaultdict(list)
    best_accuracy_1 = 0
    best_accuracy_2 = 0
    for epoch in range(EPOCHS):
        print(f'Epoch {epoch + 1}/{EPOCHS}')
        print('-' * 10)
        train_acc_1, train_mse_1, train_acc_2, train_mse_2, train_loss = train_epoch(
            model,
            mtl,
            train_data_loader,
            loss_fn_CE,
            loss_fn_MSE,
            optimizer,
            device,
            scheduler,
            len(df_train)
        )
        print(
            f'Train loss {train_loss} accuracy_1a {train_acc_1} accuracy_1c {train_acc_2} MSE_1b {train_mse_1} MSE_2a {train_mse_2}')

        val_acc_1, val_mse_1, val_acc_2, val_mse_2, val_loss = eval_model(
            model,
            mtl,
            val_data_loader,
            loss_fn_CE,
            loss_fn_MSE,
            device,
            len(df_val)
        )
        print(val_acc_1)
        print(
            f'Val   loss {val_loss} accuracy_1a {val_acc_1} accuracy_1c {val_acc_2} MSE_1b {val_mse_1} MSE_2a {val_mse_2}')

        print()
        history['train_acc_1'].append(train_acc_1)
        history['train_acc_2'].append(train_acc_2)
        history['train_loss'].append(train_loss)
        history['val_acc_1'].append(val_acc_1)
        history['val_acc_2'].append(val_acc_2)
        history['val_loss'].append(val_loss)

        # save the best model
        if val_acc_1 > best_accuracy_1:
            torch.save(model.state_dict(), 'best_model_state.bin')
            best_accuracy_1 = val_acc_1
            best_accuracy_2 = val_acc_2
        elif val_acc_1 == best_accuracy_1:
            if val_acc_2 > best_accuracy_2:
                torch.save(model.state_dict(), 'best_model_state.bin')
                best_accuracy_1 = val_acc_1
                best_accuracy_2 = val_acc_2

    # load task dataset
    df = pd.read_csv("./public_test.csv")
    df = df[['text']]

    class_names_1 = ['is_humor', 'not_humor']
    class_names_2 = ['is_CON', 'not_CON']

    final_test_data_loader = create_data_loader(df, False, tokenizer, MAX_LEN, BATCH_SIZE)

    # use the best model we have to predict the result
    model.load_state_dict(torch.load('./best_model_state.bin'))

    model = model.to(device)

    y_review_texts, y_pred = get_predictions(
        model,
        False,
        final_test_data_loader
    )

    # product a CVS file for task result
    result = pd.read_csv("./public_test.csv", header=0)
    label_1 = pd.DataFrame({'is_humor': y_pred[0]})
    label_1 = label_1[['is_humor']]
    label_2 = pd.DataFrame({'humor_rating': y_pred[1]})
    label_2 = label_2[['humor_rating']]
    label_3 = pd.DataFrame({'humor_controversy': y_pred[2]})
    label_3 = label_3[['humor_controversy']]
    label_4 = pd.DataFrame({'offense_rating': y_pred[3]})
    label_4 = label_4[['offense_rating']]
    result = pd.concat([result, label_1, label_2, label_3, label_4], axis=1)
    print(result)
    result.to_csv("task_result.csv")

    # product a CVS file for result of training data
    y_review_texts, y_pred, y_test = get_predictions(
        model,
        True,
        test_data_loader
    )
    # print out the result matrix
    print(classification_report(y_test[0], y_pred[0], target_names=class_names_1))
    print(mean_squared_error(y_test[1][y_test[0] == 1], y_pred[1][y_test[0] == 1]))
    print(classification_report(y_test[2][y_test[0] == 1], y_pred[2][y_test[0] == 1], target_names=class_names_2))
    print(mean_squared_error(y_test[3], y_pred[3]))

    text = pd.DataFrame({'text': y_review_texts})
    text = text[['text']]
    label_1 = pd.DataFrame({'is_humor': y_pred[0]})
    label_1 = label_1[['is_humor']]
    label_2 = pd.DataFrame({'humor_rating': y_pred[1]})
    label_2 = label_2[['humor_rating']]
    label_3 = pd.DataFrame({'humor_controversy': y_pred[2]})
    label_3 = label_3[['humor_controversy']]
    label_4 = pd.DataFrame({'offense_rating': y_pred[3]})
    label_4 = label_4[['offense_rating']]
    target_1 = pd.DataFrame({'target_1': y_test[0]})
    target_1 = target_1[['target_1']]
    target_2 = pd.DataFrame({'target_2': y_test[1]})
    target_2 = target_2[['target_2']]
    target_3 = pd.DataFrame({'target_3': y_test[2]})
    target_3 = target_3[['target_3']]
    target_4 = pd.DataFrame({'target_4': y_test[3]})
    target_4 = target_4[['target_4']]
    own_result = pd.concat([text, label_1, target_1, label_2, target_2, label_3, target_3, label_4, target_4], axis=1)
    own_result.to_csv("result.csv")
