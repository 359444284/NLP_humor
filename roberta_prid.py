import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AlbertTokenizer, AlbertModel, AlbertConfig, AlbertForPreTraining, AdamW, \
     get_linear_schedule_with_warmup, AlbertForSequenceClassification, AutoModel, AutoTokenizer, \
     RobertaConfig, RobertaModel, RobertaTokenizer
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from collections import defaultdict
import pandas as pd
import random
import matplotlib as plt

RANDOM_SEED = 70
BATCH_SIZE = 8
MAX_LEN = 150
EPOCHS = 10
torch.cuda.current_device()
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed(seed)


# class GPReviewDataset(Dataset):
#     def __init__(self, reviews, tokenizer, max_len):
#         self.reviews = reviews
#         self.tokenizer = tokenizer
#         self.max_len = max_len

#     def __len__(self):
#         return len(self.reviews)

#     def __getitem__(self, item):
#         review = str(self.reviews[item])
#         encoding = self.tokenizer.encode_plus(
#               review,
#               add_special_tokens=True,
#               max_length=self.max_len,
#               return_token_type_ids=False,
#               padding='max_length',
#               return_attention_mask=True,
#               return_tensors='pt',
#         )

#         return{
#             'review_text': review,
#             'input_ids': encoding['input_ids'].flatten(),
#             'attention_mask': encoding['attention_mask'].flatten(),
#         }

class GPReviewDataset(Dataset):
    def __init__(self, dataframe, with_label, tokenizer, max_len):
        self.reviews=dataframe.text.to_numpy()
        self.with_label = with_label
        if with_label:
          self.targets=dataframe.list.to_numpy()
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

# def create_data_loader1(df, tokenizer, max_len, batch_size):
#     ds = GPReviewDataset1(
#         reviews=df.text.to_numpy(),
#         targets=df.list.to_numpy(),
#         tokenizer=tokenizer,
#         max_len=max_len
#     )

#     return DataLoader(
#         ds,
#         batch_size=batch_size,
#         num_workers=4
#     )

class MyModel(nn.Module):
    def __init__(self, freeze_bert=False):
        super(MyModel, self).__init__()
#         albert_xxlarge_configuration = AlbertConfig(output_hidden_states=True, output_attentions=True, return_dict=True)
#         self.model = AlbertModel.from_pretrained(pretrained_model_name_or_path=MODEL_PATH, config=albert_xxlarge_configuration)
#         self.model = RobertaModel.from_pretrained(pretrained_model_name_or_path=MODEL_PATH, output_hidden_states=True, output_attentions=True, return_dict=True)
        self.model = AutoModel.from_pretrained(pretrained_model_name_or_path=MODEL_PATH, output_hidden_states=True, output_attentions=True, return_dict=True)
        if freeze_bert:
            for p in self.model.parameters():
                p.requires_grad = False

        # share layer
        self.softmax_all_layer = nn.Softmax(-1)
        self.pooler = nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size)
        self.pooler_activation = nn.Tanh()
        self.nn_dense = nn.Linear(self.model.config.hidden_size, 1)
        self.truncated_normal_(self.nn_dense.weight)
        self.act = nn.ReLU()
        
#         self.conv = nn.Conv2d(in_channels=13, out_channels=13, kernel_size=(3, 4096), padding=True)
#         self.relu = nn.ReLU()
#         self.pool = nn.MaxPool2d(kernel_size=3, stride=1)
        self.dropout = nn.Dropout(0.2)
#         self.fc = nn.Linear(1924, 3) # before : 442 with max_length 36 # 806 with max_length 64
#         self.flat = nn.Flatten()
#         self.fc_size = 1924

        # is_humour
        self.tower_1 = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(self.model.config.hidden_size, 2),
#             nn.Linear(self.fc_size, 2),
            nn.Softmax(dim=1)
        )
        
        # humor_rating
        self.tower_2 = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(self.model.config.hidden_size, 1)
#             nn.Linear(self.fc_size, 1)
        )
        
        # humor_controversy
        self.tower_3 = nn.Sequential(
            nn.Dropout(p=0.3),
#             nn.Linear(self.fc_size, 2),
            nn.Linear(self.model.config.hidden_size, 2),
            nn.Softmax(dim=1)
        )
        
        # offense_rating
        self.tower_4 = nn.Sequential(
            nn.Dropout(p=0.3),
#             nn.Linear(self.fc_size, 1)
            nn.Linear(self.model.config.hidden_size, 1)
        )
      
    def truncated_normal_(self,tensor,mean=0,std=0.02):
        with torch.no_grad():
            size = tensor.shape
            tmp = tensor.new_empty(size+(4,)).normal_()
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
#         layer_logits = []
#         for layer in outputs.hidden_states[1:]:
#             out = self.nn_dense(layer)
#             layer_logits.append(self.act(out))
# #             layer_logits.append(out)

#         layer_logits = torch.cat(layer_logits, axis=2)
#         layer_dist = self.softmax_all_layer(layer_logits)
#         seq_out = torch.cat([torch.unsqueeze(x, axis=2) for x in outputs.hidden_states[1:]], axis=2)
#         pooled_output = torch.matmul(torch.unsqueeze(layer_dist, axis=2), seq_out)
#         pooled_output = torch.squeeze(pooled_output, axis=2)
#         pooled_output = self.pooler_activation(self.pooler(pooled_output[:, 0])) if self.pooler is not None else None
        pooled_output = outputs.pooler_output

        output1 = self.tower_1(pooled_output)
        output2 = self.tower_2(pooled_output).clamp(0, 5)
        output3 = self.tower_3(pooled_output)
        output4 = self.tower_4(pooled_output).clamp(0, 5)
        return output1, output2, output3, output4
#         return output3


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
        output2 = output2[:,0]
        output4 = output4[:,0]
        _, preds1 = torch.max(output1, dim=1)
        mes1 = (output2 - targets[:,1]).norm(2).pow(2)
        _, preds3 = torch.max(output3, dim=1)
        mes2 = (output4 - targets[:,3]).norm(2).pow(2)
        
        loss, log_vars = mtl(output1,
                             output2[preds1 == 1],
                             output3[preds1 == 1],
                             output4,
                             [targets[:,0].type(torch.cuda.LongTensor), targets[:,1][preds1 == 1], targets[:,2][preds1 == 1].type(torch.cuda.LongTensor), targets[:,3]]
                         )

        correct_predictions1 += torch.sum(preds1 == targets[:,0])
        acc1 = correct_predictions1.double() / n_examples
        correct_predictions2 += torch.sum(preds3 == targets[:,2])
        acc2 = correct_predictions2.double() / n_examples


        losses.append(loss.item())
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
    return acc1, mes1, acc2, mes2, np.mean(losses)

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
            output2 = output2[:,0]
            output4 = output4[:,0]
         
            _, preds1 = torch.max(output1, dim=1)
            mes1 = (output2 - targets[:,1]).norm(2).pow(2)
            _, preds3 = torch.max(output3, dim=1)
            mes2 = (output4 - targets[:,3]).norm(2).pow(2)

            loss, log_vars = mtl(output1,
                             output2[preds1 == 1],
                             output3[preds1 == 1],
                             output4,
                             [targets[:,0].type(torch.cuda.LongTensor), targets[:,1][preds1 == 1], targets[:,2][preds1 == 1].type(torch.cuda.LongTensor), targets[:,3]]
                         )

            correct_predictions1 += torch.sum(preds1 == targets[:,0])
            acc1 = correct_predictions1.double() / n_examples
            correct_predictions2 += torch.sum(preds3 == targets[:,2])
            acc2 = correct_predictions2.double() / n_examples

            losses.append(loss.item())
    return acc1, mes1, acc2, mes2, np.mean(losses)

def get_predictions(model, data_loader):
    model = model.eval()
    review_texts = []
    predictions = []
    p1 = []
    p2 = []
    p3 = []
    p4 = []

    with torch.no_grad():
        for d in data_loader:
            texts = d["review_text"]
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            output1, output2, output3, output4 = model(
#             output1 = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            output2 = output2[:,0]
            output4 = output4[:,0]
            
            _, preds1 = torch.max(output1, dim=1)
            _, preds3 = torch.max(output3, dim=1)
            
            review_texts.extend(texts)
            p1.extend(preds1)
            p2.extend(output2)
            p3.extend(preds3)
            p4.extend(output4)

    p1 = torch.stack(p1).cpu()
    p2 = torch.stack(p2).cpu()
    p3 = torch.stack(p3).cpu()
    p4 = torch.stack(p4).cpu()
    predictions = [p1,p2,p3,p4]
#     predictions = [p1]

    return review_texts, predictions

def get_predictions1(model, data_loader):
    model = model.eval()
    review_texts = []
    predictions = []
    p1 = []
    p2 = []
    p3 = []
    p4 = []
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
            targets = d["targets"].to(device)
            output1, output2, output3, output4 = model(
#             output1 = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            output2 = output2[:,0]
            output4 = output4[:,0]
            
            _, preds1 = torch.max(output1, dim=1)
            _, preds3 = torch.max(output3, dim=1)
            
            review_texts.extend(texts)
            p1.extend(preds1)
            p2.extend(output2)
            p3.extend(preds3)
            p4.extend(output4)
            r1.extend(targets[:,0])
            r2.extend(targets[:,1])
            r3.extend(targets[:,2])
            r4.extend(targets[:,3])
    p1 = torch.stack(p1).cpu()
    p2 = torch.stack(p2).cpu()
    p3 = torch.stack(p3).cpu()
    p4 = torch.stack(p4).cpu()
    r1 = torch.stack(r1).cpu()
    r2 = torch.stack(r2).cpu()
    r3 = torch.stack(r3).cpu()
    r4 = torch.stack(r4).cpu()
    predictions = [p1,p2,p3,p4]
    real_values = [r1,r2,r3,r4]

    return review_texts, predictions, real_values


class MultiTaskLossWrapper(nn.Module):
    def __init__(self, task_num, loss_function_CE):
        super(MultiTaskLossWrapper, self).__init__()
        self.task_num = task_num
        self.log_vars = nn.Parameter(torch.zeros((task_num)))
        self.loss_function_CE = loss_function_CE
        
    def forward(self, output1, output2, output3, output4, targets):
        

        precision1 = torch.exp(-2 * self.log_vars[0])
        loss = torch.sum(precision1/2 * self.loss_function_CE(output1, targets[0]) + self.log_vars[0], -1)
        
        if output2.numel():
            precision2 = torch.exp(-2 * self.log_vars[1])
            loss += torch.sum(precision2/2 * (targets[1] - output2) ** 2. + self.log_vars[1], -1)
        
        if output2.numel():
            precision3 = torch.exp(-self.log_vars[2])
            loss += torch.sum(precision3/2 * self.loss_function_CE(output3, targets[2]) + self.log_vars[2], -1)
        
        precision4 = torch.exp(-2 * self.log_vars[3])
        loss += torch.sum(precision4/2 * (targets[3] - output4) ** 2. + self.log_vars[3], -1)

        loss = torch.mean(loss)

        return loss, self.log_vars.data.tolist()

if __name__ == '__main__':

     set_seed(RANDOM_SEED)

#      MODEL_PATH = 'roberta-large'
     MODEL_PATH = 'albert-xxlarge-v2'
     tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, output_hidden_states=True, return_dict=True)

     df = pd.read_csv("./public_test.csv")
     df = df[['text']]

     class_names_1 = ['is_humor', 'not_humor']
     class_names_2 = ['is_CON', 'not_CON']

     test_data_loader = create_data_loader(df, False, tokenizer, MAX_LEN, BATCH_SIZE)

     model = MyModel()
     if torch.cuda.device_count()>1:
          model=nn.DataParallel(model,device_ids=[1,2])


     model.load_state_dict(torch.load('./best_model_state.bin'))

     model = model.to(device)

     optimizer = AdamW(model.parameters(), lr=2e-6, correct_bias=False)

     loss_fn_CE = nn.CrossEntropyLoss().to(device)
     loss_fn_MSE = nn.MSELoss().to(device)
     mtl = MultiTaskLossWrapper(4,loss_fn_CE).to(device)

     y_review_texts, y_pred= get_predictions(
          model,
          test_data_loader
     )

     result = pd.read_csv("./public_test.csv", header=0)
     label_1 = pd.DataFrame({'is_humor':y_pred[0]})
     label_1 = label_1[['is_humor']]
     label_2 = pd.DataFrame({'humor_rating':y_pred[1]})
     label_2 = label_2[['humor_rating']]
     label_3 = pd.DataFrame({'humor_controversy':y_pred[2]})
     label_3 = label_3[['humor_controversy']]
     label_4 = pd.DataFrame({'offense_rating':y_pred[3]})
     label_4 = label_4[['offense_rating']]
     result = pd.concat([result,label_1,label_2,label_3,label_4],axis=1)
#      result = pd.concat([result,label_1],axis=1)
     print(result)
     result.to_csv("task1a.csv")

     df1 = pd.read_csv("./datas/task1/train/train.csv")
     df1 = df1.fillna(0.0)
     df1['list'] = df1[df1.columns[2:]].values.tolist()
     df1 = df1[['text', 'list']]
     df_train1, df_test1 = train_test_split(
       df1,
       test_size=0.1,
       random_state=RANDOM_SEED
     )
     df_val1, df_test1 = train_test_split(
       df_test1,
       test_size=0.5,
       random_state=RANDOM_SEED
     )

     train_data_loader = create_data_loader(df_train1, True, tokenizer, MAX_LEN, BATCH_SIZE)
     val_data_loader = create_data_loader(df_val1, True, tokenizer, MAX_LEN, BATCH_SIZE)
     own_test_data_loader = create_data_loader(df_test1, True, tokenizer, MAX_LEN, BATCH_SIZE)
     data1 = next(iter(train_data_loader))


     y_review_texts, y_pred, y_test = get_predictions1(
       model,
       own_test_data_loader
     )

     print(classification_report(y_test[0], y_pred[0], target_names=class_names_1))
     print((y_test[1] - y_pred[1]).norm(2).pow(2))
     print(classification_report(y_test[2], y_pred[2], target_names=class_names_2))
     print((y_test[3] - y_pred[3]).norm(2).pow(2))
     
     text = pd.DataFrame({'text':y_review_texts})
     text = text[['text']]
     label_1 = pd.DataFrame({'is_humor':y_pred[0]})
     label_1 = label_1[['is_humor']]
     label_2 = pd.DataFrame({'humor_rating':y_pred[1]})
     label_2 = label_2[['humor_rating']]
     label_3 = pd.DataFrame({'humor_controversy':y_pred[2]})
     label_3 = label_3[['humor_controversy']]
     label_4 = pd.DataFrame({'offense_rating':y_pred[3]})
     label_4 = label_4[['offense_rating']]
     target_1 = pd.DataFrame({'target_1':y_test[0]})
     target_1 = target_1[['target_1']]
     target_2 = pd.DataFrame({'target_2':y_test[1]})
     target_2 = target_2[['target_2']]
     target_3 = pd.DataFrame({'target_3':y_test[2]})
     target_3 = target_3[['target_3']]
     target_4 = pd.DataFrame({'target_4':y_test[3]})
     target_4 = target_4[['target_4']]
     own_result = pd.concat([text, label_1, target_1, label_2, target_2, label_3, target_3, label_4, target_4],axis=1)
#      result = pd.concat([result,label_1],axis=1)
     print(result)
     own_result.to_csv("result.csv")


