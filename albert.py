import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AlbertTokenizer, AlbertModel, AlbertConfig, AlbertForPreTraining, AdamW, \
    get_linear_schedule_with_warmup, AlbertForSequenceClassification, AutoModel, AutoTokenizer, \
    RobertaConfig, RobertaModel, RobertaTokenizer
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from collections import defaultdict
import pandas as pd
import random
import matplotlib.pyplot as plt
# 70
RANDOM_SEED = 70
BATCH_SIZE = 8
MAX_LEN = 150
EPOCHS = 15
torch.cuda.current_device()
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')


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
            'targets': torch.tensor(target, dtype=torch.float)
        }

def create_data_loader(df, tokenizer, max_len, batch_size):
    ds = GPReviewDataset(
        reviews=df.text.to_numpy(),
        targets=df.list.to_numpy(),
        tokenizer=tokenizer,
        max_len=max_len
    )

    return DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=4
    )

class MyModel(nn.Module):
    def __init__(self, freeze_bert=False):
        super(MyModel, self).__init__()
        albert_xxlarge_configuration = AlbertConfig(output_hidden_states=True, output_attentions=True, add_pooling_layer=False)
        self.model = AlbertModel.from_pretrained(pretrained_model_name_or_path=MODEL_PATH, config=albert_xxlarge_configuration)
        #self.model = RobertaModel.from_pretrained(pretrained_model_name_or_path=MODEL_PATH, output_hidden_states=True, output_attentions=True)
        #self.model = AutoModel.from_pretrained(pretrained_model_name_or_path=MODEL_PATH)
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

        # is_humour
        self.tower_1 = nn.Sequential(
            nn.Dropout(p=0.8),
            nn.Linear(self.model.config.hidden_size, 2),
            nn.Softmax(dim=1)
        )
        
        # humor_rating
        self.tower_2 = nn.Sequential(
            nn.Dropout(p=0.8),
            nn.Linear(self.model.config.hidden_size, 1)
        )
        
        # humor_controversy
        self.tower_3 = nn.Sequential(
            nn.Dropout(p=0.8),
            nn.Linear(self.model.config.hidden_size, 2),
            nn.Softmax(dim=1)
        )
        
        # offense_rating
        self.tower_4 = nn.Sequential(
            nn.Dropout(p=0.8),
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
        layer_logits = []
        for layer in outputs[2][1:]:
            out = self.nn_dense(layer)
            layer_logits.append(self.act(out))
#             layer_logits.append(out)

        layer_logits = torch.cat(layer_logits, axis=2)
        layer_dist = self.softmax_all_layer(layer_logits)
        seq_out = torch.cat([torch.unsqueeze(x, axis=2) for x in outputs[2][1:]], axis=2)
        pooled_output = torch.matmul(torch.unsqueeze(layer_dist, axis=2), seq_out)
        pooled_output = torch.squeeze(pooled_output, axis=2)

        pooled_output = self.pooler_activation(self.pooler(pooled_output[:, 0])) if self.pooler is not None else None
#         pooled_output = outputs[1]


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
#         output1 = model(
          input_ids=input_ids,
          attention_mask=attention_mask
        )
        output2 = output2[:,0]
        output4 = output4[:,0]

#         output1, output2 = model(
#           input_ids=input_ids,
#           attention_mask=attention_mask
#         )

        _, preds1 = torch.max(output1, dim=1)
        mes1 = (output2 - targets[:,1]).norm(2).pow(2)
        _, preds3 = torch.max(output3, dim=1)
        mes2 = (output4 - targets[:,3]).norm(2).pow(2)
        
#         _, preds1 = torch.max(output1, dim=1)
#         _, preds3 = torch.max(output2, dim=1)

#         loss, log_vars = mtl(output1,
#                              output2[preds1 == 1],
#                              output3[preds1 == 1],
#                              output4,
#                              [targets[:,0].type(torch.cuda.LongTensor), targets[:,1][preds1 == 1], targets[:,2][preds1 == 1].type(torch.cuda.LongTensor), targets[:,3]]
#                          )
        loss = 0
#         loss = loss_fn_CE(output1, targets[:,2].type(torch.cuda.LongTensor))
        loss1 = loss_fn_CE(output1, targets[:,0].type(torch.cuda.LongTensor))
        loss4 = loss_fn_MSE(output4, targets[:,3])
        loss += 0.05*loss1 + 0.045*loss4
        if output2[preds1 == 1].numel():
            
            loss2 = loss_fn_MSE(output2[preds1 == 1], targets[:,1][preds1 == 1])
            
            loss3 = loss_fn_CE(output3[preds1 == 1], targets[:,2][preds1 == 1].type(torch.cuda.LongTensor))
            loss += 0.05*loss2 + 0.45*loss3
            loss = loss
        else:
            loss = loss
        
        
        
        
#         loss, log_vars = mtl(output1,
#                              output2[preds1 == 1],
#                              [targets[:,0].type(torch.cuda.LongTensor), targets[:,1][preds1 == 1], targets[:,2][preds1 == 1].type(torch.cuda.LongTensor), targets[:,3]]
#                              )

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
#     return acc1, np.mean(losses)


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
#             output1 = model(
              input_ids=input_ids,
              attention_mask=attention_mask
            )
            output2 = output2[:,0]
            output4 = output4[:,0]

#             output1, output2 = model(
#               input_ids=input_ids,
#               attention_mask=attention_mask
#             )

            _, preds1 = torch.max(output1, dim=1)
            mes1 = (output2 - targets[:,1]).norm(2).pow(2)
            _, preds3 = torch.max(output3, dim=1)
            mes2 = (output4 - targets[:,3]).norm(2).pow(2)
        
#             _, preds1 = torch.max(output1, dim=1)
#             _, preds3 = torch.max(output2, dim=1)
        
#             loss, log_vars = mtl(output1,
#                                  output2[preds1 == 1],
#                                  output3[preds1 == 1],
#                                  output4,
#                                  [targets[:,0].type(torch.cuda.LongTensor), targets[:,1][preds1 == 1], targets[:,2][preds1 == 1].type(torch.cuda.LongTensor), targets[:,3]]
#                              )
            loss = 0
#             loss = loss_fn_CE(output1, targets[:,2].type(torch.cuda.LongTensor))
            loss1 = loss_fn_CE(output1, targets[:,0].type(torch.cuda.LongTensor))
            loss4 = loss_fn_MSE(output4, targets[:,3])
            loss += 0.05*loss1 + 0.45*loss4
            if output2[preds1 == 1].numel():

                loss2 = loss_fn_MSE(output2[preds1 == 1], targets[:,1][preds1 == 1])

                loss3 = loss_fn_CE(output3[preds1 == 1], targets[:,2][preds1 == 1].type(torch.cuda.LongTensor))
                loss += 0.05*loss2 + 0.45*loss3
                loss = loss
            else:
                loss = loss
        
#             loss, log_vars = mtl(output1,
#                                  output2[preds1 == 1],
#                                  [targets[:,0].type(torch.cuda.LongTensor), targets[:,1][preds1 == 1], targets[:,2][preds1 == 1].type(torch.cuda.LongTensor), targets[:,3]]
#                                  )
            correct_predictions1 += torch.sum(preds1 == targets[:,0])
            acc1 = correct_predictions1.double() / n_examples
            correct_predictions2 += torch.sum(preds3 == targets[:,2])
            acc2 = correct_predictions2.double() / n_examples

            losses.append(loss.item())
    return acc1, mes1, acc2, mes2, np.mean(losses)
#     return acc1, np.mean(losses)

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
            
            review_texts.extend(texts)
            predictions.extend([output1, mes1, output3, mes2])
            prediction_probs.extend([output1, output2, output3, output4])
            real_values.extend([targets[:,0], targets[:,1], targets[:,2], targets[:,3]])
    predictions = torch.stack(predictions).cpu()
    prediction_probs = torch.stack(prediction_probs).cpu()
    real_values = torch.stack(real_values).cpu()
    return review_texts, predictions, prediction_probs, real_values

class MultiTaskLossWrapper(nn.Module):
    def __init__(self, task_num, loss_function_CE):
        super(MultiTaskLossWrapper, self).__init__()
        self.task_num = task_num
        self.log_vars = nn.Parameter(torch.zeros((task_num)))
        self.loss_function_CE = loss_function_CE
        
    def forward(self, output1, output2, output3, output4, targets):
#     def forward(self, output1, output2, targets):
        

        precision1 = torch.exp(-2 * self.log_vars[0])
        loss = torch.sum(precision1/2 * self.loss_function_CE(output1, targets[0]) + self.log_vars[0], -1)
        
        if output2.numel():
            precision2 = torch.exp(-2 * self.log_vars[1])
            loss += torch.sum(precision2/2 * (targets[1] - output2) ** 2. + self.log_vars[1], -1)
        
#         if output2.numel():
#             precision3 = torch.exp(-self.log_vars[1])
#             loss += torch.sum(precision3/2 * self.loss_function_CE(output3, targets[2]) + self.log_vars[2], -1)
        
        precision4 = torch.exp(-2 * self.log_vars[3])
        loss += torch.sum(precision4/2 * (targets[3] - output4) ** 2. + self.log_vars[3], -1)

        loss = torch.mean(loss)

        return loss, self.log_vars.data.tolist()

class ScheduledOptim(object):
    '''A wrapper class for learning rate scheduling'''
 
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.lr = self.optimizer.param_groups[0]['lr']
        self.current_steps = 0
 
    def step(self):
        "Step by the inner optimizer"
        self.current_steps += 1
        self.optimizer.step()
 
    def zero_grad(self):
        "Zero out the gradients by the inner optimizer"
        self.optimizer.zero_grad()
 
    def set_learning_rate(self, lr):
        self.lr = lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
 
    @property
    def learning_rate(self):
        return self.lr
 
 
def find_lr():
    pass
 
 
def train_model():
    pass
 

if __name__ == '__main__':

    set_seed(RANDOM_SEED)

    MODEL_PATH = 'albert-xxlarge-v2'
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, output_hidden_states=True, return_dict=True)

    df = pd.read_csv("./datas/task1/train/train.csv")
    df = df.fillna(0.0)
    df['list'] = df[df.columns[2:]].values.tolist()
    df = df[['text', 'list']]

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

    train_data_loader = create_data_loader(df_train, tokenizer, MAX_LEN, BATCH_SIZE)
    val_data_loader = create_data_loader(df_val, tokenizer, MAX_LEN, BATCH_SIZE)
    test_data_loader = create_data_loader(df_test, tokenizer, MAX_LEN, BATCH_SIZE)
    data = next(iter(train_data_loader))

    model = MyModel()
    #model.load_state_dict(torch.load('./best_model_state.bin'))

    if torch.cuda.device_count()>1:
      model=nn.DataParallel(model,device_ids=[1,2])

    model = model.to(device)
    input_ids = data['input_ids'].to(device)
    attention_mask = data['attention_mask'].to(device)

    optimizer = AdamW(model.parameters(), lr=2e-6, correct_bias=False)
    total_steps = len(train_data_loader) * EPOCHS

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )
    
    loss_fn_CE = nn.CrossEntropyLoss().to(device)
    loss_fn_MSE = nn.MSELoss().to(device)
    mtl = MultiTaskLossWrapper(4,loss_fn_CE).to(device)
#     basic_optim = AdamW(model.parameters(), lr=1e-6, correct_bias=False)
#     optimizer1 = ScheduledOptim(basic_optim)
    
#     lr_mult = (1 / 1e-5) ** (1 / 100)
#     lr = []
#     losses = []
#     best_loss = 1e9
#     for d in train_data_loader:
#         texts = d["review_text"]
#         input_ids = d["input_ids"].to(device)
#         attention_mask = d["attention_mask"].to(device)
#         targets = d["targets"].to(device)
#         output1, output2, output3, output4 = model(
#                 input_ids=input_ids,
#                 attention_mask=attention_mask
#             )
#         output2 = output2[:,0]
#         output4 = output4[:,0]
#         _, preds1 = torch.max(output1, dim=1)
#         mes1 = (output2 - targets[:,1]).norm(2).pow(2)
#         _, preds3 = torch.max(output3, dim=1)
#         mes2 = (output4 - targets[:,3]).norm(2).pow(2)
        
#         loss, log_vars = mtl(preds1,
#                              output2[preds1 == 1],
#                              preds3[preds1 == 1],
#                              output4,
#                              [targets[:,0], targets[:,1][preds1 == 1], targets[:,2][preds1 == 1], targets[:,3]]
#                          )
#         # backward
#         optimizer1.zero_grad()
#         loss.backward()
#         optimizer1.step()
#         lr.append(optimizer1.learning_rate)
#         losses.append(loss.item())
#         optimizer1.set_learning_rate(optimizer1.learning_rate * lr_mult)
#         if loss.item() < best_loss:
#             best_loss = loss.item()
#         if loss.item() > 100 * best_loss or optimizer1.learning_rate > 1.:
#             break
    

#     print([lr, losses])


    history = defaultdict(list)
    best_accuracy_1 = 0
    best_accuracy_2 = 0
    for epoch in range(EPOCHS):
        print(f'Epoch {epoch + 1}/{EPOCHS}')
        print('-' * 10)
        train_acc_1, train_mse_1, train_acc_2, train_mse_2, train_loss = train_epoch(
#         train_acc_1, train_loss = train_epoch(
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
        print(f'Train loss {train_loss} accuracy1 {train_acc_1} accuracy2 {train_acc_2}')
#         print(f'Train loss {train_loss} accuracy1 {train_acc_1}')
        val_acc_1, val_mse_1, val_acc_2, val_mse_1, val_loss = eval_model(
#         val_acc_1, val_loss = eval_model(
            model,
            mtl,
            val_data_loader,
            loss_fn_CE,
            loss_fn_MSE,
            device,
            len(df_val)
        )
        print(f'Val   loss {val_loss} accuracy1 {val_acc_1} accuracy2 {val_acc_2}')
#         print(f'Val   loss {val_loss} accuracy1 {val_acc_1}')
        print()
        history['train_acc_1'].append(train_acc_1)
        history['train_acc_2'].append(train_acc_2)
        history['train_loss'].append(train_loss)
        history['val_acc_1'].append(val_acc_1)
        history['val_acc_2'].append(val_acc_2)
        history['val_loss'].append(val_loss)

        if val_acc_2 > best_accuracy_2:
            torch.save(model.state_dict(), 'best_model_state.bin')
            best_accuracy_1 = val_acc_1
            best_accuracy_2 = val_acc_2
        elif val_acc_2 == best_accuracy_2:
            if val_acc_1 > best_accuracy_1:
                torch.save(model.state_dict(), 'best_model_state.bin')
                best_accuracy_1 = val_acc_1
                best_accuracy_2 = val_acc_2




    y_review_texts, y_pred, y_pred_probs, y_test = get_predictions(
        model,
        test_data_loader
    )

    print(classification_report(y_test[:,0], y_pred[:,0], target_names=class_names_1))
    print((y_test[:,1] - y_pred[:,1]).norm(2).pow(2))
    print(classification_report(y_test[:,2], y_pred[:,2], target_names=class_names_2))
    print((y_test[:,3] - y_pred[:,3]).norm(2).pow(2))

