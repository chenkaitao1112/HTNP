import os
import random
from sklearn.metrics import f1_score
import torch.nn.functional as F
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"
import time
import torch.nn as nn
import pickle
from config import args
import numpy as np
import torch
import copy
import os
import torch.utils.data as Data
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from model import  Predictor



class Trainer:
    def __init__(self, f, seed):
        self.device = torch.device('cuda:{}'.format(args.gpu))
        self.fold = f
        self.warmup_steps = 10
        self.seed = seed
        self.model = Predictor(
            eventlog=args.eventlog,
            d_model=args.d_model,
            f=self.fold
        ).to(self.device)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.criterion_reg = nn.SmoothL1Loss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
        self.scheduler_cosine = CosineAnnealingLR(self.optimizer, args.epochs - self.warmup_steps)
        self.scheduler_warmup = LambdaLR(self.optimizer, lr_lambda=lambda epoch: epoch / self.warmup_steps)

    @staticmethod
    def Union(lst1, lst2):
        final_list = lst1 + lst2
        return final_list

    def train(self, list_view_train, num_view, cat_view, epoch):
        self.model.train()
        torch_dataset_train = Data.TensorDataset(*list_view_train)
        loader_train = Data.DataLoader(
            dataset=torch_dataset_train,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
        )
        att_str = cat_view + num_view + ['y', 'y_remain']

        total_loss = 0.0
        total_samples = 0
        total_correct = 0

        for batch_num, data in enumerate(loader_train):
            att = [data[i].to(self.device) for i in range(len(att_str))]
            bs = att[0].size(0)

            self.optimizer.zero_grad()
            logits, enc_self_attns, remain_pred = self.model(att_str, att)


            labels_cls = att[-2].to(torch.long)
            labels_reg = att[-1].to(torch.float32)

            loss_cls = self.criterion(logits, labels_cls)
            loss_reg = self.criterion_reg(remain_pred.squeeze(), labels_reg)
            alpha = 0.01
            batch_loss = loss_cls +  alpha * loss_reg

            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.optimizer.step()

            total_loss += batch_loss.item() * bs
            total_samples += bs
            total_correct += (logits.argmax(dim=1) == labels_cls).sum().item()

        avg_loss = total_loss / max(total_samples, 1)
        acc = total_correct / max(total_samples, 1)

        print(f'\tLoss: {avg_loss:.5f}(train)\t|\tAcc: {acc * 100:.2f}%(train)\tsampleNumber:{total_samples:.0f}')


    def eval(self, eval_model, list_view_valid, num_view, cat_view):
        eval_model.eval()
        torch_dataset_valid = Data.TensorDataset(*list_view_valid)
        loader_valid = Data.DataLoader(
            dataset=torch_dataset_valid,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
        )
        att_str = cat_view + num_view + ['y', 'y_remain']

        total_loss = 0.0
        total_samples = 0
        total_correct = 0

        with torch.no_grad():
            for batch_num, data in enumerate(loader_valid):
                att = [data[i].to(self.device) for i in range(len(att_str))]
                bs = att[0].size(0)

                logits, enc_self_attns, remain_pred = eval_model(att_str, att)

                labels_cls = att[-2].to(torch.long)
                labels_reg = att[-1].to(torch.float32)

                loss_cls = self.criterion(logits, labels_cls)
                loss_reg = self.criterion_reg(remain_pred.squeeze(), labels_reg)
                alpha = 0.05
                batch_loss = loss_cls + alpha * loss_reg

                total_loss += batch_loss.item() * bs
                total_samples += bs
                total_correct += (logits.argmax(dim=1) == labels_cls).sum().item()

        avg_loss = total_loss / max(total_samples, 1)
        avg_acc = total_correct / max(total_samples, 1)
        return avg_loss, avg_acc

    def eval_f1(self, eval_model, list_view_valid, num_view, cat_view, stage=1):  # 增加一个 stage 参数
        eval_model.eval()
        torch_dataset_valid = Data.TensorDataset(*list_view_valid)
        loader_valid = Data.DataLoader(
            dataset=torch_dataset_valid,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
        )
        att_str = cat_view + num_view + ['y', 'y_remain']

        total_loss = 0.0
        total_samples = 0
        total_correct = 0

        all_labels = []
        all_preds = []

        with torch.no_grad():
            for batch_num, data in enumerate(loader_valid):
                att = [data[i].to(self.device) for i in range(len(att_str))]
                bs = att[0].size(0)

                logits, enc_self_attns, remain_pred = eval_model(att_str, att)
                labels_cls = att[-2].to(torch.long)
                loss_cls = self.criterion(logits, labels_cls)

                if stage == 1:
                    labels_reg = att[-1].to(torch.float32)
                    loss_reg = self.criterion_reg(remain_pred.squeeze(), labels_reg)
                    alpha = 0.05
                    batch_loss = loss_cls + alpha * loss_reg
                else:
                    batch_loss = loss_cls

                total_loss += batch_loss.item() * bs
                total_samples += bs
                preds = logits.argmax(dim=1)
                total_correct += (preds == labels_cls).sum().item()

                all_labels.append(labels_cls.cpu())
                all_preds.append(preds.cpu())


        avg_loss = total_loss / max(total_samples, 1)
        avg_acc = total_correct / max(total_samples, 1)

        final_labels = torch.cat(all_labels).numpy()
        final_preds = torch.cat(all_preds).numpy()

        macro_f1 = f1_score(final_labels, final_preds, average='macro', zero_division=0)

        return avg_loss, avg_acc, macro_f1


    def train_val(self):
        with open("data/" + args.eventlog + "/" + args.eventlog + '_num_cols.pickle', 'rb') as pickle_file:
            num_view = pickle.load(pickle_file)
        with open("data/" + args.eventlog + "/" + args.eventlog + '_cat_cols.pickle', 'rb') as pickle_file:
            cat_view = pickle.load(pickle_file)

        best_val_acc = 0
        best_epoch = 0
        best_model = None
        print('Starting model...')

        patience = 10
        wait = 0
        min_epochs = 20

        list_cat_view_train = []

        for col in cat_view:
            list_cat_view_train.append(torch.from_numpy(
                np.load("data/" + args.eventlog + "/" + args.eventlog + "_" + col + "_" + str(self.fold) + "_train.npy").astype(int)))

        list_cat_view_valid = []
        for col in cat_view:
            list_cat_view_valid.append(torch.from_numpy(
                np.load("data/" + args.eventlog + "/" + args.eventlog + "_" + col + "_" + str(self.fold) + "_valid.npy").astype(int)))

        list_cat_view_test = []
        for col in cat_view:
            list_cat_view_test.append(torch.from_numpy(
                np.load("data/" + args.eventlog + "/" + args.eventlog + "_" + col + "_" + str(
                    self.fold) + "_test.npy").astype(int)))

        list_num_view_train = []
        for col in num_view:
            list_num_view_train.append(torch.from_numpy(
                np.load("data/" + args.eventlog + "/" + args.eventlog + "_" + col + "_" + str(self.fold) + "_train.npy",
                        allow_pickle=True)).to(torch.float32))

        list_num_view_valid = []
        for col in num_view:
            list_num_view_valid.append(torch.from_numpy(
                np.load("data/" + args.eventlog + "/" + args.eventlog + "_" + col + "_" + str(
                    self.fold) + "_valid.npy",
                        allow_pickle=True)).to(torch.float32))

        list_num_view_test = []
        for col in num_view:
            list_num_view_test.append(torch.from_numpy(
                np.load("data/" + args.eventlog + "/" + args.eventlog + "_" + col + "_" + str(
                    self.fold) + "_test.npy",
                        allow_pickle=True)).to(torch.float32))

        list_view_train = self.Union(list_cat_view_train, list_num_view_train)
        list_view_valid = self.Union(list_cat_view_valid, list_num_view_valid)
        list_view_test = self.Union(list_cat_view_test, list_num_view_test)

        y_train = np.load("data/" + args.eventlog + "/" + args.eventlog + "_y_" + str(self.fold) + "_train.npy")
        y_valid = np.load("data/" + args.eventlog + "/" + args.eventlog + "_y_" + str(self.fold) + "_valid.npy")
        y_test = np.load("data/" + args.eventlog + "/" + args.eventlog + "_y_" + str(self.fold) + "_test.npy")

        y_train = torch.from_numpy(y_train - 1).to(torch.long)
        y_valid = torch.from_numpy(y_valid - 1).to(torch.long)
        y_test = torch.from_numpy(y_test - 1).to(torch.long)

        remain_y_train = np.load(f"data/{args.eventlog}/{args.eventlog}_y_remain_{self.fold}_train.npy")
        remain_y_valid = np.load(f"data/{args.eventlog}/{args.eventlog}_y_remain_{self.fold}_valid.npy")
        remain_y_test = np.load(f"data/{args.eventlog}/{args.eventlog}_y_remain_{self.fold}_test.npy")

        remain_y_train = torch.from_numpy(remain_y_train).to(torch.float32)
        remain_y_valid = torch.from_numpy(remain_y_valid).to(torch.float32)
        remain_y_test = torch.from_numpy(remain_y_test).to(torch.float32)

        mean = remain_y_train.mean()
        std = remain_y_train.std()

        remain_y_train = (remain_y_train - mean) / std
        remain_y_valid = (remain_y_valid - mean) / std
        remain_y_test = (remain_y_test - mean) / std

        list_view_train.append(y_train)
        list_view_train.append(remain_y_train)

        list_view_valid.append(y_valid)
        list_view_valid.append(remain_y_valid)

        list_view_test.append(y_test)
        list_view_test.append(remain_y_test)

        timeRun = []
        for epoch in range(args.epochs):
            start_time = time.perf_counter()


            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats(self.device)

            self.train(list_view_train, num_view, cat_view, epoch)


            if torch.cuda.is_available():
                peak_memory_bytes = torch.cuda.max_memory_allocated(self.device)
                peak_memory_gb = peak_memory_bytes / (1024 ** 3)
                #print(f"\tPeak GPU Memory Usage: {peak_memory_gb:.2f} GB")



            valid_loss, valid_acc = self.eval(self.model, list_view_valid, num_view, cat_view)
            print('-' * 89)
            print(f'\tEpoch: {epoch:d}\t|\tLoss: {valid_loss:.5f}(valid)\t|\tAcc: {valid_acc * 100:.2f}%(valid)')
            print('-' * 89)

            if valid_acc > best_val_acc:
                best_val_acc = valid_acc
                best_epoch = epoch + 1
                best_model = copy.deepcopy(self.model)
                wait = 0
            else:
                wait += 1
                if epoch + 1 >= min_epochs and wait >= patience:
                    print(f"Early stopping at epoch {epoch + 1}, best epoch was {best_epoch}")
                    break

            if epoch < self.warmup_steps:
                self.scheduler_warmup.step()
            else:
                self.scheduler_cosine.step()

            end_time = time.perf_counter()
            run_time = end_time - start_time
            if epoch < 5:
                timeRun.append(run_time)
            #print("hg run_time:",run_time)


        total_run_time = sum(timeRun)
        num_epochs_recorded = len(timeRun)
        average_time = total_run_time / num_epochs_recorded
        #print(f"average run time a epoch: {average_time:.2f} seconds")

        path = os.path.join("model", args.eventlog)
        if not os.path.exists(path):
            os.makedirs(path)
        model_path = 'model/' + str(args.eventlog) + '/' + args.eventlog + '_' + str(args.n_layers) + '_' + str(args.n_heads) + '_' + str(args.epochs) + '_' + str(
            self.fold) +'_seed' + str(self.seed) +'_model.pkl'

        torch.save(best_model, model_path)
        check_model = torch.load(model_path,weights_only=False)
        valid_loss, valid_acc = self.eval(check_model, list_view_valid, num_view, cat_view)
        print('-' * 89)
        print(f'\tBest_Epoch: {best_epoch:d}\t|\tBest_Loss: {valid_loss:.5f}(valid)\t|\tBest_Acc: {valid_acc * 100:.2f}%(valid)')
        print('-' * 89)
        test_loss, test_acc = self.eval(check_model, list_view_test, num_view, cat_view)
        print('-' * 89)
        print(f'\tBest_Loss: {test_loss:.5f}(test)\t|\tBest_Acc: {test_acc * 100:.2f}%(test)')
        print('-' * 89)





def analyze_wrong_predictions(eval_model, list_view_valid, num_view, cat_view, device, k=5, max_cases=10):
    eval_model.eval()
    torch_dataset_valid = Data.TensorDataset(*list_view_valid)
    loader_valid = Data.DataLoader(
        dataset=torch_dataset_valid,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    att_str = cat_view + num_view + ['y']
    wrong_cases = []

    with torch.no_grad():
        for batch_num, data in enumerate(loader_valid):
            att = []
            for i in range(len(att_str)):
                att.append(data[i].to(device))
            outputs, _ = eval_model(att_str, att)
            probs = F.softmax(outputs, dim=-1)
            preds = probs.argmax(dim=-1)
            labels = att[-1]

            wrong_idx = (preds != labels).nonzero(as_tuple=True)[0]
            for idx in wrong_idx:
                true_label = labels[idx].item()
                pred_label = preds[idx].item()

                num_classes = probs.size(1)
                k_safe = min(k, num_classes)  # 防止越界
                topk_probs, topk_idx = probs[idx].topk(k_safe)

                wrong_cases.append({
                    "true_label": true_label,
                    "pred_label": pred_label,
                    "topk_idx": topk_idx.cpu().tolist(),
                    "topk_probs": topk_probs.cpu().tolist()
                })

    print("=== Wrong Predictions (showing first {} cases) ===".format(max_cases))
    for case in wrong_cases[:max_cases]:
        print(f"True: {case['true_label']} | Pred: {case['pred_label']}")
        for i, (cls, prob) in enumerate(zip(case["topk_idx"], case["topk_probs"])):
            print(f"   Top{i+1}: class={cls}, prob={prob:.4f}")
        print("-" * 40)

    return wrong_cases


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

if __name__ == "__main__":
    seeds = [42,133,12345]

    for seed in seeds:
        set_seed(seed)
        print(f"training seed: {seed}==========================================================================")
        for f in range(3):
            new_Trainer = Trainer(f,seed)
            new_Trainer.train_val()

