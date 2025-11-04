import torch.nn as nn
import pickle
from config import args
import numpy as np
from sklearn.metrics import classification_report, precision_recall_fscore_support, mean_squared_error, \
    mean_absolute_error
import torch
import os
from sklearn.preprocessing import LabelBinarizer
import torch.utils.data as Data
from sklearn.metrics import roc_auc_score, average_precision_score
import torch.nn.functional as F
device = torch.device('cuda:{}'.format(args.gpu))


def multiclass_roc_auc_score(y_test, y_pred, average):
    lb = LabelBinarizer()
    lb.fit(y_test)
    y_test = lb.transform(y_test)
    y_pred = lb.transform(y_pred)
    return roc_auc_score(y_test, y_pred, average=average)


def multiclass_pr_auc_score(y_test, y_pred, average):
    lb = LabelBinarizer()
    lb.fit(y_test)
    y_test = lb.transform(y_test)
    y_pred = lb.transform(y_pred)
    return average_precision_score(y_test, y_pred, average=average)


def mc_dropout_predict(model, att_str, att, n_iterations=1):
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()

    preds_cls = []
    preds_reg = []
    with torch.no_grad():
        for _ in range(n_iterations):
            outputs_cls, _, outputs_reg = model(att_str, att)
            prob = F.softmax(outputs_cls, dim=-1)
            preds_cls.append(prob)
            preds_reg.append(outputs_reg.squeeze())

    avg_prob = torch.stack(preds_cls).mean(dim=0)
    final_cls = avg_prob.argmax(dim=-1)

    avg_reg = torch.stack(preds_reg).mean(dim=0)

    return final_cls, avg_prob, avg_reg


def ensemble_predict(models, att_str, att, device, n_mc=1):
    all_probs = []
    all_regs = []
    with torch.no_grad():
        for model in models:
            model.eval()
            pred_cls, avg_prob, avg_reg = mc_dropout_predict(model, att_str, att, n_iterations=n_mc)
            all_probs.append(avg_prob)
            all_regs.append(avg_reg)

    mean_prob = torch.stack(all_probs).mean(dim=0)
    mean_reg = torch.stack(all_regs).mean(dim=0)
    final_cls = mean_prob.argmax(dim=-1)

    return final_cls, mean_prob, mean_reg

class MAAP_Test:
    def __init__(self, f):
        self.device = torch.device('cuda:{}'.format(args.gpu))
        self.fold = f
        self.criterion = nn.CrossEntropyLoss()

        self.criterion_cls = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.criterion_reg = nn.MSELoss()


    @staticmethod
    def Union(lst1, lst2):
        final_list = list(set(lst1) | set(lst2))
        return final_list

    def mapping(self, df_train, df_valid, df_test, col):
        list_word = self.Union(self.Union(df_train[col].unique(), df_valid[col].unique()),df_test[col].unique())
        mapping = dict(zip(set(list_word), range(1, len(list_word) + 1)))
        len_mapping = len(set(list_word))
        return mapping, len_mapping

    def test(self, eval_model, list_view_test, num_view, cat_view,use_mc_dropout=True):

        loader_test = Data.DataLoader(
            Data.TensorDataset(*list_view_test),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
        )

        att_str = cat_view + num_view + ['y', 'remain_time']
        result_path = f"result/{args.eventlog}"
        os.makedirs(result_path, exist_ok=True)
        outfile2 = open(result_path + "/" + args.eventlog + "_" + ".txt", 'a')

        total_loss, total_cls_loss, total_reg_loss = 0.0, 0.0, 0.0
        total_samples, correct = 0, 0
        Y_test_int, preds_a, remain_true, remain_pred = [], [], [], []

        for data in loader_test:
            att = [d.to(self.device) for d in data]
            pred_cls, avg_prob, avg_reg = ensemble_predict(eval_model, att_str, att, self.device)
            outputs_cls = avg_prob
            outputs_reg = avg_reg
            loss_cls = self.criterion_cls(outputs_cls, att[-2])
            loss_reg = self.criterion_reg(outputs_reg.squeeze(), att[-1].float())

            loss = loss_cls + 0.1 * loss_reg

            correct += (pred_cls == att[-2]).sum().item()
            total_samples += att[-2].size(0)

            Y_test_int.append(att[-2].cpu())
            preds_a.append(outputs_cls.cpu())
            remain_true.append(att[-1].cpu())
            remain_pred.append(outputs_reg.squeeze().cpu())

            total_loss += loss.item() * att[-2].size(0)
            total_cls_loss += loss_cls.item() * att[-2].size(0)
            total_reg_loss += loss_reg.item() * att[-2].size(0)

        acc = correct / total_samples
        total_loss /= total_samples
        total_cls_loss /= total_samples
        total_reg_loss /= total_samples

        Y_test_int = torch.cat(Y_test_int, 0).numpy()
        preds_a = torch.cat(preds_a, 0).numpy()
        remain_true = torch.cat(remain_true, 0).numpy()
        remain_pred = torch.cat(remain_pred, 0).numpy()

        preds_cls = np.argmax(preds_a, axis=1)
        precision, recall, fscore, _ = precision_recall_fscore_support(
            Y_test_int, preds_cls, average='macro', zero_division=0
        )
        auc_score_macro = multiclass_roc_auc_score(Y_test_int, preds_cls, average="macro")
        prauc_score_macro = multiclass_pr_auc_score(Y_test_int, preds_cls, average="macro")

        mse = mean_squared_error(remain_true, remain_pred)
        mae = mean_absolute_error(remain_true, remain_pred)

        print(classification_report(Y_test_int, preds_cls, digits=3, zero_division=0))
        outfile2.write(classification_report(Y_test_int, preds_cls, digits=3, zero_division=0))

        print(f'\nAUC: {auc_score_macro:.4f}')
        print(f'\nPRAUC: {prauc_score_macro:.4f}')
        outfile2.write(f'\nAUC: {auc_score_macro:.4f}')
        outfile2.write(f'\nPRAUC: {prauc_score_macro:.4f}')
        outfile2.write(f'\nMSE (remain_time): {mse:.4f}')
        outfile2.write(f'\nMAE (remain_time): {mae:.4f}\n')
        outfile2.flush()
        outfile2.close()

        return total_loss, acc


    def Final_test(self):
        with open("data/" + args.eventlog + "/" + args.eventlog + '_num_cols.pickle', 'rb') as pickle_file:
            num_view = pickle.load(pickle_file)
        with open("data/" + args.eventlog + "/" + args.eventlog + '_cat_cols.pickle', 'rb') as pickle_file:
            cat_view = pickle.load(pickle_file)
        print('Starting model...')
        list_cat_view_test = []
        for col in cat_view:
            list_cat_view_test.append(torch.from_numpy(
                np.load("data/" + args.eventlog + "/" + args.eventlog + "_" + col + "_" + str(self.fold) + "_test.npy").astype(int)))
        list_num_view_test = []
        for col in num_view:
            list_num_view_test.append(torch.from_numpy(
                np.load("data/" + args.eventlog + "/" + args.eventlog + "_" + col + "_" + str(self.fold) + "_test.npy",
                        allow_pickle=True)).to(torch.float32))

        list_view_test = list_cat_view_test + list_num_view_test

        y_test = np.load("data/" + args.eventlog + "/" + args.eventlog + "_y_" + str(self.fold) + "_test.npy")

        y_test = torch.from_numpy(y_test - 1).to(torch.long)

        remain_y_test = np.load(f"data/{args.eventlog}/{args.eventlog}_y_remain_{self.fold}_test.npy")
        remain_y_test = torch.from_numpy(remain_y_test).to(torch.float32)

        list_view_test.append(y_test)
        list_view_test.append(remain_y_test)


        seeds = [42, 133, 12345]


        models = []
        for seed in seeds:
            model_path = (
                f'model/{args.eventlog}/{args.eventlog}_'
                f'{args.n_layers}_{args.n_heads}_{args.epochs}_{self.fold}_seed{seed}_ab_tr_model.pkl'
            )
            model = torch.load(model_path, map_location=device)
            model.to(device)
            models.append(model)

        test_loss, test_acc = self.test(models, list_view_test, num_view, cat_view)
        print('-' * 89)
        print(f'\tLast_Loss: {test_loss:.5f}(test)\t|\tLast_Acc: {test_acc * 100:.2f}%(test)')
        print('-' * 89)



if __name__ == "__main__":
    for f in range(3):
        new_MAAP = MAAP_Test(f)
        new_MAAP.Final_test()