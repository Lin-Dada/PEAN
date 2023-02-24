# coding: UTF-8
import numpy as np
import torch
import torch.nn.functional as F
from sklearn import metrics
import time
from utils import get_time_dif
from tensorboardX import SummaryWriter


def train(config, model, train_iter, dev_iter, test_iter):
    start_time = time.time()
    if config.load:
        print("\n\nloading model from: {}".format(config.save_path))
        with open(config.print_path, 'a') as f:
            f.write("loading model from: {}\n".format(config.save_path))
        model.load_state_dict(torch.load(config.save_path))

    model.train()
    # L2
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=0.001)

    # lr
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.992)

    total_batch = 0  # current epoch
    dev_best_loss = float('inf')
    last_improve = 0
    flag = False  # whether it is long time without improvement
    writer = SummaryWriter(log_dir=config.log_path + '/' + time.strftime('%m-%d_%H.%M', time.localtime()))

    f_loss = open(config.loss_path,"w")
    f_loss.write("train_loss, dev_loss\n")
    for epoch in range(config.num_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
        lr = optimizer.state_dict()['param_groups'][0]['lr']
        if lr>1e-5:
            scheduler.step()
        print("lr now is: ", lr)
        with open(config.print_path, 'a') as f:
            f.write('Epoch [{}/{}]\n'.format(epoch + 1, config.num_epochs))
            f.write("lr now is: {}\n".format(optimizer.state_dict()['param_groups'][0]['lr']))

        for i, (trains, labels) in enumerate(train_iter):
            # trains: (x, seq_len), y
            outputs = model(trains)
            model.zero_grad()
            if config.imploss:
                # improved loss function
                loss = F.cross_entropy(outputs[0], labels) + F.cross_entropy(outputs[1], labels) + F.cross_entropy(outputs[2], labels)
            else:
                loss = F.cross_entropy(outputs[0], labels)
            loss.backward()
            optimizer.step()
            if total_batch % 100 == 0:
                true = labels.data.cpu()
                predic = torch.max(outputs[0].data, 1)[1].cpu()
                train_acc = metrics.accuracy_score(true, predic)
                dev_acc, dev_loss = evaluate(config, model, dev_iter)
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    torch.save(model.state_dict(), config.save_path)
                    improve = '*'
                    last_improve = total_batch
                else:
                    improve = ''
                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%},  Time: {5} {6}'
                print(msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc, time_dif, improve))
                with open(config.print_path, 'a') as f:
                    f.write(msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc, time_dif, improve)+'\n')
                writer.add_scalar("loss/train", loss.item(), total_batch)
                writer.add_scalar("loss/dev", dev_loss, total_batch)
                writer.add_scalar("acc/train", train_acc, total_batch)
                writer.add_scalar("acc/dev", dev_acc, total_batch)

                f_loss.write('{0:>5.2},{1:>5.2}\n'.format(loss.item(),dev_loss))

                model.train()
            total_batch += 1
            if total_batch - last_improve > config.require_improvement:
                # early stop
                print("No optimization for a long time, auto-stopping...")
                with open(config.print_path, 'a') as f:
                    f.write("No optimization for a long time, auto-stopping...\n")
                flag = True
                break
        if flag:
            break
    writer.close()
    f_loss.close()
    return test(config, model, test_iter)


def test(config, model, test_iter):
    model.load_state_dict(torch.load(config.save_path))
    model.eval()
    start_time = time.time()
    test_acc, test_loss, f1, test_confusion = evaluate(config, model, test_iter, test=True)
    ftr, tpr, ftf = OtherMetrics(test_confusion)
    time_dif = get_time_dif(start_time)

    if config.mode == "train":
        with open(config.print_path, 'a') as f:
            f.write('\n')
            f.write("Confusion Matrix...\n")
            print(test_confusion,file=f)
            f.write('\n')
            f.write("Time usage:{}\n".format(time_dif))
            f.write("Time now is :{}\n".format(time.strftime('%m-%d-%H:%M', time.localtime())))
    return test_acc, test_loss, f1, ftr, tpr, ftf

def evaluate(config, model, data_iter, test=False):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    diff = .0
    datasize = 0
    with torch.no_grad():
        with open("./result.csv",'w') as f:
            f.write("Real,Predict,Result\n")
            for texts, labels in data_iter:
                outputs = model(texts)
                loss = F.cross_entropy(outputs[0], labels)
                loss_total += loss
                labels = labels.data.cpu().numpy()
                stime = time.time() #

                predic = torch.max(outputs[0].data, 1)[1].cpu().numpy()
                etime = time.time()
                diff += etime-stime
                datasize += len(data_iter)
                classlist = ["163Mail", "360Safe", "12306", "Alipay", "Apple", "Baidu", "CSDN", "HuaweiCloud", "JD",
                             "MingyuanCloud", "Mozilla", "QQ", "QQMail", "Taobao", "Wechat", "Weibo", "WPS", "YoudaoNote",
                             "Zhihu"]
                pre = predic.tolist()
                lab = labels.tolist()
                for i in range(len(pre)):
                    if pre[i]==lab[i]:
                        result = "Correct"
                    else:
                        result = "Error"
                    f.write(classlist[pre[i]]+","+classlist[lab[i]]+","+result+"\n")

                labels_all = np.append(labels_all, labels)
                predict_all = np.append(predict_all, predic)

    acc = metrics.accuracy_score(labels_all, predict_all)
    if test:
        print("\nTime Difference (s): " + str(diff))
        print("Per Sample Use (s): " + str(diff / datasize) + "\n")
        report = metrics.classification_report(labels_all, predict_all, target_names=config.class_list, digits=4)
        print(report)
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        f1 = metrics.f1_score(labels_all, predict_all, average='macro')
        return acc, loss_total / len(data_iter), f1, confusion
    return acc, loss_total / len(data_iter)

def OtherMetrics(cnf_matrix):
    FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)
    FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
    TP = np.diag(cnf_matrix)
    TN = cnf_matrix.sum() - (FP + FN + TP)

    FP = FP.astype(float)
    FN = FN.astype(float)
    TP = TP.astype(float)
    TN = TN.astype(float)

    TPR = TP / (TP + FN)  # recall
    FPR = FP / (FP + TN)

    FTF = 0
    weight = cnf_matrix.sum(axis=1)
    w_sum = weight.sum(axis=0)

    for i in range(len(weight)):
        FTF += weight[i] * TPR[i] / (1+FPR[i])
    FTF /= w_sum

    return float(str(np.around(np.mean(FPR), decimals=4).tolist())), float(str(np.around(np.mean(TPR), decimals=4).tolist())), \
           float(str(np.around(FTF, decimals=4)))
