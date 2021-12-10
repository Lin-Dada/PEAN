# coding: UTF-8
import time
import torch
import numpy as np
from train_eval import train
import argparse
from config import Config
from utils import build_dataset, build_iterator
import PEAN_model

parser = argparse.ArgumentParser(description='Traffic Classification')
parser.add_argument('--pad_num', type=int, default=10, help='the padding size of packet num')
parser.add_argument('--pad_len', default=400, type=int, help='the padding size(length) of each packet')
parser.add_argument('--pad_len_seq', default=10, type=int, help='the padding size of packet length sequence')
parser.add_argument('--emb', default=128, type=int, help='the emb size of bytes')
parser.add_argument('--device', default='cuda:0', type=str, help='the training device')
parser.add_argument('--load', default=False, type=bool, help='whether train on previous model')
parser.add_argument('--batch', default=64, type=int, help='batch_size')
parser.add_argument('--feature', default='ensemble', type=str, help='length / raw / ensemble')
parser.add_argument('--method', default='trf', type=str, help='lstm / trf (Sequential Layer)')
parser.add_argument('--embway', default='random', type=str, help='random / pretrain (for raw)')
parser.add_argument('--imploss', default=False, type=bool, help='whether to use improved loss')
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('--length_emb_size', default=32, type=int, help='len emb size')
parser.add_argument('--lenhidden', default=128, type=int, help='len hidden size')
parser.add_argument('--embhidden', default=1024, type=int, help='emb hidden size')
parser.add_argument('--seed', default=1, type=int, help='random seed')
parser.add_argument('--trf_heads', default=8, type=int, help='transformers heads number')
parser.add_argument('--trf_layers', default=2, type=int, help='transformers layers')
parser.add_argument('--mode', default='train', type=str, help='train/test')
parser.add_argument('--k', default='10', type=int, help='k fold validation')
parser.add_argument('--epoch', default='300', type=int, help='epoch')
args = parser.parse_args()

def get_k_fold_data(k, i, X):
    assert k > 1
    fold_size = len(X) // k

    X_train = None
    for j in range(k):
        X_part = X[j * fold_size: (j + 1) * fold_size]
        if j == i:
            X_valid = X_part
        elif X_train is None:
            X_train = X_part
        else:
            X_train = X_train + X_part
    return X_train, X_valid

def get_model(config):
    return PEAN_model.PEAN(config).to(config.device)

def get_config():
    config = Config()
    config.pad_num = args.pad_num
    config.pad_length = args.pad_len
    config.pad_len_seq = args.pad_len_seq
    config.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')  # 设备
    config.mode = args.mode
    config.embedding_size = args.emb
    config.batch_size = args.batch
    config.load = args.load
    config.lenlstmhidden_size = args.lenhidden
    config.emblstmhidden_size = args.embhidden
    config.feature = args.feature
    config.method = args.method
    config.embway = args.embway
    config.length_emb_size = args.length_emb_size
    config.imploss = args.imploss
    config.learning_rate = args.lr
    config.seed = args.seed
    config.trf_heads = args.trf_heads
    config.trf_layers = args.trf_layers
    config.k = args.k
    config.num_epochs = args.epoch
    if args.mode == "test":
        config.load = True
        config.num_epochs = 0

    name = "{}_{}".format(config.feature, config.seed)
    if config.feature != "length":
        name += "_{}_{}_{}_{}_{}_{}".format(config.embway, config.method, config.embedding_size, config.pad_num,
                                            config.pad_length, config.emblstmhidden_size)
    if config.feature == "length" or config.feature == "ensemble":
        name += "_{}_{}".format(config.pad_len_seq, config.lenlstmhidden_size)
    if config.method == "trf":
        if config.trf_heads == 8 and config.trf_layers == 2:
            pass
        else:
            name += "_{}_{}".format(config.trf_heads, config.trf_layers)
    if config.imploss:
        name += "_imploss"
    config.print_path = config.record_path + name + ".txt"  # record console log
    config.loss_path = config.loss_path + name + ".txt"     # record loss
    config.save_path = config.save_path + name + ".ckpt"    # record saved model
    print("\nModel save at: ", config.save_path)
    from transformers import BertTokenizer
    config.tokenizer = BertTokenizer(vocab_file=config.vocab_path, max_seq_length=config.pad_num - 2, max_len=config.pad_num)

    return config

def prepare_data():
    print("----------------------------\n")

    with open(config.print_path, 'a') as f:
        f.write("----------------------------\n\n")

    msg = "Iput Feature: {}\nRandom Seed: {}\n".format(config.feature, config.seed)
    if args.feature == "raw" or args.feature == "ensemble":
        msg += "Sequential use: {}\n".format(config.method)
        msg += "Embedding way: {}(hidden:{})\n".format(config.embway, config.emblstmhidden_size)
        if config.method == "pretrain":
            msg += "Bert Size: {}\n".format(config.bert_dim)
        else:
            msg += "Embedding Size: {}\n".format(config.embedding_size)
        msg += "Pad_num: {}\n".format(config.pad_num)
        msg += "Pad_len: {}\n".format(config.pad_length)

    if config.feature == "length" or args.feature == "ensemble":
        msg += "Length use: lstm(emb: {}, hidden:{})\n".format(config.length_emb_size, config.lenlstmhidden_size)
        msg += "Pad_len_seq: {}\n".format(config.pad_len_seq)

    if config.method == "trf":
        msg += "trf heads:{}\n".format(config.trf_heads)
        msg += "trf_layers: {}\n".format(config.trf_layers)

    msg += "Use Improved loss: {}\n".format(config.imploss)
    msg += "Learning Rate: {}\n".format(config.learning_rate)
    msg += "Batch Size:{}\n".format(config.batch_size)

    print(msg)
    with open(config.print_path, 'a') as f:
        f.write(msg)
    print("----------------------------\n")
    with open(config.print_path, 'a') as f:
        f.write("----------------------------\n\n")

    print("Loading data...")
    with open(config.print_path, 'a') as f:
        f.write("Loading data...\n")


    train_data = build_dataset(config)

    print("train_set: {}".format(len(train_data)))
    with open(config.print_path, 'a') as f:
        f.write("train_set: {}\n".format(len(train_data)))
    return train_data

if __name__ == '__main__':
    config = get_config()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

    train_data = prepare_data()

    valid_loss, valid_acc, valid_fpr, valid_tpr, valid_ftf, valid_f1 = 0, 0, 0, 0, 0, 0
    model = get_model(config)
    print(model.parameters, "\n")
    print("Total number of paramerters in networks is {}  ".format(sum(x.numel() for x in model.parameters())))

    for i in range(config.k):
        train_, test_ = get_k_fold_data(config.k, i, train_data)
        model = get_model(config)

        train_iter = build_iterator(train_, config)
        test_iter = build_iterator(test_, config)
        dev_iter = build_iterator(test_, config)
        acc_, loss_, f1_, fpr_, tpr_, ftf_ = train(config, model, train_iter, dev_iter, test_iter)

        print('*' * 25, 'result of', i + 1, 'fold', '*' * 25)
        print('loss:%.6f' % loss_, 'acc:%.4f' % acc_, 'FPR:%.4f' % fpr_, 'TPR:%.4f' % tpr_, 'FTF:%.4f' % ftf_, 'F1-macro:%.4f' % f1_, "\n")
        valid_loss += loss_
        valid_acc += acc_
        valid_fpr += fpr_
        valid_tpr += tpr_
        valid_ftf += ftf_
        valid_f1 += f1_
    print("\n", '#' * 10, 'final result of all k fold', '#' * 10)
    print('acc:%.4f' % (valid_acc/config.k), 'F1-macro:%.4f' % (valid_f1/config.k), \
          'TPR:%.4f' % (valid_tpr/config.k), 'FPR:%.4f' % (valid_fpr/config.k), \
          'FTF:%.4f' % (valid_ftf/config.k), 'loss:%.6f' % (valid_loss/config.k), "\n")

