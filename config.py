import torch
import os
class Config(object):
    def __init__(self,):
        self.model_name = "PEAN"
        pretrain_path = './Model/pretrain/'
        record_path = './Model/record/'
        log_path = './Model/log/'
        loss_path = './Model/loss/'
        save_path = './Model/save/'
        dirs = [pretrain_path, record_path, log_path, loss_path, save_path]
        for dir in dirs:
            if not os.path.exists(dir):
                os.makedirs(dir)

        self.pretrainModel_json = pretrain_path + 'model_128d_8h_2l/config.json'
        self.pretrainModel_path = pretrain_path + 'model_128d_8h_2l/model_128d_8h_2l.pth'
        self.dataset = 'sni_whs'
        self.train_path = './TrafficData/' + '{}_train.txt'.format(self.dataset)
        self.class_list = [x.strip() for x in open('./TrafficData/class.txt').readlines()]
        self.save_path = save_path
        self.record_path = record_path
        self.loss_path = loss_path
        self.log_path = log_path
        self.vocab_path = './Config/vocab.txt'
        self.n_vocab = 261
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dropout = 0.5
        self.require_improvement = 10000
        self.num_classes = len(self.class_list)
        self.bert_dim = 128  # It must be consistent with the settings in pretrain_config.json
        self.num_layers = 2
        self.middle_fc_size = 2048
