import torch
import torch.nn as nn
from transformers import AutoConfig,AutoModelWithLMHead,AutoModel
import TRF

class ModelConfig():
    def __init__(self, config):
        myconfig = AutoConfig.for_model('bert').from_json_file(config.pretrainModel_json)
        model_mlm = AutoModelWithLMHead.from_config(myconfig)
        model_mlm = model_mlm.to(config.device)
        model_mlm.load_state_dict(torch.load(config.pretrainModel_path))
        model_mlm_dict = model_mlm.state_dict()  # get pre-trained parameters

        self.model_bert = AutoModel.from_config(myconfig)
        model_bert_dict = self.model_bert.state_dict()

        model_mlm_dict = {k: v for k, v in model_mlm_dict.items() if k in model_bert_dict}
        model_bert_dict.update(model_mlm_dict)
        self.model_bert.load_state_dict(model_bert_dict)

class PEAN(nn.Module):
    def __init__(self, config):
        super(PEAN, self).__init__()
        self.config = config
        self.mode = config.mode
        self.emb_size = config.embedding_size

        if config.feature == "raw":
            if config.embway == "random":
                self.emb = nn.Embedding(config.n_vocab, self.emb_size, padding_idx=0)
            elif config.embway == "pretrain":
                bertConfig = ModelConfig(config)
                self.emb = bertConfig.model_bert
                for param in self.emb.parameters():
                    param.requires_grad = True
                self.emb_size = config.bert_dim

            if config.method == "lstm":
                self.emblstm = nn.LSTM(self.emb_size, config.emblstmhidden_size, config.num_layers,
                                       bidirectional=True, batch_first=True, dropout=config.dropout)
                self.fc01 = nn.Linear(config.emblstmhidden_size * 2, config.num_classes)
            elif config.method == "trf":
                self.TRF = TRF.Model(config=self.config)
                self.fc01 = nn.Linear(self.emb_size * config.pad_num, config.num_classes)

        elif config.feature == "length":
            self.length_embedding = nn.Embedding(2000, config.length_emb_size, padding_idx=0)
            self.lenlstm = nn.LSTM(config.length_emb_size, config.lenlstmhidden_size, config.num_layers,
                                   bidirectional=True, batch_first=True, dropout=config.dropout)
            self.fc02 = nn.Linear(config.lenlstmhidden_size * config.num_layers, config.num_classes)

        elif config.feature == "ensemble":
            self.length_embedding = nn.Embedding(2000, config.length_emb_size, padding_idx=0)
            self.lenlstm = nn.LSTM(config.length_emb_size, config.lenlstmhidden_size, config.num_layers,
                                   bidirectional=True, batch_first=True, dropout=config.dropout)
            if config.embway == "random":
                self.emb = nn.Embedding(config.n_vocab, self.emb_size, padding_idx=0)
            elif config.embway == "pretrain":
                bertConfig = ModelConfig(config)
                self.emb = bertConfig.model_bert
                for param in self.emb.parameters():
                    param.requires_grad = True
                self.emb_size = config.bert_dim
            if config.method == "lstm":
                self.emblstm = nn.LSTM(self.emb_size, config.emblstmhidden_size, config.num_layers,
                                       bidirectional=True, batch_first=True, dropout=config.dropout)
                self.fc = nn.Linear((config.emblstmhidden_size + config.lenlstmhidden_size)*2, config.num_classes)
                self.fc01 = nn.Linear(config.emblstmhidden_size * 2, config.num_classes)

            elif config.method == "trf":
                self.TRF = TRF.Model(config=config)
                self.fc = nn.Linear((self.emb_size  * config.pad_num + config.lenlstmhidden_size*2), config.num_classes)
                self.fc01 = nn.Linear(self.emb_size  * config.pad_num, config.num_classes)

            self.fc02 = nn.Linear(config.lenlstmhidden_size * 2, config.num_classes)

    def forward(self, x):
        config = self.config
        traffic_bytes_idss = x[0]
        length_seq = x[1]

        if config.feature == "raw" or config.feature == "ensemble":
            hidden_feature = torch.Tensor(config.pad_num, len(traffic_bytes_idss), self.emb_size).to(config.device)
            if config.embway == "pretrain":
                for i in range(config.pad_num):
                    _, pooled = self.emb((traffic_bytes_idss[:, i, :]))
                    hidden_feature[i, :, :] = pooled
            elif config.embway == "random":
                for i in range(config.pad_num):
                    packet_emb = self.emb((traffic_bytes_idss[:, i, :]))  # [batch_size,pad_length,emb_size]
                    packet_feature = torch.mean(packet_emb, dim=1)  #[batch_size,emb_size]  packet-level embedding
                    hidden_feature[i, :, :] = packet_feature

            hidden_feature = hidden_feature.permute(1, 0, 2)
            if config.method =="lstm":
                output, (final_hidden_state, final_cell_state) = self.emblstm(hidden_feature)
                out1 = output[:, -1, :]
            elif config.method =="trf":
                out1 = self.TRF(hidden_feature)

        if config.feature == "length" or config.feature == "ensemble":
            input = self.length_embedding(length_seq).reshape(-1, config.pad_len_seq, config.length_emb_size)
            output, (final_hidden_state, final_cell_state) = self.lenlstm(input)
            out2 = output[:, -1, :]


        if config.feature == "raw":
            final_output = self.fc01(out1)
            return (final_output, None, None)
        elif config.feature == "length":
            final_output = self.fc02(out2)
            return final_output, None, None
        else:
            out1_classification = self.fc01(out1)
            out2_classification = self.fc02(out2)
            middle_layer = torch.cat((out1, out2), dim=1)
            final_output = self.fc(middle_layer)
            return (final_output, out1_classification, out2_classification)
