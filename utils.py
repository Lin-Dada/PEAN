import os
import torch
import numpy as np
import pickle as pkl
from tqdm import tqdm
import time
from datetime import timedelta
import pickle


UNK, PAD, CLS, SEP = '[UNK]', '[PAD]', '[CLS]','[SEP]'

def build_dataset(config):
    def load_dataset(path, pad_num = 10, pad_length = 400, pad_len_seq = 10):
        cache_dir = './DataCache/'
        cached_dataset_file = cache_dir + '{}_{}_{}_{}.txt'.format(config.dataset, pad_num, pad_length, pad_len_seq)
        if os.path.exists(cached_dataset_file):
            print("Loading features from cached file {}".format(cached_dataset_file))
            with open(cached_dataset_file, "rb") as handle:
                contents = pickle.load(handle)
                return contents
        else:
            print("Creating training dataset....")
            contents = []
            with open(path, 'r') as f:
                for line in tqdm(f):
                    if not line:
                        continue
                    item = line.split('\t')
                    flow = item[0:-2]  # packets
                    if len(flow) < 2:
                        continue
                    if len(flow) > pad_num:
                        flow = flow[0 : pad_num]
                    length_seq = item[-2].strip().split(' ')
                    length_seq = list(map(int, length_seq))

                    label = item[-1]
                    masks = []
                    seq_lens = []
                    traffic_bytes_idss = []
                    for packet in flow:
                        traffic_bytes = config.tokenizer.tokenize(packet)
                        if len(traffic_bytes) <= pad_length - 2:
                            traffic_bytes = [CLS] + traffic_bytes + [SEP]
                        else:
                            traffic_bytes = [CLS] + traffic_bytes
                            traffic_bytes[pad_length - 1] = SEP


                        seq_len = len(traffic_bytes)
                        mask = []
                        traffic_bytes_ids = config.tokenizer.convert_tokens_to_ids(traffic_bytes)

                        if pad_length:
                            if len(traffic_bytes) < pad_length:
                                mask = [1] * len(traffic_bytes_ids) + [0] * (pad_length - len(traffic_bytes))  # [1,1,...,1,0,0]
                                traffic_bytes_ids += ([0] * (pad_length - len(traffic_bytes)))
                            else:
                                mask = [1] * pad_length
                                traffic_bytes_ids = traffic_bytes_ids[:pad_length]
                                seq_len = pad_length
                        traffic_bytes_idss.append(traffic_bytes_ids)
                        seq_lens.append(seq_len)
                        masks.append(mask)


                        if pad_len_seq:
                            if len(length_seq) < pad_len_seq:
                                length_seq += [0] * (pad_len_seq - len(length_seq))
                            else:
                                length_seq = length_seq[:pad_len_seq]


                    if pad_num: 
                        if len(traffic_bytes_idss) < pad_num:
                            len_tmp = len(traffic_bytes_idss)

                            mask = [0] * pad_length
                           
                            traffic_bytes_ids = [1] + [0] * (pad_length-2) + [2]
                            seq_len = 0
                            for i in range(pad_num - len_tmp):
                                masks.append(mask)
                                traffic_bytes_idss.append(traffic_bytes_ids)
                                seq_lens.append(seq_len)
                        else:
                            traffic_bytes_idss = traffic_bytes_idss[:pad_num]
                            masks = masks[:pad_num]
                            seq_lens = seq_lens[:pad_num]

                    contents.append((traffic_bytes_idss, seq_lens, masks, length_seq, int(label))) 

            print("Saving dataset cached file {}".format(cached_dataset_file))
            if not os.path.exists(cache_dir):
                os.makedirs(cache_dir)
            with open(cached_dataset_file, "wb") as handle:
                pickle.dump(contents, handle, protocol=pickle.HIGHEST_PROTOCOL)
            return contents

    train = load_dataset(config.train_path, config.pad_num, config.pad_length, config.pad_len_seq)
    return train


class DatasetIterater(object):
    def __init__(self, batches, batch_size, device, pad_len_seq):
        self.batch_size = batch_size
        self.batches = batches
        self.n_batches = len(batches) // batch_size
        self.residue = False
        if len(batches) % self.n_batches != 0:
            self.residue = True
        self.index = 0
        self.device = device
        self.pad_len_seq = pad_len_seq

    def _to_tensor(self, datas):
        # datas: batch_size * contents
        # contents: traffic_bytes_idss, seq_lens, masks, length_seq, int(label)
        traffic_bytes_idss = torch.LongTensor([_[0] for _ in datas]).to(self.device)
        length_seq = torch.LongTensor([_[3] for _ in datas])
        length_seq = torch.reshape(length_seq, (-1,self.pad_len_seq,1)).to(self.device)
        seq_lens = torch.LongTensor([_[1] for _ in datas]).to(self.device)
        masks = torch.LongTensor([_[2] for _ in datas]).to(self.device)
        label = torch.LongTensor([_[4] for _ in datas]).to(self.device)

        return (traffic_bytes_idss, length_seq, seq_lens, masks), label

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches

def build_iterator(dataset, config):
    iter = DatasetIterater(dataset, config.batch_size, config.device, config.pad_len_seq)
    return iter

def get_time_dif(start_time):
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


