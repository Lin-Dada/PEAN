# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""

import argparse
import glob
import logging
import os
import pickle
import random
import re
import shutil
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
import time
import copy

from transformers import (
    MODEL_WITH_LM_HEAD_MAPPING,
    AdamW,
    AutoConfig,
    AutoModelWithLMHead,
    AutoTokenizer,
    BertTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    get_linear_schedule_with_warmup,
    AutoModel,
)

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

logger = logging.getLogger(__name__)

MODEL_CONFIG_CLASSES = list(MODEL_WITH_LM_HEAD_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

class InputFeatures(object):
    """A single set of features of data."""
    def __init__(self, input_ids, attention_mask, token_type_ids, masked_lm_labels):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.masked_lm_labels = masked_lm_labels

class TextDataset(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, args, file_path: str, block_size=512):
        assert os.path.isfile(file_path)
        directory, filename = os.path.split(file_path)
        cached_features_file = os.path.join(
            directory, args.model_type + "_cached_lm_" + str(block_size) + "_" + filename
        )

        block_size = block_size - (tokenizer.max_len - tokenizer.max_len_single_sentence)

        if os.path.exists(cached_features_file) and not args.overwrite_cache:
            logger.info("Loading features from cached file %s", cached_features_file)
            with open(cached_features_file, "rb") as handle:
                self.examples = pickle.load(handle)
        else:
            logger.info("Creating features from dataset file at %s", directory)
            #CLS, SEP = '[CLS]', '[SEP]'
            self.examples = []
            with open(file_path, encoding="utf-8") as f:
                text = f.readlines()
                tokenized_text = []
                for line in text:
                    line = line.strip().split(' ')
                    if len(line) > block_size:
                        line = line[:block_size]
                    if len(line)==1:  # 去掉空字符
                        continue
                    tokenized_line = tokenizer.convert_tokens_to_ids(line)
                    tokenized_text.append(tokenized_line)
                    # print(len(tokenized_line))
            """
            tokenized_text : list, [[id1, id2, ..., ], [id1, id2, ..., ], ..., [id1, id2, ..., ]], 
            self.examples : list, [[cls_id, id1, id2, ..., seq_id], [cls_id, id1, id2, ..., seq_id], ...], 
            """
            for i in range(len(tokenized_text)):  # Truncate in block of block_size
                self.examples.append(tokenizer.build_inputs_with_special_tokens(tokenized_text[i]))

            # Note that we are loosing the last truncated example here for the sake of simplicity (no padding)
            # If your dataset is small, first you should look for a bigger one :-) and second you
            # can change this behavior by adding (model specific) padding.

            logger.info("Saving features into cached file %s", cached_features_file)
            with open(cached_features_file, "wb") as handle:
                pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return torch.tensor(self.examples[item], dtype=torch.long)


class LineByLineTextDataset(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, args, file_path: str, block_size=512):
        assert os.path.isfile(file_path)
        # Here, we do not cache the features, operating under the assumption
        # that we will soon use fast multithreaded tokenizers from the
        # `tokenizers` repo everywhere =)
        logger.info("Creating features from dataset file at %s", file_path)

        with open(file_path, encoding="utf-8") as f:
            lines = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]

        self.examples = tokenizer.batch_encode_plus(lines, add_special_tokens=True, max_length=block_size)["input_ids"]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i], dtype=torch.long)


def load_and_cache_examples(args, tokenizer, evaluate=False):
    file_path = args.eval_data_file if evaluate else args.train_data_file
    if args.line_by_line:
        return LineByLineTextDataset(tokenizer, args, file_path=file_path, block_size=args.block_size)
    else:
        return TextDataset(tokenizer, args, file_path=file_path, block_size=args.block_size)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def _sorted_checkpoints(args, checkpoint_prefix="checkpoint", use_mtime=False) -> List[str]:
    ordering_and_checkpoint_path = []

    glob_checkpoints = glob.glob(os.path.join(args.output_dir, "{}-*".format(checkpoint_prefix)))

    for path in glob_checkpoints:
        if use_mtime:
            ordering_and_checkpoint_path.append((os.path.getmtime(path), path))
        else:
            regex_match = re.match(".*{}-([0-9]+)".format(checkpoint_prefix), path)
            if regex_match and regex_match.groups():
                ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))

    checkpoints_sorted = sorted(ordering_and_checkpoint_path)
    checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
    return checkpoints_sorted


def _rotate_checkpoints(args, checkpoint_prefix="checkpoint", use_mtime=False) -> None:
    if not args.save_total_limit:
        return
    if args.save_total_limit <= 0:
        return

    # Check if we should delete older checkpoint(s)
    checkpoints_sorted = _sorted_checkpoints(args, checkpoint_prefix, use_mtime)
    if len(checkpoints_sorted) <= args.save_total_limit:
        return

    number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - args.save_total_limit)
    checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
    for checkpoint in checkpoints_to_be_deleted:
        logger.info("Deleting older checkpoint [{}] due to args.save_total_limit".format(checkpoint))
        shutil.rmtree(checkpoint)


def mask_tokens(inputs: torch.Tensor, tokenizer: PreTrainedTokenizer, args) -> Tuple[torch.Tensor, torch.Tensor]:
    """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """

    if tokenizer.mask_token is None:
        raise ValueError(
            "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer."
        )
    """
    inputs ：tensor, [[cls_id, id1, id2, ..., sep_id, pad_id,..., pad_id], [cls_id, id1, id2, ..., sep_id, pad_id,..., pad_id], ...], 
    """
    labels = inputs.clone()
    # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
    """
    probability_matrix : tensor, [[0.15, ..., 0.15], [0.15, ..., 0.15], ...]
    """
    probability_matrix = torch.full(labels.shape, args.mlm_probability)
    """
    special_tokens_mask : list, [[1,0,0,...,1,0,0,...], [1,0,0,...,1,0,0,...], ....]
    """
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
    ]
    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)

    if tokenizer._pad_token is not None:
        padding_mask = labels.eq(tokenizer.pad_token_id)
        probability_matrix.masked_fill_(padding_mask, value=0.0)
    """
    masked_indices : tensor, [[False, False, ..., True, ..., ], [False, False, ..., True, ..., ], ...]
    labels ：tensor, [[-100, id1, -100, ..., idn, -100,..., -100], [-100, id1, -100, ..., idn, -100,..., -100], ...]
    """
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # We only compute loss on masked tokens
    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels


def train(args, train_dataset, model: PreTrainedModel, tokenizer: PreTrainedTokenizer) -> Tuple[int, float]:
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()
    """
    train_dataset : list, [[cls_id, id1, id2, ..., sep_id], [cls_id, id1, id2, ..., sep_id], ...]
    train_dataset_pad ： tensor, [[cls_id, id1, id2, ..., sep_id, pad_id,..., pad_id], [cls_id, id1, id2, ..., sep_id, pad_id,..., pad_id], ...]
    """
    train_dataset_pad = pad_sequence(train_dataset, batch_first=True, padding_value=tokenizer.pad_token_id)
    train_sampler = RandomSampler(train_dataset_pad) if args.local_rank == -1 else DistributedSampler(train_dataset_pad)
    train_dataloader = DataLoader(
        train_dataset_pad, sampler=train_sampler, batch_size=args.train_batch_size
    )

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    # scheduler = get_linear_schedule_with_warmup(
    #     optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    # )
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_proportion * t_total,
        num_training_steps=t_total)

    # Check if saved optimizer or scheduler states exist
    if (
            args.model_name_or_path
            and os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt"))
            and os.path.isfile(os.path.join(args.model_name_or_path, "scheduler.pt"))
    ):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model, device_ids=[args.gpu_start, args.gpu_start + 1, args.gpu_start + 2,
                                                         args.gpu_start + 3])

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint
    if args.model_name_or_path and os.path.exists(args.model_name_or_path):
        try:
            # set global_step to gobal_step of last saved checkpoint from model path
            checkpoint_suffix = args.model_name_or_path.split("-")[-1].split("/")[0]
            global_step = int(checkpoint_suffix)
            epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
            steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info("  Continuing training from epoch %d", epochs_trained)
            logger.info("  Continuing training from global step %d", global_step)
            logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
        except ValueError:
            logger.info("  Starting fine-tuning.")

    model_to_resize = model.module if hasattr(model, "module") else model  # Take care of distributed/parallel training
    model_to_resize.resize_token_embeddings(len(tokenizer))

    model.zero_grad()

    train_iterator = range(
        epochs_trained, int(args.num_train_epochs)
    )

    set_seed(args)  # Added here for reproducibility
    best_eval_loss = 9e8
    for e in train_iterator:
        # epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        nb_tr_steps = 0
        tr_loss = 0.0
        time_inter = 0.0
        batch_step = 0
        batch_steps = len(train_dataloader)
        for step, batch in enumerate(train_dataloader):
            start_time = time.time()
            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue
            inputs, labels = mask_tokens(batch, tokenizer, args) if args.mlm else (batch, batch)
            inputs = inputs.to(args.device)
            labels = labels.to(args.device)
            model.train()
            outputs = model(inputs, masked_lm_labels=labels) if args.mlm else model(inputs, labels=labels)


            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            nb_tr_steps += 1

            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                end_time = time.time()
                time_inter = time_inter + end_time - start_time
                if args.each_batch_eval:
                    eval_result = evaluate(args, model, tokenizer)

                    batch_step += 1
                    if step != batch_steps - 1:
                        print(
                            "\r============================ -epoch %d[%d/%d] -train_loss %.4f -eval_loss %.4f -train_batch_spend_time %.4fs" %
                            (e, batch_step, batch_steps, tr_loss / nb_tr_steps, eval_result['eval_loss'], time_inter),
                            end="", flush=True)
                    else:
                        print(
                            "\r============================ -epoch %d[%d/%d] -train_loss %.4f -eval_loss %.4f -train_batch_spend_time %.4fs\n" %
                            (e, batch_step, batch_steps, tr_loss / nb_tr_steps, eval_result['eval_loss'], time_inter),
                            end="", flush=True)
                else:
                    batch_step += 1
                    if step != batch_steps - 1:
                        print(
                            "\r============================ -epoch %d[%d/%d] -train_loss %.4f -train_batch_spend_time %.4fs" %
                            (e, batch_step, batch_steps, tr_loss / nb_tr_steps, time_inter), end="", flush=True)
                    else:
                        print(
                            "\r============================ -epoch %d[%d/%d] -train_loss %.4f -train_batch_spend_time %.4fs\n" %
                            (e, batch_step, batch_steps, tr_loss / nb_tr_steps, time_inter), end="", flush=True)

            if args.max_steps > 0 and global_step > args.max_steps:
                break

        if args.each_epoch_eval:
            eval_result = evaluate(args, model, tokenizer)

            train2eval_loss = tr_loss / nb_tr_steps
            if e == 0:
                with open(os.path.join(args.output_dir, "train_eval_loss.txt"), "w", encoding="utf-8") as f:
                    f.write("============================ -epoch %d -train_loss %.4f -eval_loss %.4f\n" %
                            (e, train2eval_loss, eval_result['eval_loss']))
            else:
                with open(os.path.join(args.output_dir, "train_eval_loss.txt"), "a", encoding="utf-8") as f:
                    f.write("============================ -epoch %d -train_loss %.4f -eval_loss %.4f\n" %
                            (e, train2eval_loss, eval_result['eval_loss']))
            logger.info("============================ -epoch %d -train_loss %.4f -eval_loss %.4f\n" %
                        (e, train2eval_loss, eval_result['eval_loss']))

        if best_eval_loss > eval_result['eval_loss']:
            best_eval_loss = eval_result['eval_loss']
            best_train_loss = train2eval_loss
            best_epoch = e
            best_model = copy.deepcopy(model)
            print("saving model..............")
            output_dir = args.output_dir
            os.makedirs(output_dir, exist_ok=True)
            model_to_save = (
                best_model.module if hasattr(best_model, "module") else best_model
            )  # Take care of distributed/parallel training
            model_to_save.save_pretrained(output_dir)
            torch.save(best_model.state_dict(), os.path.join(output_dir, pretrain_model_name+".pth"))

        if args.max_steps > 0 and global_step > args.max_steps:
            # train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    # output_dir = os.path.join(args.output_dir, "epoch-{}-loss-{}-model".format(best_epoch, round(best_eval_loss, 4)))
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    model_to_save = (
        best_model.module if hasattr(best_model, "module") else best_model
    )  # Take care of distributed/parallel training
    model_to_save.save_pretrained(output_dir)
    torch.save(best_model.state_dict(), os.path.join(output_dir, pretrain_model_name+".pth"))
    # best_tokenizer.save_pretrained(output_dir)
    # torch.save(best_args, os.path.join(output_dir, "training_args.bin"))
    logger.info("Saving epoch-{}-loss-{:.4f}-model to %s".format(best_epoch, best_eval_loss), output_dir)

    return global_step, best_epoch, best_eval_loss, best_train_loss


def evaluate(args, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, prefix="") -> Dict:
    eval_output_dir = args.output_dir
    eval_dataset = load_and_cache_examples(args, tokenizer, evaluate=True)
    if args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir, exist_ok=True)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

    eval_dataset_pad = pad_sequence(eval_dataset, batch_first=True, padding_value=tokenizer.pad_token_id)
    eval_sampler = RandomSampler(eval_dataset_pad) if args.local_rank == -1 else DistributedSampler(eval_dataset_pad)
    eval_dataloader = DataLoader(
        eval_dataset_pad, sampler=eval_sampler, batch_size=args.eval_batch_size
    )
    set_seed(args)

    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    model.eval()
    eval_loss = 0
    nb_eval_steps = 0

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        inputs, labels = mask_tokens(batch, tokenizer, args) if args.mlm else (batch, batch)
        inputs = inputs.to(args.device)
        labels = labels.to(args.device)

        with torch.no_grad():
            outputs = model(inputs, masked_lm_labels=labels) if args.mlm else model(inputs, labels=labels)
            loss = outputs[0]

        if args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu.
        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps

        eval_loss += loss.item()
        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    perplexity = torch.exp(torch.tensor(eval_loss))

    result = {"perplexity": perplexity, "eval_loss": eval_loss}

    return result


def main():
    global pretrain_model_name
    pretrain_model_name = "model_128d_8h_2l"
    TRAIN_FILE = "./TrafficData/pretrain_train.txt"
    EVAL_FILE = "./TrafficData/pretrain_test.txt"
    output_dir = "./Model/pretrain/" + pretrain_model_name
    config_name = "./Config/pretrain_config.json"
    tokenizer_name = "./Config/vocab.txt"

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    train_data_file = TRAIN_FILE
    eval_data_file = EVAL_FILE

    model_type = "bert"
    model_name_or_path = None
    do_train = True
    do_eval = True
    do_test = True
    do_fune_tune = False
    overwrite_output_dir = True
    overwrite_cache = False
    seed_flag = True
    evaluate_during_training = False
    each_epoch_eval = True
    each_checkpoint_eval = False
    each_batch_eval = False
    mlm = True
    line_by_line = False
    max_seq_length = 400
    per_gpu_train_batch_size = 128
    per_gpu_eval_batch_size = 128
    per_gpu_test_batch_size = 128
    learning_rate = 1e-3
    warmup_proportion = 0.1
    num_train_epochs = 100
    logging_steps = -1
    save_steps = -1
    gpu_start = 0
    gpu_num = torch.cuda.device_count()
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--train_data_file", default=train_data_file, type=str, help="The input training data file (a text file)."
    )
    parser.add_argument(
        "--each_epoch_eval", default=each_epoch_eval, type=str,
    )
    parser.add_argument(
        "--each_batch_eval", default=each_batch_eval, type=str,
    )
    parser.add_argument(
        "--each_checkpoint_eval", default=each_checkpoint_eval, type=str,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=output_dir,
        # required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--model_type", type=str, default=model_type, help="The model architecture to be trained or fine-tuned.",
    )
    parser.add_argument(
        "--gpu_start", default=gpu_start, type=int,
    )
    parser.add_argument(
        "--gpu_num", default=gpu_num, type=int,
    )
    # Other parameters
    parser.add_argument(
        "--eval_data_file",
        default=eval_data_file,
        type=str,
        help="An optional input evaluation data file to evaluate the perplexity on (a text file).",
    )
    parser.add_argument(
        "--line_by_line",
        default=line_by_line,
        action="store_true",
        help="Whether distinct lines of text in the dataset are to be handled as distinct sequences.",
    )
    parser.add_argument(
        "--should_continue", action="store_true", help="Whether to continue from latest checkpoint in output_dir"
    )
    parser.add_argument(
        "--model_name_or_path",
        default=model_name_or_path,
        type=str,
        help="The model checkpoint for weights initialization. Leave None if you want to train a model from scratch.",
    )

    parser.add_argument(
        "--mlm", default=mlm, action="store_true",
        help="Train with masked-language modeling loss instead of language modeling."
    )
    parser.add_argument(
        "--mlm_probability", type=float, default=0.15, help="Ratio of tokens to mask for masked language modeling loss"
    )

    parser.add_argument(
        "--config_name",
        default=config_name,
        type=str,
        help="Optional pretrained config name or path if not the same as model_name_or_path. If both are None, initialize a new config.",
    )
    parser.add_argument(
        "--tokenizer_name",
        default=tokenizer_name,
        type=str,
        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path. If both are None, initialize a new tokenizer.",
    )
    parser.add_argument(
        "--cache_dir",
        default=None,
        type=str,
        help="Optional directory to store the pre-trained models downloaded from s3 (instead of the default one)",
    )
    parser.add_argument(
        "--block_size",
        default=max_seq_length,
        type=int,
        help="Optional input sequence length after tokenization."
             "The training dataset will be truncated in block of this size for training."
             "Default to the model max input length for single sentence inputs (take into account special tokens).",
    )
    parser.add_argument("--do_train", default=do_train, action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", default=do_eval, action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument("--do_fune_tune", default=do_fune_tune, action="store_true")
    parser.add_argument(
        "--evaluate_during_training", default=evaluate_during_training, action="store_true",
        help="Run evaluation during training at each logging step."
    )

    parser.add_argument("--per_gpu_train_batch_size", default=per_gpu_train_batch_size, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=per_gpu_eval_batch_size, type=int,
        help="Batch size per GPU/CPU for evaluation."
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--learning_rate", default=learning_rate, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs", default=num_train_epochs, type=float, help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--warmup_proportion", default=warmup_proportion, type=int,
                        help="Linear warmup over warmup_steps.")

    parser.add_argument("--logging_steps", type=int, default=logging_steps, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=save_steps, help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--save_total_limit",
        type=int,
        default=None,
        help="Limit the total amount of checkpoints, delete the older checkpoints in the output_dir, does not delete by default",
    )
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name_or_path ending and ending with step number",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument(
        "--overwrite_output_dir", default=overwrite_output_dir, action="store_true",
        help="Overwrite the content of the output directory"
    )
    parser.add_argument(
        "--overwrite_cache", default=overwrite_cache, action="store_true",
        help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
             "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--server_ip", type=str, default="", help="For distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="For distant debugging.")
    parser.add_argument('--device', default='cuda:0', type=str, help='the training device')
    args = parser.parse_args()

    if args.model_type in ["bert", "roberta", "distilbert", "camembert"] and not args.mlm:
        raise ValueError(
            "BERT and RoBERTa-like models do not have LM heads but masked LM heads. They must be run using the --mlm "
            "flag (masked language modeling)."
        )
    if args.eval_data_file is None and args.do_eval:
        raise ValueError(
            "Cannot do evaluation without an evaluation data file. Either supply a file to --eval_data_file "
            "or remove the --do_eval argument."
        )
    if args.should_continue:
        sorted_checkpoints = _sorted_checkpoints(args)
        if len(sorted_checkpoints) == 0:
            raise ValueError("Used --should_continue but no checkpoint was found in --output_dir.")
        else:
            args.model_name_or_path = sorted_checkpoints[-1]

    if (
            os.path.exists(args.output_dir)
            and os.listdir(args.output_dir)
            and args.do_train
            and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )

    args.n_gpu = 1
    device = args.device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    # Set seed
    if seed_flag:
        set_seed(args)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training download model & vocab

    if args.config_name:
        config = AutoConfig.for_model(args.model_type).from_json_file(args.config_name)
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
    else:
        # When we release a pip version exposing CONFIG_MAPPING,
        # we can do `config = CONFIG_MAPPING[args.model_type]()`.
        raise ValueError(
            "You are instantiating a new config instance from scratch. This is not supported, but you can do it from another script, save it,"
            "and load it from here, using --config_name"
        )

    if args.tokenizer_name:
        Tokenizer = BertTokenizer
        tokenizer = Tokenizer(vocab_file=args.tokenizer_name, max_seq_length=args.block_size - 2,
                              max_len=args.block_size)

    elif args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported, but you can do it from another script, save it,"
            "and load it from here, using --tokenizer_name"
        )

    if args.block_size <= 0:
        args.block_size = tokenizer.max_len
        # Our input block size will be the max possible for the model
    else:
        args.block_size = min(args.block_size, tokenizer.max_len)

    if args.model_name_or_path:
        model = AutoModelWithLMHead.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            cache_dir=args.cache_dir,
        )
    else:
        logger.info("Training new model from scratch")
        if args.do_fune_tune:
            # model = AutoModelWithLMHead.from_pretrained(args.output_dir)
            model = AutoModel.from_pretrained(args.output_dir)
            # model = finetune_cls(model)
        else:
            # model = AutoModelWithLMHead.from_pretrained(os.path.join(args.output_dir, "checkpoint-50000"))
            model = AutoModelWithLMHead.from_config(config)
            # model = BertForMaskedLM(config=BertConfig.from_json_file(args.bert_config_json))
            # model = BertForMaskedLM(config=config)

    model.to(args.device)
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)

    if args.local_rank == 0:
        torch.distributed.barrier()  # End of barrier to make sure only the first process in distributed training download model & vocab

    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        if args.local_rank not in [-1, 0]:
            torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training process the dataset, and the others will use the cache

        train_dataset = load_and_cache_examples(args, tokenizer, evaluate=False)

        if args.local_rank == 0:
            torch.distributed.barrier()

        global_step, best_epoch, best_eval_loss, best_train_loss = train(args, train_dataset, model, tokenizer)
        logger.info(
            "In all, %d epoch were trained, global steps were %d. best epoch is %d, best train loss is %f, best eval loss is %f.",
            args.num_train_epochs, global_step, best_epoch, best_train_loss, best_eval_loss)


    if args.do_eval:
        train_dataset = load_and_cache_examples(args, tokenizer, evaluate=False)
        train_dataset_pad = pad_sequence(train_dataset, batch_first=True, padding_value=tokenizer.pad_token_id)
        train_sampler = RandomSampler(train_dataset_pad) if args.local_rank == -1 else DistributedSampler(
            train_dataset_pad)
        train_dataloader = DataLoader(
            train_dataset_pad, sampler=train_sampler, batch_size=args.train_batch_size
        )
        # model = AutoModelWithLMHead.from_pretrained(args.output_dir)
        # model.to(args.device)
        # 验证集整体测试
        model = AutoModelWithLMHead.from_config(config)
        model.to(args.device)
        # model = torch.nn.DataParallel(model, device_ids=[args.gpu_start, args.gpu_start + 1, args.gpu_start + 2,
        #                                                  args.gpu_start + 3])
        model.load_state_dict(torch.load(os.path.join(args.output_dir, pretrain_model_name+".pth")))

        eval_result = evaluate(args, model, tokenizer)
        train2eval_loss = 0
        nb_train_eval_steps = 0
        for batch in tqdm(train_dataloader, desc="TrainSet Evaluating"):
            inputs, labels = mask_tokens(batch, tokenizer, args) if args.mlm else (batch, batch)
            inputs = inputs.to(args.device)
            labels = labels.to(args.device)

            with torch.no_grad():
                outputs = model(inputs, masked_lm_labels=labels) if args.mlm else model(inputs, labels=labels)
                loss = outputs[0]

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu.
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            train2eval_loss += loss.item()
            nb_train_eval_steps += 1

        train2eval_loss = train2eval_loss / nb_train_eval_steps
        logger.info("============================ Eval directly, -train_loss %.4f -eval_loss %.4f\n" %
                    (train2eval_loss, eval_result['eval_loss']))


if __name__ == "__main__":
    main()
