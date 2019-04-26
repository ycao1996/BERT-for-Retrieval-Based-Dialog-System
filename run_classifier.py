# coding=utf-8

"""BERT finetuning runner."""

from __future__ import absolute_import, division, print_function

import argparse
import collections
import logging
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler
from tqdm import tqdm, trange

import tokenization
from modeling import BertConfig, BertForSequenceClassification
from optimization import BERTAdam
from processor import Response_Selection_Processor

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i

    features = []
    for (ex_index, example) in enumerate(tqdm(examples)):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)

        if tokens_b:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[0:(max_seq_length - 2)]

        tokens = []
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                segment_ids.append(1)
            tokens.append("[SEP]")
            segment_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        label_id = label_map[example.label]

        features.append(
                InputFeatures(
                        input_ids=input_ids,
                        input_mask=input_mask,
                        segment_ids=segment_ids,
                        label_id=label_id))
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--checkpoints_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model checkpoints will be written.")
    parser.add_argument("--vocab_file",
                        default=None,
                        type=str,
                        required=True,
                        help="The vocabulary file that the BERT model was trained on.")
    parser.add_argument("--bert_config_file",
                        default=None,
                        type=str,
                        required=True,
                        help="The config json file corresponding to the pre-trained BERT model. \n"
                             "This specifies the model architecture.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the results of model will be written.")
    parser.add_argument("--init_checkpoint",
                        default=None,
                        type=str,
                        required=True,
                        help="Initial checkpoint (usually from a pre-trained BERT model).")
    
    # Other parameters
    parser.add_argument("--do_train",
                        default=True,
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_test",
                        default=False,
                        action='store_true',
                        help="Whether to do test.")
    parser.add_argument("--do_eval",
                        default=False,
                        action='store_true',
                        help="Whether to run eval on the test set.")                    
    parser.add_argument("--do_lower_case",
                        default=False,
                        action='store_true',
                        help="Whether to lower case the input text. True for uncased models, False for cased models.")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        default=False,
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--accumulate_gradients",
                        type=int,
                        default=1,
                        help="Number of steps to accumulate gradient on (divide the batch_size and accumulate)")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed', 
                        type=int, 
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumualte before performing a backward/update pass.")
    args = parser.parse_args()

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device %s n_gpu %d distributed training %r", device, n_gpu, bool(args.local_rank != -1))

    if args.accumulate_gradients < 1:
        raise ValueError("Invalid accumulate_gradients parameter: {}, should be >= 1".format(
                            args.accumulate_gradients))

    args.train_batch_size = int(args.train_batch_size / args.accumulate_gradients)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    bert_config = BertConfig.from_json_file(args.bert_config_file)

    if args.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length {} because the BERT model was only trained up to sequence length {}".format(
             args.max_seq_length, bert_config.max_position_embeddings))

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    os.makedirs(args.output_dir, exist_ok=True)

    processor = Response_Selection_Processor()
    label_list = processor.get_labels()  # ['0', '1']

    tokenizer = tokenization.FullTokenizer(
        vocab_file=args.vocab_file, do_lower_case=args.do_lower_case)

    # model and optimizer
    model = BertForSequenceClassification(bert_config, len(label_list))
    if args.init_checkpoint is not None:
        model.bert.load_state_dict(torch.load(args.init_checkpoint, map_location='cpu'))
    model.to(device)

    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    no_decay = ['bias', 'gamma', 'beta']
    optimizer_parameters = [
         {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
          'weight_decay_rate': 0.01},
         {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
          'weight_decay_rate': 0.0}
         ]

    # training set
    train_examples = None
    num_train_steps = None
    if args.do_train:
        train_examples = processor.get_train_examples(args.data_dir)  # return a list of examples
        num_train_steps = int(
            len(train_examples) / args.train_batch_size * args.num_train_epochs)

        train_features = convert_examples_to_features(
            train_examples, label_list, args.max_seq_length, tokenizer)
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_steps)

        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)

        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)

        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

    # test set
    if args.do_eval:
        # test_examples = processor.get_test_examples(args.data_dir)
        test_examples = processor.get_dev_examples(args.data_dir)
        test_features = convert_examples_to_features(
            test_examples, label_list, args.max_seq_length, tokenizer)

        all_input_ids = torch.tensor([f.input_ids for f in test_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in test_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in test_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in test_features], dtype=torch.long)

        test_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        test_dataloader = DataLoader(test_data, batch_size=args.eval_batch_size, shuffle=False)

    # test set
    if args.do_test:
        test_examples2 = processor.get_test_examples(args.data_dir)
        # test_examples = processor.get_dev_examples(args.data_dir)
        test_features2 = convert_examples_to_features(
            test_examples2, label_list, args.max_seq_length, tokenizer)

        all_input_ids = torch.tensor([f.input_ids for f in test_features2], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in test_features2], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in test_features2], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in test_features2], dtype=torch.long)

        test_data2 = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        test_dataloader2 = DataLoader(test_data2, batch_size=args.eval_batch_size, shuffle=False)

    optimizer = BERTAdam(optimizer_parameters,
                         lr=args.learning_rate,
                         warmup=args.warmup_proportion,
                         t_total=num_train_steps)

    # train
    output_log_file = os.path.join(args.output_dir, "log.txt")
    print("output_log_file=", output_log_file)
    with open(output_log_file, "w") as writer:
        if args.do_eval:
            writer.write("epoch\tglobal_step\tloss\ttest_loss\ttest_accuracy\n")
        else:
            writer.write("epoch\tglobal_step\tloss\n")
    
    global_step = 0
    epoch = 0
    if args.do_train:
        for _ in trange(int(args.num_train_epochs), desc="Epoch"):
            epoch += 1
            model.train()
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch
                loss, _ = model(input_ids, segment_ids, input_mask, label_ids)
                if n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                loss.backward()
                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    optimizer.step()    # We have accumulated enought gradients
                    model.zero_grad()
                    global_step += 1
        
            # do_eval
            if args.do_eval:
                model.eval()
                test_loss, test_accuracy = 0, 0
                nb_test_steps, nb_test_examples = 0, 0
                with open(os.path.join(args.output_dir, "test_ep_"+str(epoch)+".txt"), "w") as f_test:
                    for input_ids, input_mask, segment_ids, label_ids in test_dataloader:
                        input_ids = input_ids.to(device)
                        input_mask = input_mask.to(device)
                        segment_ids = segment_ids.to(device)
                        label_ids = label_ids.to(device)

                        with torch.no_grad():
                            tmp_test_loss, logits = model(input_ids, segment_ids, input_mask, label_ids)

                        logits = F.softmax(logits, dim=-1)
                        logits = logits.detach().cpu().numpy()
                        label_ids = label_ids.to('cpu').numpy()
                        outputs = np.argmax(logits, axis=1)
                        for output_i in range(len(outputs)):
                            f_test.write(str(outputs[output_i]))
                            for ou in logits[output_i]:
                                f_test.write(" "+str(ou))
                            f_test.write("\n")
                        tmp_test_accuracy = np.sum(outputs == label_ids)

                        test_loss += tmp_test_loss.mean().item()
                        test_accuracy += tmp_test_accuracy

                        nb_test_examples += input_ids.size(0)
                        nb_test_steps += 1

                test_loss = test_loss / nb_test_steps
                test_accuracy = test_accuracy / nb_test_examples

            result = collections.OrderedDict()
            if args.do_eval:
                result = {'epoch': epoch,
                            'global_step': global_step,
                            'loss': tr_loss/nb_tr_steps,
                            'test_loss': test_loss,
                            'test_accuracy': test_accuracy}
            else:
                result = {'epoch': epoch,
                            'global_step': global_step,
                            'loss': tr_loss/nb_tr_steps}

            logger.info("***** Eval results *****")
            with open(output_log_file, "a+") as writer:
                for key in result.keys():
                    logger.info("  %s = %s\n", key, str(result[key]))
                    writer.write("%s\t" % (str(result[key])))
                writer.write("\n")

            if args.checkpoints_dir is not None:
                # Save a trained model and the associated configuration
                model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                model_name = "pytorch_model_" + str(epoch+1) + ".bin"
                output_model_file = os.path.join(args.checkpoints_dir, model_name)
                torch.save(model_to_save.state_dict(), output_model_file)

            if args.do_test:
                model.eval()
                # test_loss, test_accuracy = 0, 0
                nb_test_steps, nb_test_examples = 0, 0
                with open(os.path.join(args.output_dir, "predict_ep_"+str(epoch)+".txt"), "w") as f_test:
                    for input_ids, input_mask, segment_ids, label_ids in test_dataloader2:
                        input_ids = input_ids.to(device)
                        input_mask = input_mask.to(device)
                        segment_ids = segment_ids.to(device)
                        label_ids = label_ids.to(device)

                        with torch.no_grad():
                            tmp_test_loss, logits = model(input_ids, segment_ids, input_mask, label_ids)

                        logits = F.softmax(logits, dim=-1)
                        logits = logits.detach().cpu().numpy()
                        # label_ids = label_ids.to('cpu').numpy()
                        outputs = np.argmax(logits, axis=1)
                        for output_i in range(len(outputs)):
                            f_test.write(str(outputs[output_i]))
                            for ou in logits[output_i]:
                                f_test.write(" " + str(ou))
                            f_test.write("\n")

                        nb_test_examples += input_ids.size(0)
                        nb_test_steps += 1


if __name__ == "__main__":
    main()
