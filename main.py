# -*- coding: utf-8 -*-

import argparse
import logging
import os

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import T5ForConditionalGeneration, T5Tokenizer, get_linear_schedule_with_warmup

from data_utils import ABSADataset
from data_utils import read_line_examples_from_file

logger = logging.getLogger(__name__)


def init_args():
    parser = argparse.ArgumentParser()
    # basic settings
    parser.add_argument("--task", default='asqp', type=str, required=True,
                        help="The name of the task, selected from: [asqp, tasd, aste]")
    parser.add_argument("--dataset", default='rest15', type=str, required=True,
                        help="The name of the dataset, selected from: [rest15, rest16]")
    parser.add_argument("--model_name_or_path", default='t5-base', type=str,
                        help="Path to pre-trained model or shortcut name")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev/test set.")
    parser.add_argument("--do_direct_eval", action='store_true', 
                        help="Whether to run eval on the dev/test set.")
    parser.add_argument("--do_inference", action='store_true', 
                        help="Whether to run inference with trained checkpoints")

    # other parameters
    parser.add_argument("--max_seq_length", default=128, type=int)
    parser.add_argument("--n_gpu", default=0)
    parser.add_argument("--train_batch_size", default=16, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=16, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=3e-4, type=float)
    parser.add_argument("--num_train_epochs", default=30, type=int, 
                        help="Total number of training epochs to perform.")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    # training details
    parser.add_argument("--weight_decay", default=0.0, type=float)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float)
    parser.add_argument("--warmup_steps", default=0.0, type=float)

    args = parser.parse_args()

    # set up output dir which looks like './outputs/rest15/'
    output_dir = f"outputs/{args.dataset}"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    args.output_dir = output_dir

    return args


def get_dataset(tokenizer, type_path, args):
    return ABSADataset(tokenizer=tokenizer, data_dir=args.dataset, 
                       data_type=type_path, max_len=args.max_seq_length)


class T5FineTuner(nn.Module):
    def __init__(self, hparams, model, tokenizer):
        super(T5FineTuner, self).__init__()
        self.hparams = hparams
        self.model = model
        self.tokenizer = tokenizer

    def forward(self, input_ids, attention_mask=None, decoder_input_ids=None,
                decoder_attention_mask=None, labels=None):
        return self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
        )


def train_model(model, train_loader, optimizer, scheduler, device):
    model.train()
    total_loss = 0.0

    for batch in tqdm(train_loader, desc="Training"):
        optimizer.zero_grad()

        lm_labels = batch["target_ids"].to(device)
        lm_labels[lm_labels[:, :] == model.tokenizer.pad_token_id] = -100

        outputs = model(
            input_ids=batch["source_ids"].to(device),
            attention_mask=batch["source_mask"].to(device),
            labels=lm_labels,
            decoder_attention_mask=batch['target_mask'].to(device)
        )

        loss = outputs[0]
        loss.backward()

        total_loss += loss.item()
        optimizer.step()
        scheduler.step()

    return total_loss / len(train_loader)

def validate_model(model, val_loader, device):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            lm_labels = batch["target_ids"].to(device)
            lm_labels[lm_labels[:, :] == model.tokenizer.pad_token_id] = -100

            outputs = model(
                input_ids=batch["source_ids"].to(device),
                attention_mask=batch["source_mask"].to(device),
                labels=lm_labels,
                decoder_attention_mask=batch['target_mask'].to(device)
            )

            loss = outputs[0]
            total_loss += loss.item()

    return total_loss / len(val_loader)

def evaluate(data_loader, model, device):
    model.eval()
    outputs, targets = [], []
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            outs = model.model.generate(input_ids=batch['source_ids'].to(device), 
                                        attention_mask=batch['source_mask'].to(device), 
                                        max_length=128)

            dec = [model.tokenizer.decode(ids, skip_special_tokens=True) for ids in outs]
            outputs.extend(dec)

    save_results(outputs,  "evaluation_results.txt")
    return scores

def save_results(outputs,  file_path):
    with open(file_path, "w", encoding="utf-8") as f:
        for output in  outputs:
            f.write(f"{output} [END]\n")
    print(f"推理结果已保存到 {file_path}")

class LoggingCallback:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def on_validation_end(self, val_loss, epoch):
        self.logger.info(f"***** Validation results after epoch {epoch} *****")
        self.logger.info(f"Validation Loss: {val_loss:.4f}")

    def on_test_end(self, metrics):
        self.logger.info("***** Test results *****")
        
        output_test_results_file = os.path.join(self.output_dir, "test_results.txt")
        with open(output_test_results_file, "w") as writer:
            for key in sorted(metrics):
                if key not in ["log", "progress_bar"]:
                    self.logger.info(f"{key} = {metrics[key]}")
                    writer.write(f"{key} = {metrics[key]}\n")





# initialization
args = init_args()
print("\n", "="*30, f"NEW EXP: ASQP on {args.dataset}", "="*30, "\n")

# sanity check
# show one sample to check the code and the expected output
tokenizer = T5Tokenizer.from_pretrained(args.model_name_or_path)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Here is an example (from the train set):")
dev_dataset = ABSADataset(tokenizer=tokenizer, data_dir=args.dataset,
                      data_type='train', max_len=args.max_seq_length)
data_sample = dev_dataset[0]  # a random data sample
print('Input :', tokenizer.decode(data_sample['source_ids'], skip_special_tokens=True))
print('Output:', tokenizer.decode(data_sample['target_ids'], skip_special_tokens=True))
train_dataset = ABSADataset(tokenizer=tokenizer, data_dir=args.dataset,
                      data_type='train', max_len=args.max_seq_length)


# Initialization of the LoggingCallback
logging_callback = LoggingCallback(args.output_dir)

# training process
if args.do_train:
    print("\n****** Conduct Training ******")

    # initialize the T5 model
    model = T5FineTuner(args, T5ForConditionalGeneration.from_pretrained(args.model_name_or_path), tokenizer)
    model = model.to(device)

    train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size,
                                  drop_last=True, shuffle=True, num_workers=4)

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon)
    t_total = (len(train_loader.dataset) // (args.train_batch_size * max(1, int(args.n_gpu)))) // args.gradient_accumulation_steps * float(args.num_train_epochs)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)

    for epoch in range(int(args.num_train_epochs)):
        train_loss = train_model(model, train_loader, optimizer, scheduler, device)
        print(f"Epoch {epoch+1}/{args.num_train_epochs}, Training Loss: {train_loss:.4f}")

    # save the final model
    model.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print("Finish training and saving the model!")


if args.do_inference:
    print("\n****** Conduct inference on trained checkpoint ******")

    # initialize the T5 model from previous checkpoint
    print(f"Load trained model from {args.output_dir}")
    print('Note that a pretrained model is required and `do_true` should be False')
    tokenizer = T5Tokenizer.from_pretrained(args.output_dir)
    tfm_model = T5ForConditionalGeneration.from_pretrained(args.output_dir)

    model = T5FineTuner(args, tfm_model, tokenizer).to(device)

    sents, _ = read_line_examples_from_file(f'data/{args.dataset}/test.txt')

    test_dataset = ABSADataset(tokenizer, data_dir=args.dataset,
                               data_type='test', max_len=args.max_seq_length)
    test_loader = DataLoader(test_dataset, batch_size=32, num_workers=4)
    scores = evaluate(test_loader, model, device)



