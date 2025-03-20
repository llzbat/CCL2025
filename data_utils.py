# -*- coding: utf-8 -*-

# This script contains all data transformation and reading

from torch.utils.data import Dataset

spto={"Racism":"种族",
      "Region":"地域",
      "Sexism":"性别",
      "LGBTQ":"LGBTQ",
      "others":"其他",
      "non-hate":"不仇恨"}

def read_line_examples_from_file(data_path):

    sents, labels = [], []
    with open(data_path, 'r', encoding=''
                                       'UTF-8') as fp:
        words, labels = [], []
        for line in fp:
            line = line.strip()
            if line != '':
                words, tuples = line.split('####')
                sents.append(words.split())
                labels.append(eval(tuples))
    return sents, labels


def get_para_asqp_targets(labels):
    """
    Obtain the target sentence under the paraphrase paradigm
    """
    targets = []
    for label in labels:
        all_quad_sentences = []
        for quad in label:
            at, ac, sp, ot = quad
            if sp in spto:
                sp=spto[sp]
            if ot=="hate":
                ot="仇恨"
            else:
                ot="不仇恨"
            one_quad_sentence = f"{at} | {ac} | {sp} | {ot}"
            all_quad_sentences.append(one_quad_sentence)

        target = ' [SEP] '.join(all_quad_sentences)
        targets.append(target)
    return targets


def get_transformed_io(data_path):

    sents, labels = read_line_examples_from_file(data_path)
    # the input is just the raw sentence
    inputs = [s.copy() for s in sents]
    targets = get_para_asqp_targets(labels)
    return inputs, targets


class ABSADataset(Dataset):
    def __init__(self, tokenizer, data_dir, data_type, max_len=128):
        self.data_path = f'data/{data_dir}/{data_type}.txt'
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.data_dir = data_dir

        self.inputs = []
        self.targets = []

        self._build_examples()

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        source_ids = self.inputs[index]["input_ids"].squeeze()
        target_ids = self.targets[index]["input_ids"].squeeze()

        src_mask = self.inputs[index]["attention_mask"].squeeze()  # might need to squeeze
        target_mask = self.targets[index]["attention_mask"].squeeze()  # might need to squeeze

        return {"source_ids": source_ids, "source_mask": src_mask, 
                "target_ids": target_ids, "target_mask": target_mask}

    def _build_examples(self):

        inputs, targets = get_transformed_io(self.data_path)

        for i in range(len(inputs)):
            # change input and target to two strings
            input = ' '.join(inputs[i])
            target = targets[i]

            tokenized_input = self.tokenizer.batch_encode_plus(
              [input], max_length=self.max_len, padding="max_length",
              truncation=True, return_tensors="pt"
            )
            tokenized_target = self.tokenizer.batch_encode_plus(
              [target], max_length=self.max_len, padding="max_length",
              truncation=True, return_tensors="pt"
            )

            self.inputs.append(tokenized_input)
            self.targets.append(tokenized_target)
