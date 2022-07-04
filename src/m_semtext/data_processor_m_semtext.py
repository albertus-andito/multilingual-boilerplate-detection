import numpy as np
import pandas as pd

from transformers import AutoTokenizer


class MSemTextDataProcessor:

    def __init__(self, tokenizer_name="xlm-roberta-base", use_fast_tokenizer=True,
                 class_sequence_col_name="html/classSequence", tag_sequence_col_name="html/tagSequence",
                 text_sequence_col_name="html/textSequence",
                 html_id_col_name="html/originalId", part_col_name="html/part", label_col_name="html/label",
                 class_sequence_ids_col_name="html/classSequenceIds", tag_sequence_ids_col_name="html/tagSequenceIds",
                 text_sequence_ids_col_name="html/textSequenceIds", mask_sequence_col_name="html/masks", features_col_name="html/featureIds",
                 pad_html_to_max_blocks=True, max_blocks_per_html=85, pad_blocks="max_length", truncate_blocks=True):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=use_fast_tokenizer)

        self.class_sequence_col_name = class_sequence_col_name
        self.tag_sequence_col_name = tag_sequence_col_name
        self.text_sequence_col_name = text_sequence_col_name
        self.html_id_col_name = html_id_col_name
        self.part_col_name = part_col_name
        self.label_col_name = label_col_name

        self.class_sequence_ids_col_name = class_sequence_ids_col_name
        self.tag_sequence_ids_col_name = tag_sequence_ids_col_name
        self.text_sequence_ids_col_name = text_sequence_ids_col_name
        self.mask_sequence_col_name = mask_sequence_col_name
        self.features_col_name = features_col_name

        self.pad_html_to_max_blocks = pad_html_to_max_blocks
        self.max_blocks_per_html = max_blocks_per_html
        self.pad_blocks = pad_blocks
        self.truncate_blocks = truncate_blocks

    def process_text_block(self, class_sequence: list, tag_sequence: list, text_sequence: list):
        class_sequence = str(class_sequence)[1:-1]
        tag_sequence = str(tag_sequence)[1:-1]
        text_sequence = str(text_sequence)[1:-1]

        class_sequence_ids = self.tokenizer(class_sequence, padding=self.pad_blocks, truncation=self.truncate_blocks)["input_ids"]
        tag_sequence_ids = self.tokenizer(tag_sequence, padding=self.pad_blocks, truncation=self.truncate_blocks)["input_ids"]
        text_sequence_ids = self.tokenizer(text_sequence, padding=self.pad_blocks, truncation=self.truncate_blocks)["input_ids"]

        return [class_sequence_ids, tag_sequence_ids, text_sequence_ids]

    def process_html(self, text_blocks: list=None, class_sequences: list=None, tag_sequences: list=None, text_sequences: list=None):
        class_sequence = [str(class_sequence)[1:-1] for class_sequence in class_sequences]
        tag_sequence = [str(tag_sequence)[1:-1] for tag_sequence in tag_sequences]
        text_sequence = [str(text_sequence)[1:-1] for text_sequence in text_sequences]

        class_sequence_ids = self.tokenizer(class_sequence, padding=self.pad_blocks, truncation=self.truncate_blocks)["input_ids"]
        tag_sequence_ids = self.tokenizer(tag_sequence, padding=self.pad_blocks, truncation=self.truncate_blocks)["input_ids"]
        text_sequence_ids = self.tokenizer(text_sequence, padding=self.pad_blocks, truncation=self.truncate_blocks)["input_ids"]

        return [class_sequence_ids, tag_sequence_ids, text_sequence_ids]



    def process_dataset(self, dataset_df=None, dataset_file_path="",
                        class_sequence_col_name=None, tag_sequence_col_name=None, text_sequence_col_name=None,
                        html_id_col_name=None, part_col_name=None):
        """

        :param dataset_df: dataframe of the dataset that is the output of M52's SemText HTML Preprocessor
        :param dataset_file_path: CSV file that is the output of M52's SemText HTML Preprocessor
        :param class_sequence_col_name: Column name of class sequence in the dataframe/CSV file
        :param tag_sequence_col_name: Column name of tag sequence in the dataframe/CSV file
        :param text_sequence_col_name: Column name of text sequence in the dataframe/CSV file
        :param html_id_col_name: Column name of HTML original ID in the dataframe/CSV file
        :return:
        """
        if dataset_df is None:
            dataset_df = pd.read_csv(dataset_file_path)
        if class_sequence_col_name is None:
            class_sequence_col_name = self.class_sequence_col_name
        if tag_sequence_col_name is None:
            tag_sequence_col_name = self.tag_sequence_col_name
        if text_sequence_col_name is None:
            text_sequence_col_name = self.text_sequence_col_name
        if html_id_col_name is None:
            html_id_col_name = self.html_id_col_name
        if part_col_name is None:
            part_col_name = self.part_col_name

        df = dataset_df.copy()

        # remove list structure, use value as a single string
        class_sequence = df[class_sequence_col_name].apply(lambda s: s[1:-1]).tolist()
        tag_sequence = df[tag_sequence_col_name].apply(lambda s: s[1:-1]).tolist()
        text_sequence = df[text_sequence_col_name].apply(lambda s: s[1:-1]).tolist()

        # tokenize and get the input IDs for each token
        df[self.class_sequence_ids_col_name] = self.tokenizer(class_sequence, padding=self.pad_blocks, truncation=self.truncate_blocks)["input_ids"]
        df[self.tag_sequence_ids_col_name] = self.tokenizer(tag_sequence, padding=self.pad_blocks, truncation=self.truncate_blocks)["input_ids"]
        df[self.text_sequence_ids_col_name] = self.tokenizer(text_sequence, padding=self.pad_blocks, truncation=self.truncate_blocks)["input_ids"]

        # group by the html ID name and the part, squeezing the sequences in other columns to a list
        df = df.groupby([html_id_col_name, part_col_name]).aggregate(lambda x: list(x)).reset_index([html_id_col_name, part_col_name])

        # add padding blocks so that all HTML sequences are of the same size
        if self.pad_html_to_max_blocks:
            pad_block = self.tokenizer.model_max_length * [self.tokenizer.pad_token_id]
            df[self.class_sequence_ids_col_name] = df[self.class_sequence_ids_col_name].apply(lambda seq: seq + (self.max_blocks_per_html - len(seq)) * [pad_block])
            df[self.tag_sequence_ids_col_name] = df[self.tag_sequence_ids_col_name].apply(lambda seq: seq + (self.max_blocks_per_html - len(seq)) * [pad_block])
            df[self.text_sequence_ids_col_name] = df[self.text_sequence_ids_col_name].apply(lambda seq: seq + (self.max_blocks_per_html - len(seq)) * [pad_block])
            df[self.label_col_name] = df[self.label_col_name].apply(lambda labels: [*labels, *[0] * (self.max_blocks_per_html - len(labels))])

        # create mask column, to accommodate for text block paddings (1 means the text block is not padding, 0 means text block is padding)
        df[self.mask_sequence_col_name] = df[self.text_sequence_ids_col_name].apply(lambda seq: [0 if all(i == self.tokenizer.pad_token_id for i in ids) else 1 for ids in seq])

        # combine class, tag, and text IDs into one feature column
        df[self.features_col_name] = df[[self.class_sequence_ids_col_name, self.tag_sequence_ids_col_name, self.text_sequence_ids_col_name]].values.tolist()
        # df[self.features_col_name] = df[self.features_col_name].apply(lambda features: np.reshape(features, (self.max_blocks_per_html, 3, self.tokenizer.model_max_length)).tolist())
        return df

