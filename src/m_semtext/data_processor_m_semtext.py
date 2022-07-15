import numpy as np
import pandas as pd

from transformers import AutoTokenizer


class MSemTextDataProcessor:

    pad_blocks_options = ["max_length", "longest", "do_not_pad"]

    def __init__(self, tokenizer_name: str = "xlm-roberta-base", use_fast_tokenizer: bool = True,
                 class_sequence_col_name: str = "html/classSequence", tag_sequence_col_name: str = "html/tagSequence",
                 text_sequence_col_name: str = "html/textSequence",
                 html_id_col_name: str = "html/originalId", part_col_name: str = "html/part",
                 label_col_name: str = "html/label",
                 class_sequence_ids_col_name: str = "html/classSequenceIds",
                 tag_sequence_ids_col_name: str = "html/tagSequenceIds",
                 text_sequence_ids_col_name: str = "html/textSequenceIds", mask_sequence_col_name: str = "html/masks",
                 features_col_name: str = "html/featureIds",
                 pad_html_to_max_blocks: bool = True, max_blocks_per_html: int = 85,
                 pad_blocks: str = "max_length", truncate_blocks: bool = True):
        """

        :param tokenizer_name: name of the tokenizer, must be in the HuggingFace Hub
        :param use_fast_tokenizer: whether to use fast tokenizer or not. If so, the fast version must be in HuggingFace Hub.
        :param class_sequence_col_name: column name in the CSV file that contains the class sequences
        :param tag_sequence_col_name: column name in the CSV file that contains the tag sequences
        :param text_sequence_col_name: column name in the CSV file that contains the text sequences
        :param html_id_col_name: column name in the CSV file that contains the HTML IDs
        :param part_col_name: column name in the CSV file that contains the parts of the HTML
        :param label_col_name: column name in the CSV file that contains the labels
        :param features_col_name: column name in the CSV file that contains the features
        :param class_sequence_ids_col_name: column name that will contain the class sequence IDs
        :param tag_sequence_ids_col_name: column name that will contain the tag sequence IDs
        :param text_sequence_ids_col_name: column name that will contain the text sequence IDs
        :param mask_sequence_col_name: column name in the CSV file that contains the masks sequences
        :param pad_html_to_max_blocks: whether to pad the HTML to the maximum number of blocks or not
        :param max_blocks_per_html: the maximum number of text blocks in a single HTML.
                                    If the number exceeds this, the HTML will be split evenly.
        :param pad_blocks: whether to pad each text block or not. Options are: 'max_length', 'longest', or 'do_not_pad'.
        :param truncate_blocks: whether to truncate each text block to the maximum acceptable input length for the model
        """
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
        if pad_blocks not in self.pad_blocks_options:
            raise ValueError(f"Pad blocks option is not valid! Choose between: {self.pad_blocks_options}")
        self.pad_blocks = pad_blocks
        self.truncate_blocks = truncate_blocks

    def process_text_block(self, class_sequence: list, tag_sequence: list, text_sequence: list):
        class_sequence = str(class_sequence)[1:-1]
        tag_sequence = str(tag_sequence)[1:-1]
        text_sequence = str(text_sequence)[1:-1]

        class_sequence_ids = self.tokenizer(class_sequence, padding=self.pad_blocks, truncation=self.truncate_blocks)[
            "input_ids"]
        tag_sequence_ids = self.tokenizer(tag_sequence, padding=self.pad_blocks, truncation=self.truncate_blocks)[
            "input_ids"]
        text_sequence_ids = self.tokenizer(text_sequence, padding=self.pad_blocks, truncation=self.truncate_blocks)[
            "input_ids"]

        return [class_sequence_ids, tag_sequence_ids, text_sequence_ids]

    def process_html(self, text_blocks: list = None, class_sequences: list = None, tag_sequences: list = None,
                     text_sequences: list = None):
        class_sequence = [str(class_sequence)[1:-1] for class_sequence in class_sequences]
        tag_sequence = [str(tag_sequence)[1:-1] for tag_sequence in tag_sequences]
        text_sequence = [str(text_sequence)[1:-1] for text_sequence in text_sequences]

        class_sequence_ids = self.tokenizer(class_sequence, padding=self.pad_blocks, truncation=self.truncate_blocks)[
            "input_ids"]
        tag_sequence_ids = self.tokenizer(tag_sequence, padding=self.pad_blocks, truncation=self.truncate_blocks)[
            "input_ids"]
        text_sequence_ids = self.tokenizer(text_sequence, padding=self.pad_blocks, truncation=self.truncate_blocks)[
            "input_ids"]

        return [class_sequence_ids, tag_sequence_ids, text_sequence_ids]

    def process_dataset(self, dataset_df: pd.DataFrame = None, dataset_file_path: str = "",
                        class_sequence_col_name: str = None, tag_sequence_col_name: str = None,
                        text_sequence_col_name: str = None, html_id_col_name: str = None, part_col_name: str = None):
        """

        :param dataset_df: dataframe of the dataset that is the output of M52's SemText HTML Preprocessor
        :param dataset_file_path: CSV file that is the output of M52's SemText HTML Preprocessor
        :param class_sequence_col_name: Column name of class sequence in the dataframe/CSV file
        :param tag_sequence_col_name: Column name of tag sequence in the dataframe/CSV file
        :param text_sequence_col_name: Column name of text sequence in the dataframe/CSV file
        :param html_id_col_name: Column name of HTML original ID in the dataframe/CSV file
        :param part_col_name: Column name of the HTML part in the dataframe/CSV file
        :return: a dataframe with columns containing the IDs of the tokenized sequences
                and formatted so that each row corresponds to a sample of data (sequences of text blocks)
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
        df[self.class_sequence_ids_col_name] = self.tokenizer(class_sequence, padding=self.pad_blocks,
                                                              truncation=self.truncate_blocks)["input_ids"]
        df[self.tag_sequence_ids_col_name] = self.tokenizer(tag_sequence, padding=self.pad_blocks,
                                                            truncation=self.truncate_blocks)["input_ids"]
        df[self.text_sequence_ids_col_name] = self.tokenizer(text_sequence, padding=self.pad_blocks,
                                                             truncation=self.truncate_blocks)["input_ids"]

        # group by the html ID name and the part, squeezing the sequences in other columns to a list
        df = df.groupby([html_id_col_name, part_col_name]).aggregate(lambda x: list(x)).reset_index(
            [html_id_col_name, part_col_name])

        # add padding blocks so that all HTML sequences are of the same size
        if self.pad_html_to_max_blocks:
            pad_block = self.tokenizer.model_max_length * [self.tokenizer.pad_token_id]
            df[self.class_sequence_ids_col_name] = df[self.class_sequence_ids_col_name].apply(
                lambda seq: seq + (self.max_blocks_per_html - len(seq)) * [pad_block])
            df[self.tag_sequence_ids_col_name] = df[self.tag_sequence_ids_col_name].apply(
                lambda seq: seq + (self.max_blocks_per_html - len(seq)) * [pad_block])
            df[self.text_sequence_ids_col_name] = df[self.text_sequence_ids_col_name].apply(
                lambda seq: seq + (self.max_blocks_per_html - len(seq)) * [pad_block])
            df[self.label_col_name] = df[self.label_col_name].apply(
                lambda labels: [*labels, *[0] * (self.max_blocks_per_html - len(labels))])

        # create mask column, to accommodate for text block paddings
        # (1 means the text block is not padding, 0 means text block is padding)
        df[self.mask_sequence_col_name] = df[self.text_sequence_ids_col_name].apply(
            lambda seq: [0 if all(i == self.tokenizer.pad_token_id for i in ids) else 1 for ids in seq])

        # combine class, tag, and text IDs into one feature column
        df[self.features_col_name] = df[[self.class_sequence_ids_col_name, self.tag_sequence_ids_col_name,
                                         self.text_sequence_ids_col_name]].values.tolist()
        # df[self.features_col_name] = df[self.features_col_name].apply(lambda features: np.reshape(features, (self.max_blocks_per_html, 3, self.tokenizer.model_max_length)).tolist())
        return df
