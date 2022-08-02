from typing import Union, List

from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader

import pandas as pd
import torch

from m_semtext.data_processor_m_semtext import MSemTextDataProcessor


class MSemTextDataset(Dataset):
    def __init__(self, dataset_file_path: Union[str, List[str]] = None, tokenizer_name: str = "xlm-roberta-base",
                 use_fast_tokenizer: bool = True,
                 class_sequence_col_name: str = "html/classSequence", tag_sequence_col_name: str = "html/tagSequence",
                 text_sequence_col_name: str = "html/textSequence", class_sequence_ids_col_name: str = "html/classSequenceIds",
                 tag_sequence_ids_col_name: str = "html/tagSequenceIds",
                 text_sequence_ids_col_name: str = "html/textSequenceIds",
                 html_id_col_name: str = "html/originalId", part_col_name: str = "html/part",
                 label_col_name: str = "html/label",
                 features_col_name: str = "html/featureIds", mask_sequence_col_name: str = "html/masks",
                 pad_html_to_max_blocks: bool = True, max_blocks_per_html: int = 85, pad_blocks: str = "max_length",
                 truncate_blocks: bool = True
                 ):
        """

        :param tokenizer_name: Name of the tokenizer model to tokenize the data.
                               Normally it is the same as the embedding model name.
                               The tokenizer must be in HuggingFace Hub.
        :param use_fast_tokenizer: Whether to use fast tokenizer or not. If so, the fast version must be in HuggingFace Hub.
        :param class_sequence_col_name: Column name in the CSV file that contains the class sequences.
        :param tag_sequence_col_name: Column name in the CSV file that contains the tag sequences.
        :param text_sequence_col_name: Column name in the CSV file that contains the text sequences.
        :param html_id_col_name: Column name in the CSV file that contains the HTML IDs.
        :param part_col_name: Column name in the CSV file that contains the parts of the HTML.
        :param label_col_name: Column name in the CSV file that contains the labels.
        :param features_col_name: Column name in the CSV file that contains the features.
        :param mask_sequence_col_name: Column name in the CSV file that contains the masks sequences.
        :param pad_html_to_max_blocks: Whether to pad the HTML to the maximum number of blocks or not.
        :param max_blocks_per_html: The maximum number of text blocks in a single HTML.
                                    If the number exceeds this, the HTML will be split evenly.
        :param pad_blocks: Whether to pad each text block or not. Options are: 'max_length', 'longest', or 'do_not_pad'.
        :param truncate_blocks: Whether to truncate each text block to the maximum acceptable input length for the model.
        """
        # original dataset dataframe
        if type(dataset_file_path) == list:
            self.dataset_df = pd.concat((pd.read_csv(f) for f in dataset_file_path))
        else:
            self.dataset_df = pd.read_csv(dataset_file_path)
        self.label_col_name = label_col_name
        self.features_col_name = features_col_name
        self.mask_col_name = mask_sequence_col_name
        self.class_sequence_ids_col_name = class_sequence_ids_col_name
        self.tag_sequence_ids_col_name = tag_sequence_ids_col_name
        self.text_sequence_ids_col_name = text_sequence_ids_col_name
        self.dataset_processor = MSemTextDataProcessor(tokenizer_name=tokenizer_name, use_fast_tokenizer=use_fast_tokenizer,
                                                       class_sequence_col_name=class_sequence_col_name,
                                                       tag_sequence_col_name=tag_sequence_col_name,
                                                       text_sequence_col_name=text_sequence_col_name,
                                                       class_sequence_ids_col_name=class_sequence_ids_col_name,
                                                       tag_sequence_ids_col_name=tag_sequence_ids_col_name,
                                                       text_sequence_ids_col_name=text_sequence_ids_col_name,
                                                       html_id_col_name=html_id_col_name, part_col_name=part_col_name,
                                                       label_col_name=label_col_name,
                                                       mask_sequence_col_name=mask_sequence_col_name,
                                                       features_col_name=features_col_name,
                                                       pad_html_to_max_blocks=pad_html_to_max_blocks,
                                                       max_blocks_per_html=max_blocks_per_html, pad_blocks=pad_blocks,
                                                       truncate_blocks=truncate_blocks)
        # processed dataset where each row is an input to the model
        self.dataset = self.dataset_processor.process_dataset(self.dataset_df)
        # processed dataset dataframe with the same shape as the original dataframe
        columns_to_explode = [class_sequence_col_name, tag_sequence_col_name, text_sequence_col_name,
                              class_sequence_ids_col_name, tag_sequence_ids_col_name, text_sequence_ids_col_name,
                              mask_sequence_col_name, label_col_name]
        if 'postgresId' in self.dataset_df.columns:
            columns_to_explode.append('postgresId')
        self.processed_df = self.dataset.apply(lambda row: self._remove_padding(row), axis=1).explode(
            column=columns_to_explode)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        features = self.dataset.loc[self.dataset.index[idx], self.features_col_name]
        features = torch.tensor(features)
        labels = self.dataset.loc[self.dataset.index[idx], self.label_col_name]
        labels = torch.tensor(labels, dtype=torch.long)
        masks = self.dataset.loc[self.dataset.index[idx], self.mask_col_name]
        masks = torch.tensor(masks, dtype=torch.uint8)
        return features, labels, masks

    def get_labels(self):
        labels = self.dataset[self.label_col_name].tolist()
        labels = torch.tensor(labels, dtype=torch.long)
        return labels

    def _remove_padding(self, row):
        len_count = row[self.mask_col_name].count(1)
        row[self.class_sequence_ids_col_name] = row[self.class_sequence_ids_col_name][:len_count]
        row[self.tag_sequence_ids_col_name] = row[self.tag_sequence_ids_col_name][:len_count]
        row[self.text_sequence_ids_col_name] = row[self.text_sequence_ids_col_name][:len_count]
        feature_ids = []
        for feature in row[self.features_col_name]:
            feature_ids.append(feature[:len_count])
        row[self.features_col_name] = feature_ids
        row[self.label_col_name] = row[self.label_col_name][:len_count]
        row[self.mask_col_name] = row[self.mask_col_name][:len_count]
        return row


class MSemTextDataModule(LightningDataModule):
    def __init__(self, train_set_file_path: Union[str, List[str]] = None,
                 val_set_file_path: Union[str, List[str]] = None, test_set_file_path: Union[str, List[str]] = None,
                 predict_set_file_path: Union[str, List[str]] = None,
                 train_dataset: MSemTextDataset = None, val_dataset: MSemTextDataset = None,
                 test_dataset: MSemTextDataset = None, predict_dataset: MSemTextDataset = None,
                 batch_size: int = 8, tokenizer_name: str = "xlm-roberta-base", use_fast_tokenizer: bool = True,
                 pad_html_to_max_blocks: bool = True, max_blocks_per_html: int = 85, pad_blocks: str = "max_length",
                 truncate_blocks: bool = True):
        """

        :param train_set_file_path: Path to a CSV file containing the training data.
        :param val_set_file_path: Path to a CSV file containing the validation data.
        :param test_set_file_path: Path to a CSV file containing the test data.
        :param batch_size: Batch size of data to give to the model during training, validation, and evaluation.
        :param tokenizer_name: Name of the tokenizer model to tokenize the data.
                               Normally it is the same as the embedding model name.
                               The tokenizer must be in HuggingFace Hub.
        :param use_fast_tokenizer: Whether to use fast tokenizer or not. If so, the fast version must be in HuggingFace Hub.
        :param pad_html_to_max_blocks: Whether to pad the HTML to the maximum number of blocks or not.
        :param max_blocks_per_html: The maximum number of text blocks in a single HTML.
                                    If the number exceeds this, the HTML will be split evenly.
        :param pad_blocks: Whether to pad each text block or not. Options are: 'max_length', 'longest', or 'do_not_pad'.
        :param truncate_blocks: Whether to truncate each text block to the maximum acceptable input length for the model.
        """
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.predict_dataset = predict_dataset
        self.train_set_file_path = train_set_file_path
        self.val_set_file_path = val_set_file_path
        self.test_set_file_path = test_set_file_path
        self.predict_set_file_path = predict_set_file_path
        self.batch_size = batch_size
        self.tokenizer_name = tokenizer_name
        self.use_fast_tokenizer = use_fast_tokenizer
        self.pad_html_to_max_blocks = pad_html_to_max_blocks
        self.max_blocks_per_html = max_blocks_per_html
        self.pad_blocks = pad_blocks
        self.truncate_blocks = truncate_blocks

    def prepare_data(self):
        if self.train_set_file_path:
            print("Processing training data...")
            self.train_dataset = MSemTextDataset(dataset_file_path=self.train_set_file_path,
                                                 tokenizer_name=self.tokenizer_name,
                                                 use_fast_tokenizer=self.use_fast_tokenizer,
                                                 pad_html_to_max_blocks=self.pad_html_to_max_blocks,
                                                 max_blocks_per_html=self.max_blocks_per_html,
                                                 pad_blocks=self.pad_blocks, truncate_blocks=self.truncate_blocks)
        if self.val_set_file_path:
            print("Processing validation data...")
            self.val_dataset = MSemTextDataset(dataset_file_path=self.val_set_file_path,
                                               tokenizer_name=self.tokenizer_name,
                                               use_fast_tokenizer=self.use_fast_tokenizer,
                                               pad_html_to_max_blocks=self.pad_html_to_max_blocks,
                                               max_blocks_per_html=self.max_blocks_per_html,
                                               pad_blocks=self.pad_blocks, truncate_blocks=self.truncate_blocks)
        if self.test_set_file_path:
            print("Processing test data...")
            self.test_dataset = MSemTextDataset(dataset_file_path=self.test_set_file_path,
                                                tokenizer_name=self.tokenizer_name,
                                                use_fast_tokenizer=self.use_fast_tokenizer,
                                                pad_html_to_max_blocks=self.pad_html_to_max_blocks,
                                                max_blocks_per_html=self.max_blocks_per_html,
                                                pad_blocks=self.pad_blocks, truncate_blocks=self.truncate_blocks)
        if self.predict_set_file_path:
            print("Processing predict data...")
            self.predict_dataset = MSemTextDataset(dataset_file_path=self.predict_set_file_path,
                                                tokenizer_name=self.tokenizer_name,
                                                use_fast_tokenizer=self.use_fast_tokenizer,
                                                pad_html_to_max_blocks=self.pad_html_to_max_blocks,
                                                max_blocks_per_html=self.max_blocks_per_html,
                                                pad_blocks=self.pad_blocks, truncate_blocks=self.truncate_blocks)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.predict_dataset, batch_size=self.batch_size, shuffle=False)
