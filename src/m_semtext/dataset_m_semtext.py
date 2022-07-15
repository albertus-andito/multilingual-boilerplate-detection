from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader

import pandas as pd
import torch

from m_semtext.data_processor_m_semtext import MSemTextDataProcessor


class MSemTextDataset(Dataset):
    def __init__(self, dataset_file_path="", tokenizer_name="xlm-roberta-base", use_fast_tokenizer=True,
                 class_sequence_col_name="html/classSequence", tag_sequence_col_name="html/tagSequence",
                 text_sequence_col_name="html/textSequence",
                 html_id_col_name="html/originalId", part_col_name="html/part", label_col_name="html/label",
                 features_col_name="html/featureIds", mask_sequence_col_name="html/masks",
                 pad_html_to_max_blocks=True, max_blocks_per_html=85, pad_blocks="max_length", truncate_blocks=True
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
        dataset_df = pd.read_csv(dataset_file_path)
        self.label_col_name = label_col_name
        self.features_col_name = features_col_name
        self.mask_col_name = mask_sequence_col_name
        self.dataset_processor = MSemTextDataProcessor(tokenizer_name=tokenizer_name, use_fast_tokenizer=use_fast_tokenizer,
                                                       class_sequence_col_name=class_sequence_col_name,
                                                       tag_sequence_col_name=tag_sequence_col_name,
                                                       text_sequence_col_name=text_sequence_col_name,
                                                       html_id_col_name=html_id_col_name, part_col_name=part_col_name,
                                                       label_col_name=label_col_name,
                                                       mask_sequence_col_name=mask_sequence_col_name,
                                                       features_col_name=features_col_name,
                                                       pad_html_to_max_blocks=pad_html_to_max_blocks,
                                                       max_blocks_per_html=max_blocks_per_html, pad_blocks=pad_blocks,
                                                       truncate_blocks=truncate_blocks)
        # process the dataset
        self.dataset = self.dataset_processor.process_dataset(dataset_df)

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


class MSemTextDataModule(LightningDataModule):
    def __init__(self, train_set_file_path: str = None, val_set_file_path: str = None, test_set_file_path: str = None,
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
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.train_set_file_path = train_set_file_path
        self.val_set_file_path = val_set_file_path
        self.test_set_file_path = test_set_file_path
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

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)
