from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader

import pandas as pd
import torch
from transformers import AutoTokenizer

from m_semtext.data_processor_m_semtext import MSemTextDataProcessor


class MSemTextDataset(Dataset):
    def __init__(self, dataset_file_path="", tokenizer_name="xlm-roberta-base", use_fast_tokenizer=True,
                 class_sequence_col_name="html/classSequence", tag_sequence_col_name="html/tagSequence",
                 text_sequence_col_name="html/textSequence",
                 html_id_col_name="html/originalId", part_col_name="html/part", label_col_name="html/label",
                 features_col_name="html/featureIds", mask_sequence_col_name="html/masks",
                 pad_html_to_max_blocks=True, max_blocks_per_html=85, pad_blocks="max_length", truncate_blocks=True
                 ):
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
    def __init__(self, train_set_file_path: str = "", val_set_file_path: str = "", test_set_file_path: str = "",
                 batch_size: int = 8, tokenizer_name: str = "xlm-roberta-base"):
        super().__init__()
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.train_set_file_path = train_set_file_path
        self.val_set_file_path = val_set_file_path
        self.test_set_file_path = test_set_file_path
        self.batch_size = batch_size
        self.tokenizer_name = tokenizer_name

    def prepare_data(self):
        self.train_dataset = MSemTextDataset(dataset_file_path=self.train_set_file_path,
                                             tokenizer_name=self.tokenizer_name)
        self.val_dataset = MSemTextDataset(dataset_file_path=self.val_set_file_path,
                                           tokenizer_name=self.tokenizer_name)
        self.test_dataset = MSemTextDataset(dataset_file_path=self.test_set_file_path,
                                            tokenizer_name=self.tokenizer_name)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)
