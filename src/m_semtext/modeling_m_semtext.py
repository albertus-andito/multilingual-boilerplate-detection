from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torchcrf import CRF
from transformers import AutoModel
from typing import List

import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from m_semtext.data_processor_m_semtext import MSemTextDataProcessor


class MSemText(pl.LightningModule):
    def __init__(self, embed_model_name="xlm-roberta-base", embed_dim: int = 768,
                 embedding_feature="pooled_output", num_feature_map: int = 3,
                 filter_sizes: List[int] = [3, 5, 7], num_filters: List[int] = [128, 128, 256],
                 lstm_hidden_size: int = 512, total_length_per_seq: int = 85, num_classes: int = 2):
        super(MSemText, self).__init__()

        self.embedding = AutoModel.from_pretrained(embed_model_name)
        self.embedding_feature = embedding_feature

        if self.embedding_feature == "cnn":
            self.conv1d_list = nn.ModuleList([
                nn.Conv1d(in_channels=embed_dim, out_channels=num_filters[i], kernel_size=filter_sizes[i])
                for i in range(len(filter_sizes))
            ])

        lstm_input_size = num_feature_map * sum(num_filters) if self.embedding_feature == "cnn" else num_feature_map * embed_dim
        self.lstm = nn.LSTM(input_size=lstm_input_size,
                            hidden_size=lstm_hidden_size,
                            bidirectional=True, batch_first=True)
        self.fc = nn.Linear(2 * lstm_hidden_size, num_classes)
        self.crf = CRF(num_classes, batch_first=True)

        self.total_length_per_seq = total_length_per_seq

    def forward(self, input_ids: torch.Tensor, masks: torch.Tensor):
        input_size = input_ids.size()
        batch_size, feature_map_size, num_of_blocks, seq_length = input_size

        # flatten for easier pass to embedding model
        flatten = torch.flatten(input_ids, end_dim=-2)
        attention_mask = (~flatten.eq(1)).long()
        # get the embedding representation of the input
        if self.embedding_feature == "pooled_output":
            x_embed_flatten = model.embedding(input_ids=flatten, attention_mask=attention_mask).pooler_output
            seq_length = 1
        elif self.embedding_feature == "cls":
            x_embed_flatten = model.embedding(input_ids=flatten, attention_mask=attention_mask).last_hidden_state[:, 0 ,:]
            seq_length = 1
        else:
            x_embed_flatten = model.embedding(input_ids=flatten, attention_mask=attention_mask).last_hidden_state
        embedding_size = x_embed_flatten.size()[-1]
        x_embed = torch.reshape(x_embed_flatten, (batch_size, feature_map_size, num_of_blocks, seq_length, embedding_size))

        if self.embedding_feature == "cnn":
            # moves the feature map to first dimension for the looping below
            # and moves the embedding to the fourth dimension for Conv1D layer
            x_reshaped = x_embed.permute(1, 0, 2, 4, 3)

            # TODO: deal with padding here? ignore padding?
            feature_maps = []
            for x_feature_map in x_reshaped:  # loop through tags, classes, and texts
                # concat batches
                x_feature_map_reshaped = x_feature_map.reshape([batch_size * num_of_blocks, embedding_size, seq_length])
                # print(x_feature_map_reshaped.size())

                # pass through different filter sizes
                x_conv_list = [conv1d(x_feature_map_reshaped) for conv1d in self.conv1d_list]

                # max pool
                x_pool_list = [F.max_pool1d(x_conv, kernel_size=x_conv.shape[2]) for x_conv in x_conv_list]

                # concatenate the output of different filter sizes
                x_cat = torch.cat([x_pool.squeeze(dim=2) for x_pool in x_pool_list], dim=1)

                # split batches back to original
                x_cat_reshaped = x_cat.reshape([batch_size, num_of_blocks, x_cat.size()[-1]])
                # print(x_cat_reshaped.size())
                feature_maps.append(x_cat_reshaped)

            # concatenate the output for the different feature maps
            features = torch.cat([feature_map for feature_map in feature_maps], dim=2)
        else:
            # moves the feature map to third dimension for concat
            x_reshaped = x_embed.permute(0, 2, 1, 3, 4)
            features = x_reshaped.reshape(batch_size, num_of_blocks, feature_map_size * seq_length * embedding_size)

        length = [list(mask).count(1) for mask in masks]
        packed_input = pack_padded_sequence(features, torch.tensor(length), batch_first=True, enforce_sorted=False)
        packed_lstm_output, _ = self.lstm(packed_input)  # pass to bidirectional LSTM
        lstm_output, _ = pad_packed_sequence(packed_lstm_output, batch_first=True,
                                             total_length=self.total_length_per_seq)

        emissions = self.fc(lstm_output)  # produce emissions of LSTM

        return emissions

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        return [optimizer], [lr_scheduler]

    def training_step(self, batch, batch_idx):
        x, y, mask = batch
        y_hat = self(x, mask)
        loss = -self.crf(y_hat, y, mask, reduction="token_mean")
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, mask = batch
        y_hat = self(x, mask)
        val_loss = -self.crf(y_hat, y, mask, reduction="token_mean")
        self.log("val_loss", val_loss, on_step=True, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y, mask = batch
        y_hat = self(x, mask)
        test_loss = -self.crf(y_hat, y, mask, reduction="token_mean")
        self.log("test_loss", test_loss, on_step=True, on_epoch=True, prog_bar=True)

    def predict_step(self, batch, batch_idx, **kwargs):
        if len(batch) == 2:
            x, mask = batch
            emissions = self(x, mask)
            pred = self.crf.decode(emissions, mask)
        else:
            x, y, mask = batch
            emissions = self(x, mask)
            pred = self.crf.decode(emissions, mask)
        return pred

if __name__ == '__main__':

    processor = MSemTextDataProcessor()
    model = MSemText(total_length_per_seq=5, embedding_feature="pooled_output")

    classes = ["[]", "[one my div]", "[one my div]", "[one my div]", "[]"]
    tags = ["[body, primary headline]", "[body, division, paragraph]", "[body, division, quinary headline]",
           "[body, division, anchor]", "[body, secondary headline]"]
    text = ["[My First Heading]", "[My first paragraph.]", "[Another heading]", "[www.example.com]", "[Last heading]"]

    features = processor.process_html(class_sequences=classes, tag_sequences=tags, text_sequences=text)
    features = torch.tensor([features, features])
    labels = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1]], dtype=torch.long)
    masks = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1]], dtype=torch.uint8)

    model.training_step((features, labels, masks), 0)

    pred = model.predict_step((features, labels, masks), 0)
    print(pred)

