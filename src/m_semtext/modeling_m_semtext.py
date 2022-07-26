from typing import Union

from sklearn.utils import class_weight
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torchcrf import CRF
from torchmetrics import F1Score, MetricCollection, Precision, Recall
from transformers import AutoModel

import itertools
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from m_semtext.data_processor_m_semtext import MSemTextDataProcessor
from m_semtext.dataset_m_semtext import MSemTextDataModule
from m_semtext.output_processor_m_semtext import MSemTextOutputProcessor


class MSemText(pl.LightningModule):

    embedding_feature_options = ["pooled_output", "cls", "mean_pooling", "max_pooling", "cnn"]
    features_combination_options = ["concat", "sum", "mean", "max"]

    def __init__(self, embedding_model_name: str = "xlm-roberta-base",
                 embedding_feature: str = "pooled_output", features_combination: str = "concat",
                 num_feature_map: int = 3,
                 filter_sizes: list = None, num_filters: list = None, lstm_hidden_size: int = 512,
                 total_length_per_seq: int = 85, num_classes: int = 2,
                 continue_pre_train_embedding: bool = False, large_embedding_batch: Union[bool, float] = False,
                 learning_rate: float = 1e-3, class_weights: list = None):
        """

        :param embedding_model_name: Name of the language model to be used in the embedding model.
                                    It must be a model that exists on HuggingFace Hub.
        :param embedding_feature: Feature of the embeddings to be used.
                                The options are "pooled_output", "cls", "mean_pooling", "max_pooling", or "cnn".
        :param features_combination: The operation to do to combine the features (tags, classes, and texts).
                                    The options are "concat", "sum", "mean", and "max".
        :param num_feature_map: The number of features maps. This is used to determine the LSTM input size.
        :param filter_sizes: The sizes of kernels/filters used in the 1D CNN.
        :param num_filters: The numbers of filters/out channels used in the 1D CNN for each of the filter.
        :param lstm_hidden_size: Size of the LSTM hidden units
        :param total_length_per_seq: Total number of text blocks per sequence.
        :param num_classes: The number of classes that is used for prediction.
        :param continue_pre_train_embedding: Whether to continue the language model pre-training with the current training data or not.
        :param large_embedding_batch: When retrieving the embeddings from the language model,
                                    whether to get the embedding in large batch or not. (This could affect memory utilisation).
        """
        super(MSemText, self).__init__()
        if num_filters is None:
            num_filters = [128, 128, 256]
        if filter_sizes is None:
            filter_sizes = [3, 5, 7]
        self.embedding = AutoModel.from_pretrained(embedding_model_name)
        if not continue_pre_train_embedding:
            for param in self.embedding.parameters():
                param.requires_grad = False
        self.padding_token_id = self.embedding.config.pad_token_id
        self.embed_dim = self.embedding.config.hidden_size
        if embedding_feature not in self.embedding_feature_options:
            raise ValueError(f"Embedding feature is not valid! Choose between: {self.embedding_feature_options}")
        self.embedding_feature = embedding_feature
        if features_combination not in self.features_combination_options:
            raise ValueError(f"Features combination is not valid! Choose between: {self.features_combination_options}")
        self.features_combination = features_combination
        self.num_feature_map = num_feature_map

        if self.embedding_feature == "cnn":
            self.conv1d_list = nn.ModuleList([
                nn.Conv1d(in_channels=self.embed_dim, out_channels=num_filters[i], kernel_size=filter_sizes[i])
                for i in range(len(filter_sizes))
            ])

        if self.features_combination == "concat":
            lstm_input_size = self.num_feature_map * sum(num_filters) if self.embedding_feature == "cnn" else self.num_feature_map * self.embed_dim
        else:
            lstm_input_size = sum(num_filters) if self.embedding_feature == "cnn" else self.embed_dim
        self.lstm = nn.LSTM(input_size=lstm_input_size,
                            hidden_size=lstm_hidden_size,
                            bidirectional=True, batch_first=True)
        self.fc = nn.Linear(2 * lstm_hidden_size, num_classes)
        self.crf = CRF(num_classes, batch_first=True)

        self.total_length_per_seq = total_length_per_seq

        self.large_embedding_batch = large_embedding_batch

        self.learning_rate = learning_rate

        self.class_weights = class_weights

        metrics = MetricCollection({
            "precision_micro": Precision(num_classes=2, average="micro"),
            "precision_macro": Precision(num_classes=2, average="macro"),
            "precision_weighted": Precision(num_classes=2, average="weighted"),
            "recall_micro": Recall(num_classes=2, average="micro"),
            "recall_macro": Recall(num_classes=2, average="macro"),
            "recall_weighted": Recall(num_classes=2, average="weighted"),
            "f1_micro": F1Score(num_classes=2, average="micro"),
            "f1_macro": F1Score(num_classes=2, average="macro"),
            "f1_weighted": F1Score(num_classes=2, average="weighted")
        })
        self.train_metrics = metrics.clone(prefix="train_")
        self.validation_metrics = metrics.clone(prefix="val_")
        self.test_metrics = metrics.clone(prefix="test")

        self.save_hyperparameters()

    def forward(self, input_ids: torch.Tensor, masks: torch.Tensor):
        input_size = input_ids.size()
        batch_size, feature_map_size, num_of_blocks, seq_length = input_size

        x_embed = []
        # need to get the BERT embeddings per row because of the batch size would have been large
        # TODO: fix large embedding batch for CNN
        if self.large_embedding_batch == 1 or self.large_embedding_batch is True and self.embedding_feature != "cnn":
            for i, row in enumerate(input_ids):
                flatten = torch.flatten(row, end_dim=-2)
                # get the embedding representation of the input
                x_embed_flatten, seq_length = self._get_embedding(flatten, seq_length)
                unflatten = torch.reshape(x_embed_flatten, (feature_map_size, num_of_blocks, self.embed_dim))
                x_embed.append(unflatten)
        elif self.large_embedding_batch == 0.5 and self.num_feature_map == 3 and self.embedding_feature != "cnn":
            for i, row in enumerate(input_ids):
                first_second_features = torch.cat((row[0], row[1]))
                x_embed_flatten, seq_length = self._get_embedding(first_second_features, seq_length)
                unflatten = torch.reshape(x_embed_flatten, (2, num_of_blocks, self.embed_dim))
                third_feature_embed, seq_length = self._get_embedding(row[2], seq_length)
                third_feature_embed = torch.reshape(third_feature_embed, ((1, num_of_blocks, self.embed_dim)))
                row_embed = torch.cat((unflatten, third_feature_embed), 0)
                x_embed.append(row_embed)
        else:  # need to get the BERT embeddings per feature in each row because of the batch size would have been large
            for i, row in enumerate(input_ids):
                row_embed = []
                for j, feature in enumerate(row):
                    feature_embed, seq_length = self._get_embedding(feature, seq_length)
                    row_embed.append(feature_embed)
                row_embed = torch.stack(row_embed)
                x_embed.append(row_embed)
        x_embed = torch.stack(x_embed)

        if self.embedding_feature == "cnn":
            # moves the feature map to first dimension for the looping below
            # and moves the embedding to the fourth dimension for Conv1D layer
            x_reshaped = x_embed.permute(1, 0, 2, 4, 3)

            # TODO: deal with padding here? ignore padding?
            feature_maps = []
            for x_feature_map in x_reshaped:  # loop through tags, classes, and texts
                # concat batches
                x_feature_map_reshaped = x_feature_map.reshape([batch_size * num_of_blocks, self.embed_dim, seq_length])

                # pass through different filter sizes
                x_conv_list = [conv1d(x_feature_map_reshaped) for conv1d in self.conv1d_list]

                # max pool
                x_pool_list = [F.max_pool1d(x_conv, kernel_size=x_conv.shape[2]) for x_conv in x_conv_list]

                # concatenate the output of different filter sizes
                x_cat = torch.cat([x_pool.squeeze(dim=2) for x_pool in x_pool_list], dim=1)

                # split batches back to original
                x_cat_reshaped = x_cat.reshape([batch_size, num_of_blocks, x_cat.size()[-1]])

                feature_maps.append(x_cat_reshaped)

            if self.features_combination == "concat":
                # concatenate the output for the different feature maps
                features = torch.cat([feature_map for feature_map in feature_maps], dim=2)
            else:
                features = torch.stack(feature_maps, dim=2)
                if self.features_combination == "sum":
                    features = torch.sum(features, dim=2)
                elif self.features_combination == "mean":
                    features = torch.mean(features, dim=2)
                elif self.features_combination == "max":
                    features = torch.max(features, dim=2)[0]
        else:
            if self.features_combination == "concat":
                # moves the feature map to third dimension for concat
                x_reshaped = x_embed.permute(0, 2, 1, 3)
                features = x_reshaped.reshape(batch_size, num_of_blocks, feature_map_size * seq_length * self.embed_dim)
            else:
                features = torch.squeeze(x_embed, 1)
                if self.features_combination == "sum":
                    features = torch.sum(features, 1)
                elif self.features_combination == "mean":
                    features = torch.mean(features, 1)
                elif self.features_combination == "max":
                    features = torch.max(features, 1)[0]

        length = torch.sum(masks, dim=1).to("cpu")
        packed_input = pack_padded_sequence(features, length, batch_first=True, enforce_sorted=False)
        packed_lstm_output, _ = self.lstm(packed_input)  # pass to bidirectional LSTM
        lstm_output, _ = pad_packed_sequence(packed_lstm_output, batch_first=True,
                                             total_length=self.total_length_per_seq)

        emissions = self.fc(lstm_output)  # produce emissions of LSTM

        return emissions

    def _get_embedding(self, inputs, seq_length):
        attention_mask = (~inputs.eq(self.padding_token_id)).long()
        # get the embedding representation of the input
        if self.embedding_feature == "pooled_output":
            embeds = self.embedding(input_ids=inputs, attention_mask=attention_mask).pooler_output
            seq_length = 1
        elif self.embedding_feature == "cls":
            embeds = self.embedding(input_ids=inputs, attention_mask=attention_mask).last_hidden_state[:, 0, :]
            seq_length = 1
        # TODO: verify that this mean pooling is correct
        elif self.embedding_feature == "mean_pooling":
            # embeds = model.embedding(input_ids=inputs, attention_mask=attention_mask).last_hidden_state
            # embeds = embeds.sum(axis=1) / attention_mask.sum(axis=-1).unsqueeze(-1)
            embeds = self.embedding(input_ids=inputs, attention_mask=attention_mask)
            embeds = self._mean_pooling(embeds, attention_mask)
            seq_length = 1
        elif self.embedding_feature == "max_pooling":
            embeds = self.embedding(input_ids=inputs, attention_mask=attention_mask)
            embeds = self._max_pooling(embeds, attention_mask)
            seq_length = 1
        else:
            embeds = self.embedding(input_ids=inputs, attention_mask=attention_mask).last_hidden_state
        return embeds, seq_length

    def _mean_pooling(self, model_output, attention_mask):
        # taken from https://www.sbert.net/examples/applications/computing-embeddings/README.html
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def _max_pooling(self, model_output, attention_mask):
        # taken from https://huggingface.co/sentence-transformers/bert-base-nli-max-tokens
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        token_embeddings[input_mask_expanded == 0] = -1e9  # Set padding tokens to large negative value
        return torch.max(token_embeddings, 1)[0]

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        return [optimizer], [lr_scheduler]

    def _calculate_metrics(self, metrics: MetricCollection, y_hat, y, mask):
        predictions = torch.tensor(list(itertools.chain.from_iterable(self.crf.decode(y_hat, mask)))).to(self.device)
        y_without_mask = [y_list[:mask_list.count(1)] for y_list, mask_list in zip(y.tolist(), mask.tolist())]
        targets = torch.tensor(list(itertools.chain.from_iterable(y_without_mask))).to(self.device)
        output = metrics(predictions, targets)
        self.log_dict(output)

    def training_step(self, batch, batch_idx):
        x, y, mask = batch
        y_hat = self(x, mask)
        if self.class_weights is not None:
            y_hat = torch.mul(y_hat, torch.tensor(self.class_weights))
        loss = -self.crf(y_hat, y, mask, reduction="token_mean")
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        self._calculate_metrics(self.train_metrics, y_hat, y, mask)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, mask = batch
        y_hat = self(x, mask)
        val_loss = -self.crf(y_hat, y, mask, reduction="token_mean")
        self.log("val_loss", val_loss, on_step=True, on_epoch=True, prog_bar=True)

        self._calculate_metrics(self.validation_metrics, y_hat, y, mask)

    def test_step(self, batch, batch_idx):
        x, y, mask = batch
        y_hat = self(x, mask)
        test_loss = -self.crf(y_hat, y, mask, reduction="token_mean")
        self.log("test_loss", test_loss, on_step=True, on_epoch=True, prog_bar=True)

        self._calculate_metrics(self.test_metrics, y_hat, y, mask)

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

#
# if __name__ == '__main__':
#
#     processor = MSemTextDataProcessor()
#
#     classes = ["[]", "[one my div]", "[one my div]", "[one my div]", "[]"]
#     tags = ["[body, primary headline]", "[body, division, paragraph]", "[body, division, quinary headline]",
#            "[body, division, anchor]", "[body, secondary headline]"]
#     text = ["[My First Heading]", "[My first paragraph.]", "[Another heading]", "[www.example.com]", "[Last heading]"]
#
#     features = processor.process_html(class_sequences=classes, tag_sequences=tags, text_sequences=text)
#     features = torch.tensor([features, features])
#     labels = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]], dtype=torch.long)
#     masks = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]], dtype=torch.uint8)
#
#     labels_flattened = torch.flatten(labels)
#     class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(labels_flattened), y=labels_flattened.numpy())
#
#     model = MSemText(total_length_per_seq=5, embedding_feature="pooled_output", class_weights=class_weights)
#
#     model.training_step((features, labels, masks), 0)
#
#     pred = model.predict_step((features, labels, masks), 0)
#     print(pred)

    # data_module = MSemTextDataModule(
    #     train_set_file_path="../../dataset/dataset-test-dev/train.csv",
    #     val_set_file_path="../../dataset/dataset-test-dev/dev.csv",
    #     test_set_file_path="../../dataset/dataset-test-dev/test.csv",
    #     batch_size=2,
    #     tokenizer_name="xlm-roberta-base", use_fast_tokenizer=True)
    # data_module.prepare_data()
    #
    # model = MSemText(embedding_feature="pooled_output", large_embedding_batch=True)
    # trainer = pl.Trainer(accelerator="auto",
    #                      enable_progress_bar=True, max_epochs=1,
    #                      log_every_n_steps=50)
    # predictions = trainer.predict(model, dataloaders=data_module.test_dataloader())
    # print(predictions)
    #
    # output_processor = MSemTextOutputProcessor()
    # df = output_processor.process(predictions, data_module.test_dataloader().dataset)
    # print(df)
