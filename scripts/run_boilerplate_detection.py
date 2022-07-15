import os
import pytorch_lightning as pl
import sys

from dataclasses import dataclass, field

from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import WandbLogger
from sklearn.metrics import classification_report
from transformers import HfArgumentParser
from typing import Optional, List

from m_semtext.data_processor_m_semtext import MSemTextDataProcessor
from m_semtext.dataset_m_semtext import MSemTextDataModule
from m_semtext.modeling_m_semtext import MSemText


@dataclass
class DataTrainingArguments:
    train_file: Optional[str] = field(
        default=None, metadata={"help": "Path to a CSV file containing the training data."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "Path to a CSV file containing the validation data."}
    )
    test_file: Optional[str] = field(
        default=None, metadata={"help": "Path to a CSV file containing the test data."}
    )
    batch_size: int = field(
        default=8,
        metadata={"help": "Batch size of data to give to the model during training, validation, and evaluation."}
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={"help": "Name of the tokenizer model to tokenize the data. "
                          "Normally it is the same as the embedding model name. "
                          "If left empty, the default is the embedding model name."}
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use fast tokenizer or not. "
                          "If so, the fast version must be in HuggingFace Hub."}
    )
    pad_html_to_max_blocks: bool = field(
        default=True, metadata={"help": "Whether to pad the HTML to the maximum number of blocks or not"}
    )
    max_blocks_per_html: int = field(
        default=85,
        metadata={"help": "The maximum number of text blocks in a single HTML. "
                          "If the number exceeds this, the HTML will be split evenly."}
    )
    pad_blocks: str = field(
        default="max_length",
        metadata={"help": "Whether to pad each text block or not. "
                          "Options are: 'max_length', 'longest', or 'do_not_pad'."}
    )
    truncate_blocks: bool = field(
        default=True,
        metadata={"help": "Whether to truncate each text block to the maximum acceptable input length for the model."}
    )

    def __post_init__(self):
        if self.train_file is None and self.validation_file is None and self.test_file is None:
            raise ValueError("At least one of train_file, validation_file, and test_file needs to be given.")
        # if self.tokenizer_name is None:
        #     raise ValueError("tokenizer_name should not be empty.")
        if self.pad_blocks not in MSemTextDataProcessor.pad_blocks_options:
            raise ValueError(f"Pad blocks option is not valid! Choose between: {MSemTextDataProcessor.pad_blocks_options}")


@dataclass
class ModelArguments:
    embedding_model_name: Optional[str] = field(
        default=None,
        metadata={"help": "Name of the language model to be used for the embedding model. "
                          "It must be a model that exists on HuggingFace Hub."}
    )
    embedding_feature: Optional[str] = field(
        default="cls",
        metadata={"help": "Feature of the embeddings to be used. "
                          "The options are 'pooled_output', 'cls', 'mean_pooling', 'max_pooling', or 'cnn'."}
    )
    features_combination: Optional[str] = field(
        default="concat",
        metadata={"help": "The operation to do to combine the features (tags, classes, and texts). "
                          "The options are 'concat', 'sum', 'mean', and 'max'."}
    )
    num_feature_map: Optional[int] = field(
        default=3, metadata={"help": "The number of features maps. This is used to determine the LSTM input size."}
    )
    filter_sizes: Optional[List[int]] = field(
        default=None, metadata={"help": "The sizes of kernels/filters used in the 1D CNN."}
    )
    num_filters: Optional[List[int]] = field(
        default=None, metadata={"help": "The numbers of filters/out channels used in the 1D CNN for each of the filter."}
    )
    lstm_hidden_size: Optional[int] = field(
        default=512, metadata={"help": "Size of the LSTM hidden units."}
    )
    total_length_per_seq: Optional[int] = field(
        default=85, metadata={"help": "Total number of text blocks per sequence."}
    )
    num_classes: Optional[int] = field(
        default=2, metadata={"help": "The number of classes that is used for prediction."}
    )
    continue_pre_train_embedding: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to continue the language model pre-training with the current training data or not."}
    )
    large_embedding_batch: Optional[bool] = field(
        default=False,
        metadata={"help": "When retrieving the embeddings from the language model, "
                          "whether to get the embedding in large batch or not. (This could affect memory utilisation)."}
    )
    learning_rate: Optional[float] = field(
        default=1e-3,
        metadata={"help": "Learning rate used when training the model."}
    )

    def __post_init__(self):
        if self.embedding_model_name is None:
            raise ValueError("embedding_model_name should not be empty.")
        if self.embedding_feature not in MSemText.embedding_feature_options:
            raise ValueError(f"Embedding feature is not valid! Choose between: {MSemText.embedding_feature_options}")
        if self.features_combination not in MSemText.features_combination_options:
            raise ValueError(f"Features combination is not valid! Choose between: {MSemText.features_combination_options}")


@dataclass
class TrainingArguments:
    accelerator: Optional[str] = field(
        default="auto",
        metadata={"help": "Supports passing different accelerator types ('cpu', 'gpu', 'tpu', 'ipu', 'auto')"}
    )
    auto_find_learning_rate: bool = field(
        default=False,
        metadata={"help": "Whether to run a learning rate finder algorithm to find optimal initial learning rate."}
    )
    enable_progress_bar: bool = field(
        default=True,
        metadata={"help": "Whether to enable or disable the progress bar."}
    )
    max_epochs: int = field(
        default=10,
        metadata={"help": "Stop training once this number of epochs is reached."}
    )
    log_every_n_steps: int = field(
        default=50,
        metadata={"help": "How often to add logging rows."}
    )
    use_wandb_logger: bool = field(
        default=False,
        metadata={"help": "Whether to use Weight and Biases logger or not. "
                          "If so, you need to make sure that you've logged in by running `wandb login` "
                          "and putting your API key."}
    )
    wandb_project_name: Optional[str] = field(
        default=None,
        metadata={"help": "Weight and Biases project name that will be used only if use_wandb_logger is True."}
    )
    wandb_run_name: Optional[str] = field(
        default=None,
        metadata={"help": "Weight and Biases run name that will be used only if use_wandb_logger is True."}
    )
    do_train: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to run training or not."}
    )
    do_test: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to run testing or not."}
    )
    do_predict: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to run prediction or not."}
    )
    checkpoint_dir: Optional[str] = field(
        default=os.getcwd(),
        metadata={"help": "Path to the checkpoint directory to obtain and store the trained model."}
    )
    input_model_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the .ckpt file where the model is going to be loaded from."}
    )
    output_model_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the file where the final model is going to be saved."}
    )
    run_classification_report: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether to run classification report or not after prediction."
                          "This will only run if do_predict is True."}
    )
    # gpus: Optional[Union[int, list]] = field(
    #
    # )
    def __post_init__(self):
        if self.use_wandb_logger and self.wandb_project_name is None:
            raise ValueError("wandb_project_name must not be empty if use_wandb_logger is True!")
        if not self.do_train and not self.do_predict:
            raise ValueError("At least one of do_train and do_predict must not be None!")


def get_prediction_classification_report(predictions, prediction_data_loader):
    predictions = [p for pred in predictions for pre in pred for p in pre]

    y_test = []
    for x_batch, y_batch, mask_batch in prediction_data_loader:
        for y_list, mask_list in zip(y_batch.tolist(), mask_batch.tolist()):
            y_test.append(y_list[:mask_list.count(1)])
    y_test = [y for yy in y_test for y in yy]

    return classification_report(predictions, y_test)


if __name__ == "__main__":
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if data_args.tokenizer_name is None and model_args.embedding_model_name is not None:
        data_args.tokenizer_name = model_args.embedding_model_name

    print("Preparing datasets...")
    data_module = MSemTextDataModule(
        train_set_file_path=data_args.train_file,
        val_set_file_path=data_args.validation_file,
        test_set_file_path=data_args.test_file,
        batch_size=data_args.batch_size,
        tokenizer_name=data_args.tokenizer_name, use_fast_tokenizer=data_args.use_fast_tokenizer,
        pad_html_to_max_blocks=data_args.pad_html_to_max_blocks, max_blocks_per_html=data_args.max_blocks_per_html,
        pad_blocks=data_args.pad_blocks, truncate_blocks=data_args.truncate_blocks)
    data_module.prepare_data()
    print("Datasets prepared!")
    print("=========================================================")

    print("Initializing model...")
    if training_args.input_model_path:
        print(f"Initializing model from checkpoint: {training_args.input_model_path}")
        model = MSemText.load_from_checkpoint(training_args.input_model_path)
    else:
        model = MSemText(embedding_model_name=model_args.embedding_model_name,
                         embedding_feature=model_args.embedding_feature, features_combination=model_args.features_combination,
                         num_feature_map=model_args.num_feature_map, filter_sizes=model_args.filter_sizes,
                         num_filters=model_args.num_filters, lstm_hidden_size=model_args.lstm_hidden_size,
                         total_length_per_seq=model_args.total_length_per_seq, num_classes=model_args.num_classes,
                         continue_pre_train_embedding=model_args.continue_pre_train_embedding,
                         large_embedding_batch=model_args.large_embedding_batch, learning_rate=model_args.learning_rate)
    print("Model initialized!")
    print("=========================================================")

    if training_args.use_wandb_logger:
        logger = WandbLogger(project=training_args.wandb_project_name, name=training_args.wandb_run_name)
    else:
        logger = True
    seed_everything(42, workers=True)
    trainer = pl.Trainer(accelerator=training_args.accelerator, auto_lr_find=training_args.auto_find_learning_rate,
                         enable_progress_bar=training_args.enable_progress_bar, max_epochs=training_args.max_epochs,
                         log_every_n_steps=training_args.log_every_n_steps, logger=logger, deterministic=True,
                         default_root_dir=training_args.checkpoint_dir)

    if training_args.do_train:
        if data_args.train_file is None:
            raise ValueError("train_file must be provided if do_train is True!")
        if training_args.auto_find_learning_rate:
            print("Run hyperparameter tuning...")
            if data_args.validation_file is None:
                trainer.tune(model, train_dataloaders=data_module.train_dataloader())
            else:
                trainer.tune(model, train_dataloaders=data_module.train_dataloader(), val_dataloaders=data_module.val_dataloader())
            print("Hyperparameter tuning done!")
            print("=========================================================")
        print("Training model...")
        if data_args.validation_file is None:
            trainer.fit(model, train_dataloaders=data_module.train_dataloader())
        else:
            trainer.fit(model, train_dataloaders=data_module.train_dataloader(), val_dataloaders=data_module.val_dataloader())
        print("Model trained!")
        print("=========================================================")

        if training_args.output_model_path:
            trainer.save_checkpoint(training_args.output_model_path)

    if training_args.do_test:
        if data_args.test_file is None:
            raise ValueError("test_file must be provided if do_test is True!")
        else:
            trainer.test(model, dataloaders=data_module.test_dataloader())

    if training_args.do_predict:
        if data_args.test_file is None:
            raise ValueError("test_file must be provided if do_test is True!")
        else:
            predictions = trainer.predict(model, dataloaders=data_module.test_dataloader())
            # TODO: save output in a file
            if training_args.run_classification_report:
                report = get_prediction_classification_report(predictions, data_module.test_dataloader())
                print(report)