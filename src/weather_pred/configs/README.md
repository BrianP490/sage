# 🗂️ Configuration File Documentation

This document describes the parameters available in the [`./configs/config.json`](./config.json) file.

Note: See ['Simulated Args'](#simulated-args) section for more information on ipynb tesing.

## Data

Covers the essential data configurations for the Data Pipeline

- `dataset_url`  
  - **Type**: `str`  
  - **Default**: `PASTE URL HERE  (ex. 'hf://datasets/MaxPrestige/fake-dataset.csv')`  
  - **Description**: URL to the dataset hosted on Hugging Face.

- `root_data_dir`  
  - **Type**: `str`  
  - **Default**: `"./Data"`  
  - **Description**: Root directory where data files are stored.

- `data_file_path`  
  - **Type**: `str`  
  - **Default**: `ENTER DATA FILE NAME HERE (ex. "liquid_gold_data.csv")`  
  - **Description**: Filename of the dataset.

- `data_splits_dir`  
  - **Type**: `str`  
  - **Default**: `"DataSplits"`  
  - **Description**: Directory containing train/val/test splits.

- `scaler_dir`  
  - **Type**: `str`  
  - **Default**: `"Scalers"`  
  - **Description**: Directory to save/load data scalers.

- `target_column`  
  - **Type**: `str`  
  - **Default**: `ENTER TARGET COLUMN NAME HERE (ex. "Price")`  
  - **Description**: Column name to be predicted.

- `extra_dropped_columns`  
  - **Type**: `list`  
  - **Default**: `["ADD ANY EXTRA COLUMNS THAT NEED TO BE DROPPED HERE"]  -OR- KEEP EMPTY & REMOVE STRING (ex. [])`
  - **Description**: List of columns to drop from the dataset.

## Argparse Parser Defaults

Used to provide the argparse parser default values and attributes.

- `epochs`  
  - **Type**: `int`  
  - **Default**: `8`  
  - **Description**: Number of training epochs to run. Note: Overwritten in ['Simulated Args'](#simulated-args) section during ipynb testing.

- `learning_rate`  
  - **Type**: `float`  
  - **Default**: `0.0003`  
  - **Description**: Learning rate used by the optimizer. Note: Overwritten in ['Simulated Args'](#simulated-args) section during ipynb testing.

- `max_grad_norm`  
  - **Type**: `float`  
  - **Default**: `3.0`  
  - **Description**: Maximum L2 norm for gradient clipping.

- `dataloader_batch_size`  
  - **Type**: `int`  
  - **Default**: `64`  
  - **Description**: Batch size for training and evaluation.

- `dataloader_pin_memory`
  - **Type**: `bool`  
  - **Default**: `false` 
  - **Description**: Use flag to Enable pinned memory in dataloaders. Note: Overwritten in ['Simulated Args'](#simulated-args) section during ipynb testing.

- `dataloader_num_workers`  
  - **Type**: `int`  
  - **Default**: `0`  
  - **Description**: Number of subprocesses for data loading.

- `log_iterations`  
  - **Type**: `int`  
  - **Default**: `32`  
  - **Description**: Frequency of logging training progress. Note: Overwritten in ['Simulated Args'](#simulated-args) section during ipynb testing.

- `eval_iterations` 
  - **Type**: `int`  
  - **Default**: `32`  
  - **Description**: Frequency of model evaluation. Note: Overwritten in ['Simulated Args'](#simulated-args) section during ipynb testing.

- `use_cuda`
  - **Type**: `bool`  
  - **Default**: `false` 
  - **Description**: Use flag to Enable CUDA for training if available. Overwritten in ['Simulated Args'](#simulated-args) section during ipynb testing.

- `device`
  - **Type**: `str`  
  - **Default**: `"cpu"`  
  - **Description**: Device to use for training (e.g., 'cpu', 'cuda:0').

- `save_model`
  - **Type**: `bool`  
  - **Default**: `false` 
  - **Description**: Use flag to save the trained model after training. Note: Overwritten in ['Simulated Args'](#simulated-args) section during ipynb testing.

- `model_output_path`
  - **Type**: `str`
  - **Default**: `ENTER MODEL OUTPUT PATH HERE (ex. "./Models/trained_model.pt")`  
  - **Description**: Path to save the trained model. Note: Overwritten in ['Simulated Args'](#simulated-args) section during ipynb testing.

## Model

For the model architecture, such as the number of repeating model blocks.

- `in_dim`  
  - **Type**: `int`  
  - **Default**: `ENTER INPUT DIMENSION SIZE HERE (ex. 5)`  
  - **Description**: Input dimension size.

- `intermediate_dim`  
  - **Type**: `int`  
  - **Default**: `128`  
  - **Description**: Size of intermediate hidden layers.

- `out_dim`  
  - **Type**: `int`  
  - **Default**: `ENTER OUTPUT DIMENSION SIZE HERE (ex. 2)`  
  - **Description**: Output dimension size.

- `num_blocks`  
  - **Type**: `int`  
  - **Default**: `12`  
  - **Description**: Number of model blocks used.

- `dropout_rate`  
  - **Type**: `float`  
  - **Default**: `0.1`  
  - **Description**: Dropout rate for regularization.

## Logging

Used to configure the error/info logging behaviors.

- `log_to_file`  
  - **Type**: `bool`  
  - **Default**: `True`  
  - **Description**: Enable logging to a file.

- `log_file`  
  - **Type**: `str`  
  - **Default**: `"logs/app.log"`  
  - **Description**: Path to the log file.

- `logger_name`  
  - **Type**: `str`  
  - **Default**: `"main"`  
  - **Description**: Name of the logger instance.

- `log_level`  
  - **Type**: `str`  
  - **Default**: `"INFO"`  
  - **Description**: Logging level (e.g., INFO, DEBUG).

- `log_mode`  
  - **Type**: `str`  
  - **Default**: `"w"`  
  - **Description**: File mode for logging (e.g., 'w' for overwrite, 'a' for appending).

- `log_format`  
  - **Type**: `str`  
  - **Default**: `"%(asctime)s - %(name)s - %(levelname)s - %(message)s"`  
  - **Description**: Format of the logging method in the logger.

- `date_format`  
  - **Type**: `str`  
  - **Default**: `"%Y-%m-%d %H:%M:%S"`  
  - **Description**: Date and Time format of the log messages.

- `log_to_console`  
  - **Type**: `bool`  
  - **Default**: `True`  
  - **Description**: Enable logging to the console.


## Training Defaults

Used to provide the argparse parser default values and attributes.

- `optimizer`  
  - **Type**: `str`  
  - **Default**: `"adamw"`  
  - **Description**: The optimizer used for training.

- `loss_function`  
  - **Type**: `str`  
  - **Default**: `"mae"`  
  - **Description**: The loss function used for training.

- `early_stopping`  
  - **Type**: `bool`  
  - **Default**: `false`  
  - **Description**: Enable or disable early stoppage.

- `early_stopping_patience`  
  - **Type**: `int`  
  - **Default**: `10`  
  - **Description**: Patience for early stoppage.

- `checkpoint_dir`  
  - **Type**: `str`  
  - **Default**: `"./checkpoints"`  
  - **Description**: Directory to store the checkpoints.


## Simulated Args

<details>

Required for ipynb testing simulating possible user input, due to the argparser & ipynb incompatability. Any entry from [Argparse Parser Defaults](#argparse-parser-defaults) can be added here with different 'Default' values.

- `--epochs`  
  - **Type**: `str`  
  - **Default**: `"1"`  
  - **Description**: Override number of epochs for simulation.

- `--learning_rate`  
  - **Type**: `str`  
  - **Default**: `"0.003"`  
  - **Description**: Override learning rate for simulation.

- `--log_iterations`  
  - **Type**: `str`  
  - **Default**: `"256"`  
  - **Description**: Override log frequency for simulation.

- `--eval_iterations`  
  - **Type**: `str`  
  - **Default**: `"64"`  
  - **Description**: Override evaluation frequency for simulation.

- `--dataloader_pin_memory`  
  - **Type**: `bool`  
  - **Default**: `false`  
  - **Description**: Override dataloader pin memory behavior.

- `--use_cuda`  
  - **Type**: `bool`  
  - **Default**: `false`  
  - **Description**: Override cuda behavior.

- `--save_model`  
  - **Type**: `bool`  
  - **Default**: `false`  
  - **Description**: Override model saving behavior.

- `--model_output_path`  
  - **Type**: `str`  
  - **Default**: `ENTER MODEL OUTPUT PATH HERE (ex. "./Models/my_model.pt")`
  - **Description**: Override model output path.

</details>
