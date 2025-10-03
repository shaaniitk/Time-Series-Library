# Training Process Analysis for `train_pgat_synthetic.py`

This document provides a detailed analysis of the training process orchestrated by `scripts/train/train_pgat_synthetic.py`, identifies potential memory issues, and proposes solutions to mitigate them.

## 1. Overview

The `train_pgat_synthetic.py` script is a training harness for the PGAT (Pyramidal Gated Attention Transformer) model on a synthetically generated multi-wave dataset. It is designed to not only train the model but also to monitor gradient norms and memory usage throughout the training process, providing valuable diagnostics for debugging and optimization.

The training process can be broken down into the following stages:

1.  **Configuration and Setup:** The script starts by parsing command-line arguments and loading a YAML configuration file. This configuration defines the model architecture, hyperparameters, data paths, and other settings.
2.  **Data Generation (Optional):** If the `--regenerate-data` flag is provided, the script generates a new synthetic dataset using `scripts.train.generate_synthetic_multi_wave.generate_dataset`.
3.  **Data Loading:** The script sets up data loaders for training, validation, and testing using the `setup_financial_forecasting_data` function from `data_provider.data_factory`.
4.  **Experiment Initialization:** A `GradientMonitoringExperiment` object is created. This is a specialized version of `Exp_Long_Term_Forecast` that wraps the optimizer to enable gradient tracking.
5.  **Training:** The `experiment.train()` method is called to start the main training loop. The model is trained for a specified number of epochs.
6.  **Evaluation:** After training, the `experiment.test()` method is called to evaluate the model's performance on the test set.
7.  **Reporting:** Finally, the script generates two JSON reports: one with gradient statistics and another with memory usage snapshots taken at various points during the execution.

## 2. Detailed Component Breakdown

### 2.1. Configuration (`_prepare_args`, `_load_config`)

-   **`parse_cli()`:** Defines and parses command-line arguments like `--config`, `--regenerate-data`, `--rows`, etc.
-   **`_load_config()`:** Loads the YAML configuration file specified by the `--config` argument.
-   **`_prepare_args()`:** Merges the command-line arguments and the configuration from the YAML file into a single `SimpleNamespace` object. It also sets default values for various parameters.

### 2.2. Data Handling

-   **`regenerate_dataset_if_needed()`:** If requested, this function calls `_generate_dataset` to create a new synthetic dataset as a CSV file.
-   **`setup_financial_forecasting_data()`:** This function (from `data_provider.data_factory`) is responsible for creating the PyTorch `DataLoader` instances for the training, validation, and test sets.

### 2.3. Experiment and Model (`GradientMonitoringExperiment`)

-   **`GradientMonitoringExperiment`:** This class inherits from `Exp_Long_Term_Forecast` and is the core of the training process.
-   **`_select_optimizer()`:** This method is overridden to wrap the optimizer's `step()` method. The wrapper, `tracked_step`, calls the `gradient_tracker.log_step()` method before executing the actual optimization step. This allows for capturing gradient information at every step.
-   **`Exp_Long_Term_Forecast`:** This base class (located in `exp/exp_long_term_forecasting.py`) is expected to contain the main training logic, including the model definition, loss function, and the training/validation/testing loops.

### 2.4. Diagnostics (`GradientTracker`, `MemoryDiagnostics`)

-   **`GradientTracker`:**
    -   Collects and stores gradient norms for each parameter and the global gradient norm at each training step.
    -   It uses `Welford's algorithm` for efficient online calculation of mean and variance of gradient norms.
    -   It also tracks RSS and CUDA memory usage at each step.
-   **`MemoryDiagnostics`:**
    -   Takes snapshots of memory usage at key points in the script (e.g., startup, after data loading, before training).
    -   It can dump the memory usage history to a JSON file for later analysis.

## 3. Potential Memory Issues and Bugs

### 3.1. Unbounded Memory Growth in `GradientTracker`

-   **Issue:** The `global_norms` and `vanishing_steps` lists in the `GradientTracker` class are appended to at every training step without any size limit. In long training runs with many steps, these lists can grow very large, consuming a significant amount of memory.
-   **Bug Type:** Memory Leak.
-   **Redundancy:** Storing the entire history of gradient norms might be redundant if only summary statistics or recent values are needed for analysis.

### 3.2. Inefficient Data Loading

-   **Issue:** The implementation of `setup_financial_forecasting_data` is not visible in this script. If it loads the entire dataset into memory, it can lead to very high memory consumption, especially for large datasets.
-   **Bug Type:** Inefficient Implementation.
-   **Redundancy:** Loading the entire dataset into memory is redundant if the data can be read in batches from the disk.

### 3.3. Potential Leaks in `Exp_Long_Term_Forecast`

-   **Issue:** The `Exp_Long_Term_Forecast` class is a likely place for common PyTorch memory leaks. These can include:
    -   Not detaching tensors from the computation graph when they are stored for logging (e.g., loss values).
    -   Accumulating history in tensors without `.detach()`.
    -   Keeping tensors on the GPU longer than necessary.
-   **Bug Type:** Memory Leak.

## 4. Proposed Fixes and Implementation Plan

### 4.1. Limit `GradientTracker` History

-   **Plan:** Modify the `GradientTracker` to limit the size of `global_norms` and `vanishing_steps` lists. This will be done by introducing a `max_history` parameter in the `__init__` method and using a `collections.deque` with a `maxlen` or by manually trimming the lists.
-   **Implementation:**
    1.  Add `max_history: int = 4096` to the `GradientTracker`'s `__init__`.
    2.  In the `log_step` method, after appending to `global_norms` and `vanishing_steps`, check if their length exceeds `max_history` and remove the oldest elements if necessary.

### 4.2. Investigate and Refactor Data Loading

-   **Plan:** Analyze the `data_provider/data_factory.py` file to understand the implementation of `setup_financial_forecasting_data`. If it loads the entire dataset into memory, refactor it to use a custom PyTorch `Dataset` that reads data from the disk on-the-fly.
-   **Implementation:**
    1.  Read and analyze `data_provider/data_factory.py`.
    2.  If necessary, create a new `Dataset` class that reads data from the CSV file in its `__getitem__` method.
    3.  Update `setup_financial_forecasting_data` to use this new `Dataset` class.

### 4.3. Review `Exp_Long_Term_Forecast`

-   **Plan:** Read and analyze the `exp/exp_long_term_forecasting.py` file to identify and fix any memory leaks in the training loop.
-   **Implementation:**
    1.  Read and analyze `exp/exp_long_term_forecasting.py`.
    2.  Look for tensors that are being stored without being detached from the computation graph. Use `.detach()` where appropriate.
    3.  Ensure that tensors are moved to the CPU (`.cpu()`) as soon as they are no longer needed on the GPU.
    4.  Add `torch.cuda.empty_cache()` at strategic points if memory fragmentation is suspected.
