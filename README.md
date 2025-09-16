This repository provides tools for automatic vehicle labeling in images using YOLOv5 for detection and fine-tuned VLMs (Qwen2.5-VL-3B or Gemma-3-4B) for attribute extraction. Supports fine-tuning, inference, similarity comparison between models, and evaluation against ground truth annotations.
Video guides can be found in this drive folder: https://drive.google.com/drive/folders/1jKFr_rrCbyCkJCMftWXr755QColJbYZy?usp=sharing

## 📋 1. Initial Setup & Preparation

### Environment Setup

1.  **Clone the Repository**: Start by getting the code on your local machine.
    ```bash
    git clone https://github.com/muqtasid87/vehicle-autolabeling.git
    cd vehicle-autolabeling
    ```

2.  **Install Dependencies**: The `requirements.txt` file contains all the necessary Python libraries. It's highly recommended to use a virtual environment.

    ```bash
    # Create and activate a virtual environment (optional but recommended)
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    #Install pytorch first
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    # Install the required packages
    pip install -r requirements.txt
    ```

3.  **Place YOLO Model**: Make sure the YOLOv5 model file, **`best_avc_v5.pt`**, is in the root directory of the project.

### Dataset Preparation

The project expects a specific folder structure for your data, as seen in the **`sample_dataset`** directory.

  * **/images**: This folder should contain all your image files (e.g., `.jpg`, `.png`).
  * **/annotations**: This folder should contain the corresponding JSON annotation files. Each JSON file should have the same name as its image and end with a _annotations suffix (e.g., `image1.jpg` corresponds to `image1_annotations.json`).

The JSON files are only needed for **fine-tuning** and **evaluation**. For running basic inference on new images, you only need the **/images** folder.

-----

## 🛠️ 2. Task: Fine-Tuning a VLM

This task trains a base model (Gemma or Qwen) on your custom dataset to improve its accuracy for vehicle labeling.

### How to Run

1.  **Configure the Script**: Open **`run_finetune.py`**. You can adjust the **`custom_hyperparams`** dictionary to control aspects of the training like the number of epochs, learning rate, and batch size.

    ```python
    # Example configuration in run_finetune.py
    custom_hyperparams = {
        "num_train_epochs": 3,
        "learning_rate": 2e-4,
        "per_device_train_batch_size": 4,
    }
    ```

2.  **Execute the Script**: Run the script from your terminal. It will prompt you to choose which model you want to fine-tune.

    ```bash
    python run_finetune.py
    ```

3.  **Follow the Prompt**:

      * Enter `qwen` or `gemma` when asked.
      * The script will then load your dataset from the **`sample_dataset`** folder, configure the model, and begin training. You'll see progress and metrics printed to the console. You can change the input_folder in the script to point to             your own dataset. 

### What You Get

  * The fine-tuned model components, known as LoRA adapters, will be saved in a new, timestamped directory inside the runs/ folder. The path will look something like this:

    ```runs/Qwen_Finetuning_18-21-17-09-2025/lora_model/```
    
    You will need this path for the next step.

-----

## 🚀 3. Task: Running Inference

This is the main task where the model analyzes new images and generates the vehicle attribute JSON files. You can run this with a base model or your fine-tuned model. 
**NOTE**: Batching is currently only supported on Gemma during inference. For Qwen inference, please set the batch_size to 1 to avoid errors.

### How to Run

1.  **Open the Script**: Open **`run_inference.py`**.

2.  **Choose Your Mode**: The file contains two example blocks.

      * **Option 1: Use a Fine-Tuned Model**. Uncomment this block. You **must** provide the path to your saved LoRA adapters from the fine-tuning step.

        ```python
        # Option 1: Run with LoRA adapters
        run_inference(
            model_name=model_name,
            input_folder='sample_dataset/images',
            # 👇 PASTE THE PATH FROM THE FINETUNING STEP HERE
            lora_adapter_path='runs/Qwen_Finetuning_04-05-18-08-2025/lora_model',
            use_lora=True
        )
        ```

      * **Option 2: Use a Base Model**. This block is enabled by default. It requires no fine-tuning and is great for getting a baseline result.

        ```python
        # Option 2: Run with base model only (no LoRA)
        run_inference(
            model_name=model_name,
            input_folder='sample_dataset/images',
            use_lora=False # This uses the base model
        )
        ```

3.  **Execute the Script**:

    ```bash
    python run_inference.py
    ```

4.  **Follow the Prompt**:

      * Enter `qwen` or `gemma` to select the model.
      * The script will then:
        1.  Load the images from your input folder.
        2.  Use YOLO to detect and crop vehicles.
        3.  Feed the crops to the VLM to generate attributes.
        4.  Save the results.

### What You Get

  * **JSON Outputs**: A new timestamped directory will be created (e.g., `runs/Qwen_Finetuned_Inference_[TIMESTAMP]/`). Inside it, you'll find a **`json_outputs`** subfolder containing one JSON file for each input image. These files detail all the vehicles found and their predicted attributes.

-----

## 📊 4. Task: Evaluating Model Performance

After running inference, you can use this script to score how well your model's predictions match the ground truth annotations.

### How to Run

1.  **Configure Paths**: Open **`run_evaluation.py`**. You need to update two variables:

      * **`GT_FOLDER`**: Set this to the path of your ground truth annotations folder (e.g., `sample_dataset/annotations`).
      * **`LLM_FOLDER`**: Set this to the path of the **`json_outputs`** folder that was generated during the inference step.

    <!-- end list -->

    ```python
    # Configuration in run_evaluation.py
    GT_FOLDER = r"sample_dataset/annotations" # Ground Truth
    LLM_FOLDER = r"runs/Qwen_Finetuned_Inference_11-31-16-09-2025/json_outputs" # Model Predictions
    ```

2.  **Execute the Script**:

    ```bash
    python run_evaluation.py
    ```

### What You Get

  * **Performance Report**: The script compares the ground truth to your model's predictions. It will save a detailed report (e.g., a `.csv` or `.xlsx` file) in the same folder as the LLM outputs. This report includes metrics like mAP (for detection accuracy) and field-by-field similarity scores for the JSON attributes.

-----

## 🎭 5. Task: Similarity Matching Between Qwen and Gemma Outputs
This is useful for comparing the results of two different models (e.g., base Gemma vs. fine-tuned Qwen) on the same dataset.

### How to Run

1.  **Run Inference Twice**: First, you need to run the inference task for both models you want to compare. This will give you two separate **`json_outputs`** folders.

2.  **Configure Paths**: Open **`run_similarity.py`**. Update the **`folder1`** and **`folder2`** variables to point to the two `json_outputs` directories you want to compare.

    ```python
    # Configuration in run_similarity.py
    folder1 = r"runs/Gemma_Base_Inference_16-35-07-09-2025/json_outputs"
    folder2 = r"runs/Qwen_Finetuned_Inference_11-31-16-09-2025/json_outputs"
    ```

3.  **Execute the Script**:

    ```bash
    python run_similarity.py
    ```

### What You Get

  * **Comparison File**: The script will generate and save a comparison report, typically as an Excel or CSV file. This file provides a detailed, side-by-side look at the attribute predictions from both models for each vehicle, highlighting where they agree and disagree.
