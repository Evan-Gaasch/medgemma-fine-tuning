# ImageTextToTextTuner:
> Here is documentation for the tuner abstraction/API that speeds up the tuning process in a user-friendly way.

## Overview:
This documentation aims to serve as a guide for finetuning MedGemma, an Image-Text-To-Text model, on a medical dataset (in this case, for colonoscopy) using the ImageTextToTextTuner abstraction.

To follow this documentation, a GPU with at least 16 GB of VRAM (such as an RTX 5080 or RTX 4080). Several GPUs or an A100/H100 are preferred, though this documentation is optimized for tuning on a single RTX 5080. Ensure you have enough hardware space for the dataset and model, which can exceed 20 GB depending on the dataset.

To follow this documentation, basic knowledge of Python and Linux commands are required.

## What is this abstraction?
This ImageTextToTextTuner is a Python package designed to speed up the QLoRA fine tuning process for ImageTextToText models. For a detailed view of how this abstraction works, see the TUNING_DOCUMENTATION.md file.

This package should allow for the training of most ImageTextToText models on any dataset, though the package was originally designed for MedGemma (4b-it), a Gemma 3 variant. 
Examples of models this package may work on are as follows:
- LLama 3.2 vision
- Gemma 3
- etc.

See the troubleshooting section for examples of settings you may have to change to ensure the model works. However, this abstraction is primarily oriented around MedGemma.

## Installation:
The simplest way to install the library is to move the image_text_to_text_tuner directory into the folder in which you run the script using the `scp` command. Then, reference it with an import statement. In the future, this package may be packaged with PyPI, automating the whole installation process. However, this has not been completed yet.

After setting up your device, run python3 -m venv <NAME> (on Linux) to set up a virtual environment to better manage dependencies.

Then, to enter the environment, run the command source <FILE/PATH/TO/ACTIVATE. This is generally  source <NAME>/bin/activate. If the second command does not work, you can navigate the file system through Linux Terminal to find the activate file and replace the file path with that. 

To install dependencies, `cd` into the image_text_to_text_tuner directory and run `pip install -r requirements.txt`. 
> Checkpoint: At this point, use the command `pip show transformers` when inside the virtual environment to ensure that the requirements were properly installed.

If you wish to export to GGUF format, you must first make another virtual environment. That is, `python3 -m venv gguf_conversion` (or any other name you wish to call your venv)

After activating the venv with `source gguf_conversion/bin/activate`, we must clone the llama.cpp Github repository.

llama.cpp is a Github repository that contains the code necessary to convert a HuggingFace safetensors model to GGUF format. It can be installed with `git clone https://github.com/ggml-org/llama.cpp`. Then, install the dependencies for the conversion by using `pip install -r llama.cpp/requirements.txt`. These dependencies conflict with the dependencies required for the rest of the setup, so make sure this is in a virtual environment.

## Quick Start:
After installing, run the following code to quickly start the tuning process. For more detail about the attributes that can be changed to yield optimal tuning behavior, see later sections.

```
import ImageTextToTextTuner

config = ImageTextToTextTuner.tuner.ScriptConfig()
config.dataset_name = "flaviagiammarino/vqa-rad" #Radiology Dataset (NOTE: MedGemma was trained on this dataset)
config.epochs = 0.5 #Or any other setting change

tuner = ImageTextToTextTuner.tuner.fine_tune(config)
tuner.tune()

#To Be Implemented: (Below){
eval_config = ImageTextToTextTuner.evaluator.EvalConfig()
evaluator = ImageTextToTextTuner.evaluator.evaluate_model(eval_config)
evaluator.evaluate()
```

After adjusting any attributes and runing the script, tuning will complete and a model will be saved to a file.

Note: this may take a significant amount of time depending on hardware; an RTX 5080 runs roughly 2.5 epochs a day on a 50,000 example training dataset, or roughly 125,000 examples a day.

## Tuning:
To use the package, import image_text_to_text_tuner in your Python script. Then, initialize the trainer configuration with the following syntax: `image_text_to_text_tuner.tuner.ScriptConfig()`. This initializes a dataclass that informs all of the training behavior. 

The default behavior is training MedGemma on the Kvasir-VQA dataset, but if this is not the model and/or dataset you intend to train on, change the attribute using the following syntax:
`config.dataset_name = "flaviagiammarino/vqa-rad"`
However, you must note that this dataset has already been trained on by the base model, so it is next-to-useless but serves as a good example of the syntax.

To begin tuning, pass the configuration to the ``` image_text_to_text_tuner.tuner.fine_tune``` class. 

After the trainer is initialized, the only thing that remains is tuning the model. The tuner abstraction makes this easy, possible by calling only
```tuner.tune()```.

Examples of the configuration parameters you likely want to change are as follows:
### ScriptConfig dataclass attributes:

Attributes:
```
class ScriptConfig:
    """Stores all hyperparameters and configuration settings for the script."""
    # Model and Tokenizer
    model_id: str = "google/medgemma-4b-it"
    hf_token: Optional[str] = "" # Loaded from .env file, or you can hard-code here

    # Dataset
    dataset_name: str = "SimulaMet-HOST/Kvasir-VQA"
    dataset_split: str = "raw" #Useful for only loading a subset of a dataset
    eval_split_name: str = "test"
    train_split_name: str = "train" #Leave as default if your dataset (like Kvasir-VQA) has no train/test split
    test_size: float = 0.1
    image_column: str = "image"
    question_column: str = "question"
    answer_column: str = "answer"
    num_workers: int = 2*(os.cpu_count()) #More workers speeds up mapping but takes more memory
    shuffle: bool = False #Set to false if the dataset doesn't have a default split to prevent later evaluation data contamination
    load_data_from_file: bool = True #Used to force remap (set to False) if you make a dataset change


    #**These may be model/processor specific so make sure to update them, current ones are for MedGemma**
    image_end_token: int = 260000 #the tokenized tensor value for this token
    image_token: int = 262144 #the "fill" image token for all image tokens except the start and end
    image_start_token: int = 25999 


    # Image Preprocessing
    image_resize_size: tuple = (896, 896) # Upscale images for more details


    # QLoRA & PEFT Configuration (all values from hyperparameter tuning for MedGemma)
    quantization: bool = True #Should the model be loaded in quantized form? Set this to True if you have memory constraints
    lora_r: int = 16 # Size factor of LoRA matrices
    lora_alpha: int = 32 #128 # "Strength" factor of LoRA matrices
    lora_dropout: float = 0.0 # Used for regularization
    lora_target_modules: List[str] = field(default_factory=lambda: ["gate_proj", "up_proj", "down_proj"]) #The layers/modules you want the model to act upon (generally, don't change this)
    load_in_4bit: bool = True #Whether to load in 4 bit, which yields a ~4x VRAM usage decrease in practice compared to 32 bit
    bnb_4bit_quant_type: str = "nf4" #Most efficient quantization type
    use_double_quant: bool = True

    # Training Arguments
    output_dir: str = "./medgemma-finetuned" #where the model should be saved
    epochs: float = 5.0 #How many passes over the training data the model takes (feel free to do extra and ctrl-c when you see overfitting, there is a try-finally statement at the end
    train_batch_size: int = 4 # Manually tune this to make maximum use of available memory
    eval_batch_size: int = 2
    gradient_accumulation_steps: int = 2
    learning_rate: float = 7.574325906695311e-5 #From hyperparameter tuning
    max_grad_norm: float = 1.0
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    optim: str = "paged_adamw_8bit" 
    fp16: bool = False
    bf16: bool = True
    gradient_checkpointing: bool = True
    resume_from_checkpoint: str = "path/to/checkpoint" #Replace this with a path to checkpoint or None if not loading from a checkpoint
    
    # MedGemma's typical context length is often 2048 or 4096. 
    max_sequence_length: int = 4096 

    #Hyperparameter Tuning Arguments:
    hyperparameter_sampler = optuna.samplers.TPESampler() #GP (Gaussian Process) and CMA-ES samplers also another option depending on hyperparameter space
    tune_hyperparameters: bool = False #**Make this true if you are using any model other than MedGemma on any dataset other than Kvasir-VQA**
    hyperparameter_epochs: float = 0.05 #balance of speed and 
    hyperparameter_trials: int = 40 
    hyperparameter_pruner = optuna.pruners.HyperbandPruner() # Pruner to stop unpromising trials early

    # Evaluation and Logging
    eval_strategy: str = "steps"
    eval_steps: float = 1000 #How often the model is fully evaluated
    logging_steps: int = 500
    push_to_hub: bool = False
    load_best_model_at_end: bool = True 
    torch_empty_cache_steps: int = 50 
    save_strategy: str = "steps"
    save_steps: float = 1000

    print_tokenized_output:bool = False

```
You can manually adjust any of these attributes to your desired values by creating an instance of the class, as shown in the previous section.
## Evaluation:

After training, the model is saved into the "output_dir" directory and model status is saved. If you see underfitting, however, it is wise to repeat the training but instead load from the last checkpoint. For a detailed analysis of underfitting, see the tuning documentation.

Like the tuner, an evaluation config must be set up. The syntax is almost identical to before. That is, 
```
eval_config = image_text_to_text_tuner.evaluator.EvalConfig()
```
Attributes are modified, as seen in the tuner step, and then the evaluator is created with an instance of the configuration.

To create the evaluator, use the following basic syntax: ``` evaluator = image_text_to_text_tuner.evaluator.evaluate_model(eval_config) ```

By calling `evaluator.evaluate()`, evaluation begins and a graph is produced.
Evaluation can often be slow, and mapping is required before beginning because of the different format required by the evaluation loop. This is slow, so patience is required, but no other commands or syntax is required to complete this step.

The only way to affect the behavior of the evaluation class is through the configuration. All configuration attributes are shown below:
### EvalConfig dataclass attributes:
```
class EvalConfig:
    """Stores all hyperparameters and configuration settings for the script."""
    # Model and Tokenizer
    model_id: str = "google/medgemma-4b-it"
    hf_token: Optional[str] = "" # Loaded from .env file, or replace with your token
    output_dir: str = "./medgemma-finetuned" #Where the output information is to be saved

    # Dataset
    dataset_name: str = "SimulaMet-HOST/Kvasir-VQA"
    dataset_split: str = "raw" #Useful for only loading a subset of a dataset
    eval_split_name: str = "test"
    train_split_name: str = "train" #Leave as default if your dataset (like Kvasir-VQA) has no train/test split
    test_size: float = 0.1
    image_column: str = "image"
    question_column: str = "question"
    answer_column: str = "answer"
    num_workers: int = 3*(os.cpu_count()) #Speeds up mapping
    shuffle: bool = False #Keep it false if your dataset does not already have a train/test split.
    load_data_from_file: bool = True

    # Image Preprocessing
    image_resize_size: tuple = (448, 448) # Upscale images for more details

    # Model Configuration
    quantization: bool = True
    medgemma_tuned_path: str = "./medgemma-finetuned" #Filepath to the saved model
    fp16: bool = False
    bf16: bool = True
    load_in_4bit: bool = True 
    bnb_4bit_quant_type: str = "nf4"
    use_double_quant: bool = True
    torch_empty_cache_steps: int = 50
    
    # MedGemma's typical context length is often 2048 or 4096. 
    max_sequence_length: int = 4096 

    #Prompting
    system_prompt: str = ( #Change this if you change the task, this is task specific
        """You are a leading medical expert in the domain of colonoscopy and analyze medical imagery with high accuracy,
      identifying all anomalies and answering questions factually, and concisely. Note that 'finding' is generally code for abnormality.
      Respond in a concise manner; for example, when given an image of ulcerative colitis and a prompt "Are there any 
      abnormalities in the image?", respond with "Ulcerative Colitis" """)
    eval_adaptive_prompt: bool = True
    eval_static_prompt: bool = True
    eval_base: bool = True
  ```

## GGUF conversion:
To use GGUF conversion, several steps are required.
### Creating a venv
To export your tuned model in GGUF format, you must first make a virtual environment. That is,
`python3 -m venv gguf_conversion` (or any other name you wish to call your venv)

After activating the venv with `source gguf_conversion/bin/activate`, we must clone the llama.cpp Github repository.
### llama.cpp
llama.cpp is a Github repository that contains the code necessary to convert a HuggingFace safetensors model to GGUF format. It can be installed with `git clone https://github.com/ggml-org/llama.cpp`. Then, install the dependencies for the conversion by using `pip install -r llama.cpp/requirements.txt`. These dependencies conflict with the dependencies required for the rest of the setup, so make sure this is in a virtual environment.

### Conversion
The simplest way to convert the model is to use the gguf_converter abstraction. That is,
```
gguf_config = gguf_conversion.ConverterConfig()
gguf_config.model_name = "test_model"

converter = gguf_conversion.converter(gguf_config)
``` 
Other attributes include:
```
class ConverterConfig:
    """Stores all hyperparameters and configuration settings for the script."""
    # Model and Tokenizer
    model_id: str = "google/medgemma-4b-it"
    model_name: str = "medgemma_tuned"
    hf_token: Optional[str] = None # Loaded from .env file

    # Dataset
    output_dir: str = "./medgemma-finetuned/merged_model"
    processor_dir: str = "./medgemma-finetuned/merged_model"
    adapter_path: str = "./medgemma-finetuned/backup_adapter"
    venv_python_path: str = "./gguf_conversion/lib/python3.11"

```
Make sure you update the venv_python_path if you have a virtual environment of a different name.


## Troubleshooting:
### New models:
There may be several issues with this library on models other than MedGemma. To begin, many Image Text To Text models have different layers, so `lora_target_modules` must be updated. By setting `print_model` to true, you can get the structure of the model and choose which layers to apply the LoRA fine tuning to. That is, `config.print_model = True` Find the layer names in the model print statement and record the names you want to apply the adapter to. Then, set `config.lora_target_modules = ["gate_proj", "up_proj", "down_proj"]`, and setup is complete.

Also, it is imperative to change the model's processor tokens when switching models, which is a minor inconvenience. This can be circumvented by printing the raw output during inference.

### New datasets
Finally, For different datasets ensure that when checking on the HuggingFace dataset page that you update the train/test split names based on how the dataset is split.

These are all fairly tedious processes but they could be automated in the future.
