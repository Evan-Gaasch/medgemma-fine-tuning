# -*- coding: utf-8 -*-
"""
Optimized Fine-Tuning Script for MedGemma on a Memory-Constrained GPU.

This script is designed to fine-tune the 'google/medgemma-4b-it' model using QLoRA
on a GPU with 16GB VRAM. It includes optimizations such as 4-bit
quantization, gradient checkpointing, paged optimizers, and efficient data
processing to make training feasible.

To use, create a .env file in the same directory with your Hugging Face token:
HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxx"
"""

#imports (refine tech stack to optimize)
import torch,optuna
import gc, os, io, PIL, math #base libraries needed
import numpy as np
from huggingface_hub import login,hf_hub_download
from dataclasses import dataclass, field
from datasets import load_dataset, Image, Dataset,load_from_disk
from typing import Dict, List, Optional
from peft import LoraConfig, get_peft_model,prepare_model_for_kbit_training,PeftModel
from dotenv import load_dotenv
from torch.utils.tensorboard import SummaryWriter
from safetensors.torch import save_model
import shutil,subprocess
from transformers import(
   AutoProcessor,
   AutoModelForImageTextToText,
   BitsAndBytesConfig
)

#import evaluate

gc.collect()
#print(torch.cuda.memory_allocated(0),torch.cuda.mem_get_info(device=0)) #initial memory usage
#torch.cuda.set_per_process_memory_fraction(0.9)


#configuration:
# --- Configuration Section ---
#To-Do: Load config to a file
print(f"--- Initializing --- \nUpdating environment variables to prevent memory fragmentation")
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True' #Ensures memory is not extremely fragmented

@dataclass
class ScriptConfig:
    """Stores all hyperparameters and configuration settings for the script."""
    # Model and Tokenizer
    model_id: str = "google/medgemma-4b-it"
    hf_token: Optional[str] = None # Loaded from .env file

    # Dataset
    dataset_name: str = "SimulaMet-HOST/Kvasir-VQA"
    dataset_split: str = "raw" #Useful for only loading a subset of a dataset
    eval_split_name: str = "test"
    train_split_name: str = "train" #Leave as default if your dataset (like Kvasir-VQA) has no train/test split
    test_size: float = 0.1
    image_column: str = "image"
    question_column: str = "question"
    answer_column: str = "answer"
    num_workers: int = (os.cpu_count()) #or set to a small int (4x GPU count?), os.cpu_count() may take too much memory
    shuffle: bool = False
    load_data_from_file: bool = True #Used to force remap (set to False) if you make a dataset change
     
    image_end_token: int = 256000 #the tokenized tensor value for this token
    image_token: int = 262144 #the "fill" image token for all image tokens except the start and end
    image_start_token: int = 25999 

    # Image Preprocessing
    image_resize_size: tuple = (896, 896) # Upscale images for more details

    # QLoRA & PEFT Configuration
    quantization: bool = True
    lora_r: int = 16 # From hyperparameter tuning
    lora_alpha: int = 32 #128 # From hyperparameter tuning
    lora_dropout: float = 0.0 #From hyperparameter tuning
    lora_target_modules: List[str] = field(default_factory=lambda: ["gate_proj", "up_proj", "down_proj"]) # REMOVED "lm_head" for exportability

    # Training Arguments
    output_dir: str = "./medgemma-finetuned/training_v2"
    adapter_path: str = "./medgemma-finetuned/backup_adapter/"
    resume_from_checkpoint: bool = False
    use_pretrained_adapter: bool = True
    epochs: float = 5.0
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
    load_in_4bit: bool = True
    
    # MedGemma's typical context length is often 2048 or 4096. 
    max_sequence_length: int = 4096 

    #Hyperparameter Tuning Arguments:
    hyperparameter_sampler = optuna.samplers.TPESampler() #GP (Gaussian Process) and CMA-ES samplers also another option depending on hyperparameter space
    tune_hyperparameters: bool = False
    hyperparameter_epochs: float = 0.75
    hyperparameter_trials: int = 20 
    hyperparameter_pruner = optuna.pruners.HyperbandPruner() # Pruner to stop unpromising trials early

    # Evaluation and Logging
    eval_strategy: str = "steps"
    eval_steps: float = 1000 #Dramatically slows down fine tuning if too low
    logging_steps: int = 500
    push_to_hub: bool = False
    load_best_model_at_end: bool = True 
    torch_empty_cache_steps: int = 50 
    save_strategy: str = "steps"
    save_steps: float = 1000


def get_memory_usage():
    #Returns the current GPU memory usage in GB
    allocated = torch.cuda.memory_allocated(0) / (1024**3)
    total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    free_memory = (torch.cuda.mem_get_info(device=0)[0])/ (1024**3) #includes other processes

    utilized = total-free_memory
    return (f"Utilized: {utilized:.2f} GB / {total:.2f} GB \nAllocated by PyTorch: {allocated:.2f} GB / {total:.2f}")


def main():
    """Main function to run the fine-tuning process."""
    load_dotenv()

    #Set up TensorBoard
    writer = SummaryWriter()

    config = ScriptConfig()
    config.hf_token = os.getenv("HF_TOKEN")
    if not config.hf_token:
        print("Warning: HF_TOKEN environment variable not found. Using hardcoded token (ensure it's replaced in production).")
        config.hf_token = "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" # Placeholder, replace with your actual token or remove

    print("--- Initial Configuration ---")
    print(f"Model ID: {config.model_id}")
    print(f"Dataset: {config.dataset_name}")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Initial Memory: {get_memory_usage()}")

    # --- 1. Authenticate and Load Processor ---
    if not config.hf_token:
        print("Warning: Hugging Face token not found. Proceeding without authentication. This may fail for private models/datasets.")
    else:
        print("Hugging Face token found. Logging in.")
        login(token=config.hf_token)

    processor = AutoProcessor.from_pretrained(config.model_id, use_fast=True)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    # --- 2. Load and Prepare Dataset ---
    print("\n--- Loading and Preparing Dataset ---")
    dataset = load_dataset(config.dataset_name, split=config.dataset_split)

    def format_and_resize_example(example: Dict) -> Dict:
        #Set up flag for removal
        example["remove"] = False
        #Set up blank messages in case of error
        example["messages"] = [
            {
                "role": "user",
                "content": [
                    {"type": "text"},
                    {"type": "image"}, 
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text"}
                ]
            }
        ] #This blank filler helps prevent dataset errors because the column still exists and has the same datatype

        image = example[config.image_column]
        if not isinstance(image, PIL.Image.Image):
            try:
                if isinstance(image, dict) and "bytes" in image:
                    image = PIL.Image.open(io.BytesIO(image["bytes"])).convert("RGB") 
                else:
                    # Attempt to convert other formats to PIL Image
                    image = PIL.Image.fromarray(np.array(image)).convert("RGB") 
            except Exception as e:
                print(f"Warning: Could not process image of type {type(image)}. Skipping example. Error: {e}")
                print(f"Details: {e}")
                example["remove"] = True
                return example

        # Ensure image is RGB before resizing
        image = image.convert("RGB").resize(config.image_resize_size, resample=PIL.Image.Resampling.LANCZOS)
        example[config.image_column] = image 

        prompt = example[config.question_column]

        answer = example[config.answer_column]#for the last few examples in the Kvasir VQA dataset the 'answer' is the string "nan" but not actually nan, check for this?
        if answer == "nan": #Handles the "nan" case but not real nan
             print("Answer is Not a Number")
             example["remove"] = True
             return example

        example["messages"] = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image"}, 
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": answer}
                ]
            }
        ]
        return example

    print("Mapping processing function to datasets (only resizing and message creation)...")
    original_features_to_remove = [col for col in [config.question_column, config.answer_column, "source", "img_id"] if col in dataset.features] #list comprehension to remove unnecessary columns

    #Load from dataset if the map is already completed:
    if os.path.exists("./cached_dataset") and config.load_data_from_file:
        # Load from disk the file it exists for subsequent runs
        dataset = load_from_disk("./cached_dataset")
        print("Dataset loaded from disk")
    else:
        print("Existing, formatted dataset instance not found. Mapping dataset now.")    
        dataset = dataset.map(
            format_and_resize_example,
            num_proc=config.num_workers, #Use more than 1 worker for faster mapping, make sure memory is not overused
            remove_columns=original_features_to_remove
        )
        #Remove all rows for which remove = True
        dataset = dataset.filter(lambda example: example["remove"] is not True) #Only remove = False should remain

    print("Map completed. Dataset structure after initial processing:")
    print(dataset)
    print(f"Memory after initial data processing: {get_memory_usage()}")

    if not os.path.exists("./cached_dataset"): #If not already saved, the dataset is saved here
        dataset.save_to_disk("./cached_dataset")

   # global eval_dataset #Involved in hyperparameter tuning, needs to be global, unfortunately. I will refine logic to prevent this.

    if config.train_split_name in dataset.features and config.eval_split_name in dataset.features: #Checks if dataset is already partitioned
        train_dataset = dataset[config.train_split_name]
        eval_dataset = dataset[config.eval_split_name]
        print("Dataset is already partitioned, skipping this step.")  
    else:
        dataset = dataset.train_test_split(test_size=config.test_size,shuffle = config.shuffle) #The train_test_split returns a partitioned dataset, with dictionary keys named "train" and "test"
        train_dataset = dataset["train"]
        eval_dataset = dataset["test"]
        print("Dataset partitioned into Train and Test splits.")

    # --- 3. Configure Model (Quantization & PEFT) ---
    print("\n--- Loading and Configuring Model ---")

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=config.load_in_4bit, #reduces memory usage significantly
        bnb_4bit_quant_type="nf4",#more efficient version of 4 bit quantization, "normalized 4 bit floating point"
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True, #double quantization, in this case saves up to ~0.25 GB of VRAM (~0.5 bits per parameter)
    ) if config.quantization else None

    model_torch_dtype = torch.float32 #default, "else" case

    if config.bf16 and config.fp16: 
        print("torch_dtype cannot be both bfloat16 and float 16, please adjust ScriptConfig values. torch_dtype has defaulted to bf16.")

    if config.bf16:
        model_torch_dtype = torch.bfloat16
    elif config.fp16:
        model_torch_dtype = torch.float16

    def modelInit(lora_r,lora_alpha,lora_dropout):
        # Clear any leaked models from previous trials
        torch.cuda.empty_cache()
        gc.collect()

        model = AutoModelForImageTextToText.from_pretrained(
            config.model_id,
            quantization_config=quantization_config,
            torch_dtype=model_torch_dtype,
            device_map="auto",
            trust_remote_code=True,
        )

        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=config.lora_target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )

        #Finish full model init, with PEFT setup (Hyperparameter tuning requires this to be self-contained)
        if config.quantization:
            model = prepare_model_for_kbit_training(model) #formats the quantized model properly for training regardless of gradient checkpoint status

        if config.gradient_checkpointing:
            model.config.use_cache = False # True is incompatible with gradient checkpointing

        if not config.use_pretrained_adapter:
            model = get_peft_model(model, lora_config)
        else:
            model = PeftModel.from_pretrained(model, config.adapter_path,is_trainable=True)
            print("loaded from existing")
        model.print_trainable_parameters()
    
        torch.cuda.empty_cache() #clear cache after deletion to minimize memory usage+fragmentation
        
        return model

    print(f"Memory after PEFT setup: {get_memory_usage()}")

    # --- 4. Define Data Collator ---
    class CustomDataCollatorForVision2Seq:
        # Pass the max_sequence_length from config
        def __init__(self, processor, max_sequence_length: int): 
                self.processor = processor
                self.max_length = max_sequence_length # Use the fixed max_length from config


        def __call__(self, examples: List[Dict]) -> Dict[str, torch.Tensor]:
            texts = [] #not just prompts, includes prompt+answer
            images = [] # This will collect PIL Images
            for example in examples:
                new_text = self.processor.apply_chat_template(
                    example["messages"],
                    add_generation_prompt=False,
                    tokenize=False,
                )
                texts.append(new_text)
                images.append(example[config.image_column]) 

            # Wrap each PIL Image in its own list, as the processor expects a list of lists for images
            # where each inner list corresponds to the images for a single text input in the batch.
            images_for_batch_processing = [[img] for img in images] 

            batch_inputs = self.processor(
                text=texts,
                images=images_for_batch_processing, # Pass the correctly formatted list of lists
                return_tensors="pt",
                padding="longest",
                truncation=True, # Ensure truncation is enabled
                max_length=self.max_length # Explicitly set max_length from config
            )

            #Labels setup for causal LM
            labels = torch.full_like(batch_inputs["input_ids"], -100) #sets all labels to -100, a value not read by the model

            for i, example in enumerate(examples):
                # Apply chat template specifically for the prompt part to determine its length
                # This ensures we mark only the assistant's response as targets for loss.
                prompt_messages = [example["messages"][0]] # Only the user's turn (prompt part)
                prompt_text = self.processor.tokenizer.apply_chat_template(
                    prompt_messages, tokenize=False, add_generation_prompt=False
                )
                
                # Tokenize the prompt text to get its tokenized length
                prompt_tokens = self.processor.tokenizer(
                    prompt_text, 
                    return_tensors="pt", 
                    padding=False, 
                    add_special_tokens=True # Ensure special tokens are included in length calculation
                )["input_ids"]
                
                # The prompt_length should be the length of the tokens for the prompt,
                # including any special tokens added by apply_chat_template or tokenizer
                prompt_length = prompt_tokens.shape[1]
                
                # Determine the actual length of the tokenized sequence in the batch_inputs
                # This accounts for padding/truncation applied by the processor
                actual_len = (batch_inputs["attention_mask"][i]).sum().item()
                
                # Set labels for the assistant's response part (answer)
                # Ensure the slice doesn't go out of bounds if actual_len < prompt_length, though it shouldn't
                start_index = min(prompt_length, actual_len) # Prevent negative or invalid slice start
                labels[i, start_index:actual_len] = batch_inputs["input_ids"][i, start_index:actual_len] #Reveals the parts that are not prompt
                
                batch_inputs["labels"] = labels

                #Mask the image tokens (262144,260000, 25999)
                batch_inputs["labels"][batch_inputs["labels"] == config.image_token] = -100
                batch_inputs["labels"][batch_inputs["labels"] == config.image_end_token] = -100
                batch_inputs["labels"][batch_inputs["labels"] == config.image_start_token] = -100

                #Mask padding tokens (id = 0)
                batch_inputs["labels"][batch_inputs["labels"] == 0] = -100

            #replace answer in input_ids using the same method:
            
            gc.collect()
            return batch_inputs

    # Initialize data collator with the explicit max_sequence_length
    data_collator = CustomDataCollatorForVision2Seq(processor, config.max_sequence_length)

    # --- 5. Set Up Trainer ---
    print("\n--- Setting up Trainer ---")
    from trl import SFTConfig,SFTTrainer
    
    def trainerSetup(trial): #All of this is in a function for hyperparameter tuning later
        print(f"Memory before trainerSetup: {get_memory_usage()}")
        if trial: #Adjusts desired hyperparameters
            #Tuning parameters
            learning_rate= trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True)
            gradient_accumulation_steps= trial.suggest_categorical("gradient_accumulation_steps", [1,2,4,8,16]) 
            
            #LoRA parameters:
            lora_r_val = trial.suggest_categorical("lora_r", [16, 32]) #r of 64 crashes
            lora_alpha_val = trial.suggest_categorical("lora_alpha", [16, 32, 64, 128])
            lora_dropout_val = trial.suggest_float("lora_dropout", 0.0, 0.1, step=0.01)

            #Publish hyperparameters to TensorBoard:
            writer.add_scalar('Learning Rate',learning_rate,trial.number)
            writer.add_scalar('Gradient Accumulation Steps',gradient_accumulation_steps,trial.number)
            writer.add_scalar('LoRA R',lora_r_val,trial.number)
            writer.add_scalar('LoRA Alpha',lora_alpha_val,trial.number)
            writer.add_scalar('LoRA Dropout',lora_dropout_val,trial.number)
            

            num_train_epochs = config.hyperparameter_epochs #Decreases tuning time by decreasing epochs for tuning

            #shuffle and slice the eval dataset to speed up tuning
            local_eval_dataset = eval_dataset.shuffle() #ensures samples are different each trial
            local_eval_dataset = local_eval_dataset.select(list(range(0,100))) #first 99 samples for training speed, just to speed up hyperparameter tuning
            #Name changed here to avoid global changes 
            

        else: #Use default hyperparameters
            learning_rate = config.learning_rate
            gradient_accumulation_steps = config.gradient_accumulation_steps
            num_train_epochs = config.epochs
            
            lora_r_val = config.lora_r
            lora_alpha_val = config.lora_alpha
            lora_dropout_val = config.lora_dropout
            local_eval_dataset = eval_dataset #The variable name change is more for clarity since this is already in a subroutine

        model = modelInit(lora_alpha = lora_alpha_val, lora_r = lora_r_val,lora_dropout = lora_dropout_val) #reinitializes each hyperparameter run

        training_args = SFTConfig(
            output_dir=config.output_dir,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=config.train_batch_size,
            per_device_eval_batch_size=config.eval_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            gradient_checkpointing=config.gradient_checkpointing,
            
            learning_rate=learning_rate,
            lr_scheduler_type=config.lr_scheduler_type,
            warmup_ratio=config.warmup_ratio,
            max_grad_norm=config.max_grad_norm,
            
            logging_steps=config.logging_steps,
            eval_strategy=config.eval_strategy, 
            eval_steps=config.eval_steps,
            save_strategy=config.save_strategy, 
            save_steps = config.save_steps,
            save_total_limit=2,
            load_best_model_at_end=config.load_best_model_at_end,
            
            fp16=config.fp16,
            bf16=config.bf16,
            
            report_to="tensorboard", 
            push_to_hub=config.push_to_hub,
            remove_unused_columns=False,
            torch_empty_cache_steps=config.torch_empty_cache_steps,
            
            optim = config.optim
        )

        trainer = SFTTrainer(
            model = model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=local_eval_dataset, #Refine code
            data_collator=data_collator,
        )
        print("Model and trainer are set up")
        print(f"Memory after trainer setup: {get_memory_usage()}")

        del local_eval_dataset #Prevents memory leak of ~0.14 GB/Hyperparameter Trial
        #Set up more TensorBoard logging if necessary:
        return model, trainer

    # --- 6. Hyperparameter Tuning ---
    print("\n--- Starting Hyperparameter Tuning ---")
    
    if config.tune_hyperparameters:
        def objective(trial):
            try:
                model, trainer = trainerSetup(trial = trial)
                trainer.train()

                eval_loss = trainer.evaluate()["eval_loss"]
                print(f"Eval_loss {eval_loss}")

                if trial.should_prune():
                    print(f"Trial {trial.number} pruned.")
                    raise optuna.TrialPruned()
                
                writer.add_scalar('Loss/tune',eval_loss,trial.number)

            finally:
                writer.flush()
                #Synchronize GPU and CPU so trainer is actually deleted
                torch.cuda.synchronize()
                print(f"Memory before model del: {get_memory_usage()}")
                if trainer:
                    #Delete components individually to prevent circular loading and persistent memory/memory leaks
                    if hasattr(trainer, 'model'):
                        trainer.model = None
                    if hasattr(trainer, 'optimizer'):
                        trainer.optimizer = None
                    if hasattr(trainer, 'lr_scheduler'):
                        trainer.lr_scheduler = None
                    if hasattr(trainer, 'eval_dataset'):
                        del trainer.eval_dataset #Clean up potential past dataset slices (Likely cause of past memory leak of 0.14 GB each hyperparameter trial)
    

                    del trainer
                    print(f"Trainer deleted during trial {trial.number}")
                
                
                if model:
                    model.to('cpu') #Good practice to offload the model and then delete it
                    del model #Having a distinction prevents potential circular reference errors that mean memory is not cleaned up
                    print(f"Model deleted during trial {trial.number}")

                torch.cuda.synchronize() #Helps with cleanup ocasionally
                torch.cuda.empty_cache() #clear cache after deletion to minimize memory usage+fragmentation
                print(f"Memory after model del : {get_memory_usage()}")
                gc.collect()
            return eval_loss
        
        study = optuna.create_study(sampler=config.hyperparameter_sampler,direction = "minimize",pruner = config.hyperparameter_pruner,study_name="medgemma_finetuning_hp_tuning",load_if_exists=True)
        study.optimize(objective, n_trials=config.hyperparameter_trials,show_progress_bar = True)

        print(f"Best Trial: {study.best_trial.value}")
        print(f"Best parameters {study.best_params}")
        best_params = study.best_params

        #Update configuration
        config.learning_rate = best_params.get("learning_rate", config.learning_rate)
        config.gradient_accumulation_steps = best_params.get("gradient_accumulation_steps", config.gradient_accumulation_steps)

        config.lora_r = best_params.get("lora_r", config.lora_r)
        config.lora_alpha = best_params.get("lora_alpha", config.lora_alpha)
        config.lora_dropout = best_params.get("lora_dropout", config.lora_dropout)

        #Clear memory
        gc.collect()
        torch.cuda.empty_cache()

    else:
        print("Skipping hyperparameter tuning")
    
    model,trainer = trainerSetup(trial = None) #After hyperparameter tuning, initialize normal trainer

    # --- 7. Start Training ---
    print("\n--- Starting Fine-Tuning ---")
    gc.collect()
    torch.cuda.empty_cache()
    print(f"Memory before training: {get_memory_usage()}")

    try: #This allows early stopping by the user where the model will still be saved
     trainer.train(resume_from_checkpoint = config.resume_from_checkpoint)

    # --- 8. Save Final Model and Metrics ---
    finally:
        print("\n--- Training Complete ---") #(Or keyboard interrupt)
        trainer.save_model()
        trainer.save_state()    
        print(f"Final model adapter saved to {config.output_dir}")
        print(f"Memory after training: {get_memory_usage()}")

        print("Model loaded")
        #model.save_pretrained(config.output_dir)
        save_model(model, config.output_dir)
        print("Model Saved")
        processor.save_pretrained(config.processor_dir)
        print("Model saved")

        #Save tokenizer.model 
        cached_path = hf_hub_download("google/medgemma-4b-it","tokenizer.model")
        cached_config = hf_hub_download("google/medgemma-4b-it","config.json")
        shutil.copyfile(cached_path,f"{config.processor_dir}/tokenizer.model")
        shutil.copyfile(cached_config,f"{config.processor_dir}/config.json") #Necessary for conversion
        print("Starting subprocess")
        try:
            result = subprocess.run(
                "./gguf_conversion/bin/python3.11 ./llama.cpp/convert_hf_to_gguf.py ./medgemma-finetuned/merged_adapter --outfile model_test_1.gguf --outtype bf16",
                capture_output=True,
                check = True,
                shell = True
            )
        except Exception as e:
            print(e)

        print("GGUF conversion completed")

        writer.close() #prevent memory leak
        del trainer,model
        print("Cleanup completed")


if __name__ == "__main__":
  main()

