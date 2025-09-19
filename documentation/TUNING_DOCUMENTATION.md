# Documentation:
## Abstract/Executive Summary:
This documentation aims to serve as a guide for finetuning MedGemma, an Image-Text-To-Text model, on a medical dataset (in this case, for colonoscopy) using Quantized Low-Rank Adaptation.

To follow this documentation, a GPU with at least 16 GB of VRAM (such as an RTX 5080 or RTX 4080). Several GPUs or an A100/H100 are preferred, though this documentation is optimized for tuning on a single RTX 5080. Ensure you have enough hardware space for the dataset and model, which can exceed 20 GB depending on the dataset. Training may also take long periods of time; for example, when fine-tuning MedGemma, an RTX 5080 runs roughly 2.5 epochs a day on a 50,000 example training dataset, or roughly 125,000 examples a day.

To follow this documentation, knowledge of Python, basic Linux commands, and a basic grasp of the structure and behavior of LLMs is required.

If you want an easier, more automated pipeline, see the ImageTextToTextTuner documentation in the same directory as this document. It is far more abstracted and automates fine tuning with minimal effort.

## Table of Contents
- [Environment Setup](#environment-setup)
- [Dataset Loading](#dataset-loading)
- [Model Loading](#loading-a-model)
- [Preprocessing](#processing-model-inputs)
- [Fine-Tuning](#fine-tuning)
- [Evaluation](#evaluating-model-outputs)
- [Inference](#inference)
- [Troubleshooting](#troubleshooting)
  
## Environment Setup:
Begin by connecting to the device you intend to train on and set it up. If the device is remote, use the ssh (secure shell) command to connect.  

After setting up your device, run ``` python3 -m venv <NAME> ``` (on Linux) to set up a virtual environment to better manage dependencies. 

Then, to enter the environment, run the command ``` source <FILE/PATH/TO/ACTIVATE ```. This is generally ``` source <NAME>/bin/activate```. 
If the second command does not work, you can navigate the file system through Linux Terminal to find the activate file and replace the file path with that.

After entering the virtual environment, install dependencies. For this project, these have been compiled in the **requirements.txt** document.
If the server is remote, use the scp (Secure Copy Protocol) command to copy the requirements file. 

To install the the dependencies, use``` pip install -r requirements.txt ```. Note that the file must be in your working directory (or you can manually insert a filepath).

 >**Checkpoint #1**: At this point, use the command `pip show transformers` when inside the virtual environment to ensure that the requirements were properly installed.

In addition, note that for data organization, all global variables are stored in the "ScriptConfig" dataclass.
## Dataset Loading:
### Accessing a dataset in code:
The best way to load a dataset available on the Hugging Face datasets page is to use the load_dataset() function. It follows some basic syntax rules to load the correct dataset;
that is, ``` dataset = load_dataset("creator/dataset_name")```.For example, to load the Kvasir VQA dataset—a visual question-answering dataset for colonoscopy images—use the following command: 
```dataset = load_dataset("SimulaMet-HOST/Kvasir-VQA")```. Clicking the "use this dataset" button on a Hugging Face dataset page provides this code (specific to each dataset).

Occasionally, you may only want to load a section of this dataset. If datasets are already partitioned, you can load one of the split sections of the dataset by adding the argument ```split = "NAME OF DATA SPLIT"```. 
Other options are available, such as only loading specific rows, as seen on the Hugging Face documentation [https://huggingface.co/docs/datasets/en/loading]. Loading a dataset from local files is also supported.

> **Checkpoint #2**: At this point, use `print(dataset)` to ensure the dataset has been properly loaded.

### Data preprocessing:
After a dataset is loaded, it may need data processing before use by a model during training, fine-tuning, or inference. For example, there may be unneeded columns or images in the incorrect format.
To remove columns, use ``` dataset.remove_columns("COLUMN_1_NAME", "COLUMN_2_NAME",...)```.

To reformat images and change their type, you can create a function that processes them and use ```dataset.map()``` to apply a function to an entire dataset. Pass in the function you want applied to the entire dataset (for example, MedGemma requires that inputs are formatted in a message format and that all images are scaled to 896x896 pixels). There are also several common speed-ups to this often-slow process, including setting ```num_proc = #number of cores/threads you have``` since multithreading dramatically improves performance. In addition, the map function can be optimized by batch processing, though this requires changes to the structure of the processing function.

Although increasing the number of workers improves performance, it also dramatically increases VRAM and CPU usage, so it is important to find a balance between performance and memory usage. In addition, errors, even memory errors, that occur during mapping when there is more than 1 worker are just generic errors. Therefore, if you encounter errors during the mapping step with vague tracebacks, try setting the num_workers to a lower value. If the error persists, remove the num_workers argument (or set it to 1) for the full traceback.

The map() function can also add columns; each row is passed in as an "example" and you can add columns with formatted inputs. This is extremely useful for formatting images, questions, and answers into a format that can be used later by the processor (see the example["messages"] column, among others). The `processor.apply_chat_template()` requires messages in a specific, dictionary format (as seen in the following code), and the map function facilitates the transformations required to preprocess the data.

An example of this ```.map()``` function in use (where formatMessages is a function that is passed a dictionary and returns another dictionary) is as follows:
```
def format_and_resize_example(example: Dict) -> Dict:
        #Setup flag for removal
        example["remove"] = False
        image = example[config.image_column]
        if not isinstance(image, PIL.Image.Image): #If not a PIL image, convert the image to a PIL image
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
        if answer == "nan": #Handles the "nan" case (but not real nan)
             print("Answer is Not a Number")
             example["remove"] = True #Update the flag for later processing
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
```

Note that MedGemma-4b-it (the model used for this project) accepts PIL images, URLs, and base64 strings, though PIL images are likely the best option because of their versatility. Opening most images using `PIL.Image.open()` converts them to PIL format, though there are some exceptions. To troubleshoot any problems, print the datatype of the images in your dataset using the `type(variable)` function.

During mapping, ensure that the argument `keep_in_memory` is False if you manually save the dataset to prevent repetitive cache files that consume hard drive space.

After mapping the dataset, flagged rows need to be removed. The ``` dataset.filter()``` function is essential for this task, as it keeps only the rows for which the given condition is true. For example, ``` dataset = dataset.filter(lambda example: example["remove"] is not True) ``` removes all rows for which the "remove" value is set to True. 

However, this dataset processing often takes significant amounts of time. Therefore, saving the dataset after filtering and processing effectively reduces the time and compute spent on processing, especially if multiple runs are conducted (or if one of the runs crashes in a step after data processing). Therefore, after the map() function, check if the dataset is saved, and if it is not, use the `datasets.save_to_disk()` function. Before the map() function, check if the dataset already exists, and use the `datasets.load_from_disk()` function. This saves time when debugging, since the `datasets.map()` function can often take several minutes with many workers (>50) and an RTX 5080. 

After processing the dataset, it needs to be split into training and testing sets. There is a useful function, namely ```dataset.train_test_split("test_size")``` to accomplish this seamlessly. This yields a dictionary with two keys, ["train"] and ["test"]. However, ensure you check if the dataset is already partitioned, and use the existing partitions. Partition names vary between datasets (often ["train"] and ["test"] or ["train] and ["eval"]), so ensure you check the dataset's Hugging Face page to determine the name of the splits. Alternatively, you can print out the dataset and examine the column and split names.

In addition, note that the "test_size" argument to train_test_split() must be a value between 0 and 1. Generally, a higher proportion of data is split into the training dataset. 

An example of saving the model and the train_test_split() in use is as follows:
```

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
    
dataset = dataset.train_test_split(test_size = 0.1)
  training_data = dataset["train"]
  testing_data = dataset["test"]
```

 >**Checkpoint #3**: At this point, print the model once again to make sure the transformations were completed correctly. Also, check the file path where the dataset was saved and ensure there are files in that directory.
## Loading a model:
### Basic model loading:
There are two main ways to load a model—namely, through Hugging Face's Pipeline API and more directly—and although both function properly for inference, the Pipeline abstraction is not designed for training. 

To load a model directly, use the transformers.AutoModel class family. To fine-tune MedGemma-4b-it, an image-text-text model, use the transformers.AutoModelForImageTextToText class.
The transformers.AutoModelForImageTextToText class has several arguments, some of which are optional. These include the model (if loading from Hugging Face, use the syntax "creator/model_name"),
quantization_config (set up before loading the model), torch_dtype (the resolution at which weights are stored by PyTorch), and device_map (if this value is "auto", the model is auto-distributed between devices).

If you decide to load the model with quantization, you must create a quantization configuration with the BitsAndBytes library before loading the model and then pass it into the model loader. Quantization essentially reduces the memory use of a model by reducing precision. This allows large models (such as medgemma-4b-it) to be loaded on comparatively small GPUs and trained/run. However, there is a loss of precision, so if you have access to more resources, train or run inference without quantization.

An example of how to configure quantization for Hugging Face's transformers library (using BitsAndBytes) is as follows:
```
quantization_config = transformers.BitsAndBytesConfig( 
  load_in_4bit = True #essentially, load weights/tensors as 4 bit instead of 32, making training faster
  bnb_4bit_quant_type="nf4",#more efficient version of 4 bit quantization, "normalized 4 bit floating point"
  bnb_4bit_compute_dtype=torch.bfloat16,
  bnb_4bit_use_double_quant=True, #double quantization, in this case saves up to ~0.25 GB of VRAM (~0.5 bits per parameter)
  ) #other options include load_in_8bit
```
Next, the model must be loaded. Since MedGemma is an Image Text Text model (input is image, text : output is text), use HuggingFace's transformers library and the AutoModelForImageTextToText class. By passing in the model name/creator, the model architecture is automatically configured, making this abstraction invaluable. Other parameters, such as the quantization configuration, can also be passed into the model. 

### Practical setup/application:
In practice, for hyperparameter tuning (which will be discussed later), the entire model setup needs to be in a single function, in this case called `modelInit()`. This function also needs to include all PEFT (Parameter Efficient Fine Tuning) setup, so the function can be called and an instance of the model is completely set up. 

This script uses QLoRA (Quantized Low-Rank Adaptation), a PEFT technique that creates matrices of parameters alongside the main model, so only a small portion of the model's weights are adjusted. This allows the model to be fine-tuned on a far smaller GPU than with a technique like full or half fine-tuning. When applied to all linear layers, it has a similar effect to full fine tuning, though slightly less generalization occurs.

![LoRA Fine Tuning](https://arxiv.org/html/2408.13296v1/images/Fine-tuning_vs_LoRA.png)

As seen in the image above, Low Rank Adaptation (and its quantized counterpart) add two, trainable matrices to some of the existing, frozen layers of a LLM. These adapters vary in size and "strength"(how much their activation changes the output of the layer, better known as LoRA Alpha) but are entirely trainable. This technique lowers the barrier to entry for fine tuning LLMs, making it possible for individuals or small corporations. Rank, better known as LoRA r, affects the size of the matrices, impacting learning capability. However, some configurations, especially with high r on small datasets, could promote rapid overfitting (as will be discussed in the training section.

However, on some devices, there is not enough memory to fine tune all linear layers, so only certain layers are edited. For example, the layers ("up_proj","down_proj" and "gate_proj") are all tuned, as they are standard linear layers and fairly close to the output of the model. If you have significant memory constraints and continue to receive CUDA out of memory errors, reduce the number of layers. If you still encounter memory errors, reduce the LoRA r value (which defines the size of the LoRA matrices) and adjust the LoRA alpha value (it should be 1-4 times the r value, though 2 times larger is customary). 

The LoRA alpha value impacts how much weight the model gives to LoRA matrices; high values can promote overfitting or catastrophic forgetting (where a LLM overspecializes during fine-tuning or training and forgets past knowledge), and low values mean that the fine-tuning has little effect on the output of the model. These parameters do not have to be manually tuned, however, as they are tuned in hyperparameter optimization (which will be discussed later).

For different models, to determine what the names of each layer is, simply print the initialized model instance (i.e. `print(model)`).

An example of the model initialization setup is as follows:
```
def modelInit(lora_r,lora_alpha,lora_dropout):
        medgemma_original = AutoModelForImageTextToText.from_pretrained(
            config.model_id,
            quantization_config=quantization_config,
            torch_dtype=model_torch_dtype,
            device_map="auto",
            trust_remote_code=True,
        )

        lora_config = LoraConfig(
                r=lora_r, #These LoRA variables are not necessary for inference, only for training
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=config.lora_target_modules,
                bias="none",
                task_type="CAUSAL_LM",
        )

        #Finish full model init, with PEFT setup (Hyperparameter tuning requires this to be self-contained)
        if config.quantization:
            model = prepare_model_for_kbit_training(medgemma_original) #formats the quantized model properly for training regardless of gradient checkpoint status

        if config.gradient_checkpointing:
            model.config.use_cache = False # True is incompatible with gradient checkpointing
        
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        
        return model
```

Here, the `model = get_peft_model(model,lora_config)` merges the LoRA architecture with the base model and creates a new instance of the model. This function is callable and
returns an instance of MedGemma that is perfectly set up for all tuning, allowing multiple instances (with different hyperparameters, such as LoRA r or LoRA alpha values) during hyperparameter tuning. Printing trainable parameters also serves as a way to verify that the PEFT setup was successful. 

 >**Checkpoint #4**: Using `model.print_trainable_parameters()` is an effective way to determine if the model is configured correctly. If you get a result between 0.1% and 1% of the model's total parameters, you have likely configured the model correctly. Note that changing LoRA r affects this value, so ensure LoRA r around 8-32

If you want to load an existing LoRA adapter, you can use the `PeftModel.from_pretrained()` method. That is,
```
model = PeftModel.from_pretrained(model, config.adapter_path)
model = model.merge_and_unload()
```
If you are running inference, it is important to use the `merge_and_unload()` method, but if you are continuing an old training session, make sure you do not merge the model.

## Processing model inputs:
### General processing background:
Currently, the loaded model would not be useful for either inference and training, as the current code lacks a processor.
A processor helps parse inputs, change their format, tokenize them, and convert them to tensor form. 

Using the transformers.AutoProcessor class family is the easiest way to create a processor. Using the ```.from_pretrained()``` method, you can load the default model processor which reduces the risk of errors and streamlines input processing. The syntax to set up a processor is as follows:
```
model_processor = transformers.AutoProcessor.from_pretrained("google/medgemma-4b-it") #or whatever your model path is
```
Inputs must be passed through the model processor before being fed to the model. First, text inputs in the message format shown in the data loading section must be passed through the `model_processor.apply_chat_template()` function with the `tokenize = False` argument to ensure the correct format. Then, this text must be passed through the processor with the following arguments:
```
model_input = model_processor(text = prompts, images = images, return_tensors = "pt", padding = True)
```
Note that here, images are already processed into the correct resolution, and `prompts` are the input messages with the chat-template applied. 

These processing steps occur in the data collator, a function that collects the training/testing data and processes it before passing the data into the model. It is a training argument.

Note that you can validate that everything is loaded correctly and that the processor functions effectively by running inference on an image (especially a medical image) and ensuring the output is reasonable and intelligible, as described in the final section. That is,

 >**Checkpoint #5**: If you wish to ensure your processor and model are set up correctly, skip to the Inference section and run the inference code to ensure everything is working properly. If the model outputs are nonsensical, you likely have a processor configuration error.

### Data Collator:
As previously described, a data collator collects data and processes it before passing the processed data into the model. The default data collator does not work for this problem because of the input formats of the training and testing data, so a custom collator must be used. The task of the collator is as follows:

|Input|Output|
|--------------------|--------|
|List of dictionaries|Batch, a dictionary, composed entirely of tensors (with keys that are strings) with the correct values masked|

Several examples in a batch are provided to this data collator, so the data collator must be able to handle a list of dictionaries. For each example, the prompts are formatted through the model's chat template, and added to a list of formatted prompts. Images were already processed into the correct resolution by the ```dataset.map()``` function during dataset processing, but are appended to a list of images. 

After the data collator iterates through all examples, the list of images and the list of formatted prompts are passed into the model processor. That is,
```
batch_inputs = self.processor(
  text=texts,
  images=images_for_batch_processing, # Pass the correctly formatted list of lists
  return_tensors="pt",
  padding="longest",
  truncation=True, # Ensure truncation is enabled
  max_length=self.max_length # Explicitly set max_length from config
)
```

This function returns a dictionary with various keys (including ["input_ids"] and ["labels"]). Inputs should be left un-masked, but if labels (what the model uses to compute loss) are unmasked, the model may learn to repeat parts of the prompt in its generation. Therefore, we must mask the prompt and image tokens in ["labels"] to prevent this behavior. 

First, a copy of ["input_ids"] is created, populated with the value -100 (which is not read by the model). Then, the prompt is processed again to determine its length and any values
after the end of the prompt are re-inserted to the ["labels"] tensor. However, this means the image tokens are re-added and must be masked.

The most robust way to mask image tokens is to manually mask several ids:
```
#Mask the image tokens (262144,260000, 259999)
batch_inputs["labels"][batch_inputs["labels"] == config.image_token] = -100
batch_inputs["labels"][batch_inputs["labels"] == config.image_end_token] = -100
batch_inputs["labels"][batch_inputs["labels"] == config.image_start_token] = -100
```
With the MedGemma's SigLIP processor, 262144,260000, and 259999 are the image ids and need to be masked in ["labels"].

After this has completed, the batch_inputs is returned (which contains ["input_ids"],["labels"],["pixel_values"], etc.) 

## Fine-Tuning:
### Basic trainer setup:
To begin fine tuning, the trainer has to be set up with a set of training arguments. Using trl's SFTTrainer, a modified version of Hugging Face's trainer, is often a better choice than the default trainer. For example, SFTTrainer has progress bars and more preprocessing, streamlining the training process.

SFTTrainer (and Hugging Face's trainer) require training arguments, set up through the SFTConfig class. For example, 
```
training_args = SFTConfig(
        output_dir=config.output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=config.train_batch_size,
        per_device_eval_batch_size=config.eval_batch_size,
        ...
```
These training arguments influence the behavior of the model, with factors like learning rate and batch size included. These values are often tuned by hyperparameter tuning setups (which will be discussed later).

The trainer is passed arguments including training arguments, data collators, and datasets. It can be set up with the following syntax:
```
trainer = SFTTrainer(
            model = model, #Note that the model is initialized from the modelInit() function from a previous section
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator, #The data collator from the data processing section
        )
```

### Hyperparameter tuning:
#### Background:
For hyperparameter tuning, training arguments and the trainer as a whole are re-initialized each trial and thus must be initialized through a function, but the general structure of the training setup remains the same.

For this project, Optuna, a hyperparameter tuning library, was used. Optuna has several benefits, including its versatility. Optuna works using a "study" system, where studies are set up and then optimized to minimize or maximize a metric. For example, the current setup minimizes cross-entropy loss.

Hyperparameter tuning is optional; the model can train without it. However, optimal hyperparameter settings reduce loss, increase training efficiency, and impact final model performance. The optimal setting of various hyperparameters differs based on model structure, task, etc. and although there are basic guidelines, optimizing hyperparameters automatically is generally preferred.

#### Studies:
Studies essentially define the optimization task and all associated arguments. The arguments they are passed determine optimization behavior; for example, the sampler determines which samples are chosen and the pruner terminates trials that appear unsuccessful. 

There are several samplers, each with their own advantages. For example, the random sampler randomly selects values for hyperparameters and chooses the best results. This can work with extensive compute (essentially, the "brute-force" of hyperparameter tuning). On the contrary, the TPE sampler (Tree-structured Parzen Estimator) finds combinations of hyperparameters that work better together and search areas with better results, making it compute-efficient and faster. The TPE sampler was used in this project due to compute limitations, but if compute was unlimited, the CmaEsSampler could also be used. It is more compute-heavy. However, on smaller hyperparameter search spaces, random sampling, the TPE sampler, and the CmaEsSampler perform similarly.

#### Objectives
An "objective" function is designed to return the value to maximise or minimize, and this value is passed into the study for optimization. In this function the model and trainer are initialized and training begins (albeit for only a very small number of steps) to determine the effectiveness of each training metric. Loss is evaluated and returned by this function. In addition, the trainer is promptly deleted by the function to prevent memory leaks.

This loss is then passed to the study and the study optimizes the hyperparameters over several trials to minimize loss.
```
def objective(trial):
    trainer = trainerSetup(trial = trial)
    trainer.train()

    eval_loss = trainer.evaluate()["eval_loss"]
    print(f"Eval_loss {eval_loss}")

    
    writer.add_scalar('Loss/tune',eval_loss,trial.number)

    del trainer
    gc.collect()

    return trainer.evaluate()["eval_loss"]

study = optuna.create_study(sampler=config.hyperparameter_sampler,direction = "minimize",pruner = config.hyperparameter_pruner,study_name="medgemma_finetuning_hp_tuning",load_if_exists=True)
study.optimize(objective, n_trials=config.hyperparameter_trials,show_progress_bar = True)

```

However, the deletion process behaves oddly; even though the trainer was deleted the memory was not entirely cleaned up. Therefore, deleting each attribute (for example, `trainer.model`) individually appears to help with memory usage, along with clearing the cache frequently. 

#### What hyperparameters should be tuned?

Minimizing the search space increases the odds that the optimal model configuration is found in the fewest number of trials. Even with the TPE sampler, the larger the search space, the more trials are required to find a satisfactory outcome. In this script, the following hyperparameters are tuned:
```
learning_rate= trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True)
gradient_accumulation_steps= trial.suggest_categorical("gradient_accumulation_steps", [1,2,4,8,16]) 
            
#LoRA parameters:
lora_r_val = trial.suggest_categorical("lora_r", [16, 32]) #r of 64 crashes
lora_alpha_val = trial.suggest_categorical("lora_alpha", [16, 32, 64, 128])
lora_dropout_val = trial.suggest_float("lora_dropout", 0.0, 0.1, step=0.01)
```
In this case, tuning both gradient accumulation steps and batch size is fairly redundant, since batch_size * gradient_accumulation_steps yields the "effective batch size". By removing batch size, the process is sped up dramatically. Also, categorical options or floats with step sizes decrease the total number of possibilities, speeding up tuning by reducing the number of trials required to find a result.

Note, however, that more trials will almost always yield a better outcome, since the best trial is selected (whether it be, for example, trial 5, 15, or 125). Despite this, diminishing returns are present at higher trial counts.

One method to speed up hyperparameter trials is by using a pruner, such as the `MedianPruner` or `Hyperband Pruner`. These pruners terminate failed trials early, saving compute and speeding up tuning. Various pruners have different benefits, as illustrated here: [https://optuna.readthedocs.io/en/stable/reference/pruners.html]

Another method to expedite hyperparameter tuning is to have each trial take place over a smaller epoch. However, this prioritizes fast-converging setups rather than parameter combinations that will result in the lowest final loss, leading to a suboptimal final result. In fact, many of these setups may end up overfitting—essentially, memorizing the training data and not properly generalizing—early in the training process because the hyperparameter tuner prioritized fast, aggressive growth. 
> If your hyperparameter epochs are far shorter than your training epochs, your hyperparameter tuning results will almost certainly prioritize fast-converging hyperparameter combinations



#### Cleanup and optimizations:
To extract hyperparameter information from Optuna, use the `study.best_params.get()` method for each hyperparameter (for example, "lora_r", "learning_rate", etc.). Then, use the Optuna.delete_study() method to delete the study, freeing up memory for the final training, or clear the cache and collect garbage. Collecting garbage also serves to delete the study, as there are no further references after the `study.optimize()` function is called.

Note that hyperparameter tuning is extremely sensitive to memory leaks; if even 0.1 GB accumulates after each trial, tuning may fail due to a memory issue after 40 trials. To remedy this, before deleting the model after each step, first move it to the CPU to clear up memory. Also, synchronize the GPU and CPU before deleting to ensure all trainer components are deleted. Finally, PyTorch tensors sometimes have circular references that are not identified by the garbage collector, so manually deleting these attributes is sometimes appropriate. 

By evaluating and training only a small subset of the data during training (for example, 5-10% of your training/testing split) you can dramatically speed up training and test the impacts of multiple epochs during hyperparameter tuning more quickly, approximating the behavior of the model on the full dataset. More epochs is generally preferred to more examples to determine final convergence behavior. However, if the compute is available, always run hyperparameter tuning on the full dataset to ensure accuracy.

### Training
After hyperparameter tuning, reinitialize the trainer and model (with trial = None to prevent randomized hyperparameters). Then, after clearing the cache and collecting garbage,
run `trainer.train()`. After waiting for the model to train, all fine tuning is completed. If `load_best_model_at_end()` is True, saving the model at the end will yield the best saved model. This model can now be used for inference or evaluation.

During training, watch Tensorboard (or the model outputs) for discrepancies. If training and evaluation loss differ significantly (and eval_loss > training loss), overfitting is likely. Overfitting occurs when the model "memorizes" the training data after a few epochs and fails to generalize. 

If the model appears to be overfitting, terminate the process using Ctrl-C. There is a try-finally block in the training loop that allows for manual early-stopping. 
After terminating the model, examine your minimum loss. If minimum loss or accuracy does not meet the desired threshold, examine hyperparameter tuning again, this time prioritizing final loss more than fast convergence by having longer hyperparameter tuning epochs.

 >**Checkpoint #6**: To test training, set the epochs to 0.1 and set `tune_hyperparameters = False`. If this works, set `tune_hyperparameters = True` and set `hyperparameter_epochs = 0.05`. By only using a few hyperparameter trials, you can quickly test the setup.

If you trained the model but it did not converge and requires more epochs, in `trainer.train()` you can pass the argument `resume_from_checkpoint = "path/to/checkpoint"`. Alternatively, setting `resume_from_checkpoint` to True loads the most recent checkpoint.

### Evaluating model outputs:
#### Evaluation metrics
Models are evaluated using the f1 score to determine performance. However, there are other benchmarks, such as accuracy, that are worth considering. Models are also evaluated with both static and adaptive prompts to find the model's ideal performance. 

The accuracy benchmark, `evaluate.load("accuracy")` is essentially how it sounds; true positives and true negatives are recorded, along with false positives and false negatives. However, this does not account for the distribution of different answer choices. 

On the contrary, the f1 benchmark combines length and accuracy, normalizing it. Essentially, it combines precision and recall to create a slightly more accurate view of the model's performance.

BERT-Score is an efficient way to capture semantic similarity, especially with the bertscore_f1 metric. This metric was also used in this project.

There are other benchmarks, such as the LAVE framework suggested by one paper, which uses LLMs to assess accuracy [https://doi.org/10.48550/arXiv.2310.02567]. However, the computational overhead for this is excessive.

Model outputs are evaluated on the evaluation split of the dataset to prevent data contamination, as evaluating on the training data would risk inflating the benchmarks. The f1 score is used to calculate the effectiveness of the model. The output scores of each of the 6 model configurations (base model + tuned model evaluated with no prompt, a static prompt, and an adaptive prompt) are collected and graphed using the Seaborn library.

#### Data processing - Miscellaneous
Because the model has to switch between 3 prompts (no prompt, static prompt, and adaptive prompt), the data processing function called by `map()` creates 3 columns: messages, messages-static, and messages-adaptive. Creating 3 separate datasets would have entailed excessive hard-drive usage and is incredibly inefficient. Also, to prevent weird masking issues, a fourth "message-answer" column was created that contained the answers.

However, as there are three columns, a method must be implemented to shift between the columns. To solve this, the data collator is instantiated with a "message_column" argument. For the first two evaluations (base_model and tuned_model with no prompt), the data_collator is instantiated with the argument `message_column = "messages"`. After these evaluations conclude, the data_collator is re-initiated with a different message column. While not the most elegant solution, this serves to change the column and allow for the evaluation of models with no prompt, adaptive prompts, and static prompts.

The answer column is tokenized and added to the batch_inputs["labels"] column. Unlike the training example, it is important to note that the labels are a different size than the inputs.

#### Custom evaluator function
Since the default trainers do not support evaluating quantized models and if models are not quantized, a CUDA Out Of Memory error results, a custom training loop must be created. In this project, we created a function that iterated across the entire training dataset for each model configuration, recording accuracy and bertscore data. This data was then, after the evaluation run for each model concluded, averaged  and collected for later use. 

In this `evaluate_instance` function, each example (row of the dataset) is passed through the data collator. Since the new data format means that answers are not included in the input_ids, no further processing is required. These inputs are passed into the model using the `model.generate(**model_inputs, max_new_tokens = 25, do_sample=False)` function.

Lowering the `max_new_tokens` argument in the `generate` function speeds up evaluation. However, if it is too low, the default MedGemma model (which is very verbose) may not be able to finish, ensuring that semantic similarity is lower. Despite this, there is little apparent difference between the semantic, bertscore_f1 at 25 tokens compared to 100, though this change leads to, in practice at least a 3x speed-up to evaluation time (though in theory, up to 16x increases in performance). 

The beginning of the model generation is then stripped of the prompt using list slicing. That is,
```prediction = generation[:, prompt_length:]```

Essentially, the shape of the input_ids (the prompt + image) is saved and anything that matches is removed. This means that labels and the prediction have the same shape (though often, the prediction may be longer than the answer). Finally, both the labels and prediction are padded to the same length.

#### Evaluation metrics:

After this formatting, the evaluation metrics are retrieved through the Evaluate library, a Hugging Face library. ROUGE, BERT score, and several other, more foundational metrics such as raw, absolute accuracy are included.

ROUGE measures the direct overlap between words in a prediction, but struggles occasionally with shorter answers. However, it can be a good metric to gain some insight in model performance.

```
rouge_score = rouge.compute(predictions=[prediction],
                references=[labels],                         
                rouge_types=['rougeL'],
                use_aggregator=True)
```

ROUGE 1 measures consecutive 1 word overlaps whereas ROUGE-L measures longest sequences, making both methods versatile. Utilizing ROUGE-L is standard practice and yields a good benchmark of accuracy. However, manually evaluating samples by hand (or at least a subset of them) can determine how effective the tuned model is.

Direct match accuracy is measured by directly comparing sequences (that is, it is binary). The existing accuracy metrics are designed more for classification, so a custom, very basic metric was created by determining if sequences matched exactly.

Accuracy, along with the BERT semantic similarity metrics (generated through another evaluate line) are each added to their own individual list. After evaluation has completed for each model, these metrics are averaged and the average values are stored for later graphing. Averaging is achieved by dividing the sum of the list by the length of the list. 

## Exporting the model:
### Creating a venv
To export your tuned model in GGUF format, you must first make a virtual environment. That is,
`python3 -m venv gguf_conversion` (or any other name you wish to call your venv)

After activating the venv with `source gguf_conversion/bin/activate`, we must clone the llama.cpp Github repository.
### llama.cpp
llama.cpp is a Github repository that contains the code necessary to convert a HuggingFace safetensors model to GGUF format. It can be installed with `git clone https://github.com/ggml-org/llama.cpp`. Then, install the dependencies for the conversion by using `pip install -r llama.cpp/requirements.txt`. These dependencies conflict with the dependencies required for the rest of the setup, so make sure this is in a virtual environment.

### Model Setup:
Next, the model must be set up and the LoRA adapter must be merged with the model before conversion.
To load the model, the same code as before must be used. However, llama.cpp does not support 4 bit quantization conversions, so it is important to load the model with unquantized weights. That is,
```
model = AutoModelForImageTextToText.from_pretrained( #creates an instance of the model
            "google/medgemma-4b-it", #model source
            torch_dtype = torch.float16,
            low_cpu_mem_usage=False,  
            _fast_init=False) 
        )
```
This creates an instance of the base model. Note that `low_cpu_mem_usage` and `_fast_init` prevent "lazy loading" where tensors reference each other. This is useful for inference and reduces memory issues but causes issues when saving.

Loading the adapter is accomplished as follows:
```
model = PeftModel.from_pretrained(model, config.adapter_path)
model = model.merge_and_unload()
print("LoRA adapter merged successfully.")
```
The `PeftModel.from_pretrained` loads the adapter in relation to the base model. Then, the `merge_and_unload()` method works to fuse the model and adapter so they can be saved together as if they were one model. 

After the model is loaded, sending it to the cpu (using `.to("cpu")`) decreases loading issues. Tying the weights using `model.tie_weights()` prevents the weights from becoming untied and forming meta tensors (valueless) on saving.

To save the model, use the `model.save_pretrained(config.output_dir, safe_serialization = True)` method. Safe serialization ensures that the outputs are saved to safetensor format rather than Pytorch `.bin` files.

The processor can also be saved using `processor.save_pretrained(config.processor_dir)`.

However, some files are not properly saved and have to be pulled from the model repository. That is,
`cached_path = hf_hub_download(config.model_id,"tokenizer.model")
cached_config = hf_hub_download(config.model_id,"config.json")
print("Beginning copying")
shutil.copyfile(cached_path,f"{config.processor_dir}/tokenizer.model")
shutil.copyfile(cached_config,f"{config.processor_dir}/config.json")`

Here, shutil copies the files from HuggingFace to the correct directory.

### Converting the model:
There are two ways to convert the model: through the terminal or automatically using Python's subprocess command.

The best way is generally the automatic, subprocess way. llama.cpp generates for multimodal models 2 different files (multimodal project and general model). The code to do that with subprocess is as follows:
```
result = subprocess.run(
            f"{config.venv_python_path} ./llama.cpp/convert_hf_to_gguf.py mmproj_{config.output_dir} --outfile {config.model_name}.gguf --mmproj --outtype f16 --verbose"
            capture_output=True,
            shell = True
        )
print(result)
result = subprocess.run(
    f"{config.venv_python_path} ./llama.cpp/convert_hf_to_gguf.py {config.output_dir} --outfile {config.model_name}.gguf --outtype f16 --verbose",
    capture_output=True,
    shell = True
)
```
Alternatively, you can use a command like:
`python ./llama.cpp/convert_hf_to_gguf.py ./medgemma-finetuned/merged_model_test --outfile model_tuned_2.gguf --outtype f16 --verbose`

Finally, although automatically quantizing the models is fairly difficult, to run a 4 bit quantized model with Ollama, use:
`ollama create my-quantized-model -f Modelfile --quantize q4_K_M`

## Inference:
### Generating model outputs:
There are once again two ways to generate model outputs: the Pipeline abstraction and loading the model directly. Both methods work in this case, though loading the model directly is required for evaluation. However, using the Pipeline abstraction offers less flexibility and thus loading the model directly is generally the preferred option. 

The initial loading process is nearly identical to the steps mentioned in a previous section; the model is loaded (this time without creating a PEFT config) and the processor is configured. The only major difference is loading the LoRA adapters; use the `model.load_adapter("file/path/here")` method to load the adapter from a file.

Then, configure a message template (similar to the one in the data processing step previously) and tokenize all inputs. That is, 
```
messages = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": system_prompt}
                ]
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image"}, 
                ]
            },
        ]
        
        prompt = processor.apply_chat_template( #Process input
            messages,
            add_generation_prompt = True, #Inference not training
            return_dict = False,
            tokenize = False,)
        
        model_inputs = processor(text=prompt, images = [image], return_tensors = "pt", padding = True) #The processor expects lists of lists of images

        print(f"Model Inputs: {model_inputs}")
        return model_inputs
```

This structure is almost identical to that in the map() function in the data processing section. The only major difference is that a system prompt is added and the function is designed for one prompt, not iteration across an entire dataset.

Passing these model inputs into the model is accomplished by calling `model.generate(model_inputs)`. Other arguments in the generate command include `max_new_tokens`, which specifies the maximum generation length, and `do_sample`, which, if false, makes the model deterministic.

To decode the model outputs, use the following syntax: `response = processor.batch_decode(model_output,skip_special_tokens=True) `. The batch decode method translates the tensors into a string. Although decode also works for single generated responses, batch_decode can be more efficient and works for multiple simultaneous generations, which can be useful on a larger scale.

Inference is an invaluable way to debug processor failures; if the model outputs gibberish unrelated to the prompt (including in various languages), it is likely that there is a problem with the way the processor is set up. 

### Prompt engineering:
#### Static Prompts:
Static prompts are the simplest way to extract more performance from a LLM. Several strategies are useful for prompt engineering: providing examples of desired outcomes, giving the model a "persona" and outlining the task at hand. An example of a basic static prompt for a model in the field of colonoscopy is as follows:

`You are a leading medical expert in the domain of colonoscopy and analyze medical imagery with high accuracy,
identifying all anomalies and answering questions factually, and concisely. Note that 'finding' is generally code for abnormality.
Respond in a concise manner; for example, when given an image of ulcerative colitis and a prompt "Are there any 
abnormalities in the image?", respond with "Ulcerative Colitis"`

This prompt structure may not be optimal but it satisfies many of the basic requirements for a prompt. The example is also extremely useful in communicating the output format to the model, especially the base model (which has not been trained on this dataset before). 

#### Adaptive Prompts:
Adaptive prompts have been shown to increase performance in some cases, but these systems can be incredibly complex and have high computational overhead; in fact, some systems even use small language models to generate prompts. 

In the interest of minimizing computational overhead, a "keyword matching" system was used with verbiage taken directly from the prompt. These keywords were matched to categories/tasks, with each category having its own, custom prompt. The categories are as follows:  "Instrument Details" (determine if a box or text is present, etc.), "Determine type", "Size/Measure", "Find in image", "Determine color", "Counting", "Anatomical Landmarks" (generally, finding or describing them).

An example of a prompt for one of the categories (counting) is as follows:
`Carefully count the number of target features in the image and reply concisely. For example, when given an image of the GI tract with only ulcerative colitis and the prompt "How many findings are present?", respond with "1".`

These categories were determined by examining the Kvasir-VQA dataset and sorting it. Admittedly, there are better ways to classify the dataset and choose prompts (even by using feedforward neural networks) but in the interest of time and computational cost, these options were not explored. 

The keyword logic is simple; if a keyword is detected, append the appropriate prompt to the system prompt (assuming the prompt has not already been added). Although some prompts may overlap such that both would be added for one prompt, this is unlikely (after performing a cursory evaluation of the dataset). 

#### Importance of prompt engineering:
After evaluation, the following graph of the performance of the base and tuned model with various levels of prompt engineering was generated:

[Graph + analysis]

## Troubleshooting:
### CUDA Out Of Memory issues (general advice):
CUDA Out Of Memory issues are some of the most difficult to solve. During fine-tuning, if too many layers are tuned by LoRA, the memory usage skyrockets; for example, if 3-5 layers are tuned, the memory usage is tolerable on an RTX 5080 (assuming the LoRA rank isn't excessively high). However, if more than 5 layers are tuned, creating and updating the matrices (regardless of how small the LoRA rank is) consumes much of the entry, so the script will likely crash during the backwards pass (if not sooner). Reducing LoRA rank also helps reduce memory usage.

Other reasons for this error includes fragmentation, where memory is allocated but cannot be used because there is not enough contiguous memory. To resolve this, set `os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'`.

If memory issues persist, try decreasing image resolution (for example, to 448x448) and decreasing batch size. Changing torch_dtype from float 32 to float 16 or bfloat 16 also reduces memory usage.

Also, if memory always runs out (after the first step), search for memory leaks or variables that were not freed.

### CUDA Out Of Memory issues (hyperparameter tuning):
Unfortunately, since hyperparameter tuning requires many model creations and deletions, small, un-deleted memory, over 40 trials, can cause an Out Of Memory Error. Clearing the cache and collecting garbage using `gc.collect()` is useful. Also, manually deleting attributes like trainer.model as well as trainer reduces the risk of circular references that prevent garbage collection. 

Clearing the cache and collecting garbage before model creation also reduces the risk that existing models will remain loaded, accumulating memory rapidly.

Depending on the configuration of the dataset saving, avoid terminating the process during mapping/processing, which could create large, useless cached dataset files. Also, set ` keep_in_memory = False` in the map function to minimize the risk of repetitive, redundant files if the dataset is manually saved to a directory.

### General advice:
- Always move your tuned adapter into a backup folder and make a copy of it on GitHub to prevent the loss of any valuable weights, setting back the entire training process.


