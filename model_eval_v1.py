import torch,os,PIL,io,gc
import numpy as np
from transformers import AutoProcessor,AutoModelForImageTextToText,BitsAndBytesConfig
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from huggingface_hub import login
from datasets import load_dataset,load_from_disk
import seaborn #matplotlib but prettier
import evaluate
import matplotlib.pyplot as plt
from peft import PeftModel


os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
# 1. --- Configure inputs ---
@dataclass
class ScriptConfig:
    """Stores all hyperparameters and configuration settings for the script."""
    # Model and Tokenizer
    model_id: str = "google/medgemma-4b-it"
    hf_token: Optional[str] = None # Loaded from .env file
    output_dir: str = "./medgemma-finetuned"
    adapter_path: str = "./medgemma-finetuned/backup_adapter"
    device: str = "cuda"
    image_end_token: int = 256000 #the tokenized tensor value for this token
    image_token: int = 262144 #the "fill" image token for all image tokens except the start and end
    image_start_token: int = 25999 

    # Dataset
    dataset_name: str = "SimulaMet-HOST/Kvasir-VQA"
    dataset_split: str = "raw" #Useful for only loading a subset of a dataset
    eval_split_name: str = "test"
    train_split_name: str = "train" #Leave as default if your dataset (like Kvasir-VQA) has no train/test split
    test_size: float = 0.1
    image_column: str = "image"
    question_column: str = "question"
    answer_column: str = "answer"
    num_workers: int = 3*(os.cpu_count()) #or set to a small int (4x GPU count?), os.cpu_count() may take too much memory
    shuffle: bool = False
    load_data_from_file: bool = True

    # Image Preprocessing
    image_resize_size: tuple = (448, 448) # Upscale images for more details

    # Model Configuration
    quantization: bool = True
    medgemma_tuned_path: str = "./medgemma-finetuned" #Filepath to the saved model
    fp16: bool = False
    bf16: bool = True
    torch_empty_cache_steps: int = 50
    
    # MedGemma's typical context length is often 2048 or 4096. 
    max_sequence_length: int = 4096

    system_prompt: str = (
        """
    You are a medical expert in the domain of colonoscopy and analyze medical imagery with high accuracy,
    identifying all anomalies and answering questions factually. Note that 'finding' is generally code for abnormality.
    Respond in a concise manner; for example, when given an image of ulcerative colitis and a prompt "Are there any 
    abnormalities in the image?", respond with "ulcerative colitis"
    """
    )


    #     """You are a leading medical expert in the domain of colonoscopy and analyze medical imagery with high accuracy,
    #   identifying all anomalies and answering questions factually, and concisely. Note that 'finding' is generally code for abnormality.
    #   Respond in a concise manner; for example, when given an image of ulcerative colitis and a prompt "Are there any 
    #   abnormalities in the image?", respond with "Ulcerative Colitis". Keep to this format no matter what. Do not use any HTML or markdown formatting (or formatting at all), 
    #   and just provide the answer. Do not repeat yourself or provide more than one version of the same answer. Unless there is a reason for more than 
    #   one clear answer (like multiple diseases), only respond with one answer. If you are not asked to select more than one answer (like "find all", 
    #   "check all that apply", etc.) make sure to only provide one response.Make sure, for "yes/no" answers, that you choose the correct answer.
    #   Respond in English and avoid asking follow-up questions or adding explanations.""")

    instrument_keywords:List[str] = field(default_factory=lambda:[
    "instrument","tool","artifact","object","device","box",
    "wire","implant","clip","text"  # Only if related to markings/labels on instruments
    ])

    determine_type_keywords: List[str] = field(default_factory=lambda: [
    "type","kind","is this a","classification",
    "identify","category","nature" #like "nature of"
    ])

    measure_keywords: List[str] = field(default_factory=lambda: [
    "size","length","diameter","width","how big",
    "dimension","measure","scaling"
    ])

    image_keywords: List[str] = field(default_factory=lambda: [
    "where","find","locate","region","area","zone",
    "position","highlight"
    ])

    color_keywords: List[str] = field(default_factory=lambda: [
    "color","hue","shade","tone",
    ])

    counting_keywords: List[str] = field(default_factory=lambda: [
    "how many","number of","count","total","are there any",
    "all that are present","quantity"
    ])

    landmark_keywords: List[str] = field(default_factory=lambda: [
    "landmark","anatomical","identify structure","body part",
    "specific region","boundary of","reference point","structure"])

def get_memory_usage():
    #Returns the current GPU memory usage in GB
    allocated = torch.cuda.memory_allocated(0) / (1024**3)
    total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    free_memory = (torch.cuda.mem_get_info(device=0)[0])/ (1024**3) #includes other processes

    utilized = total-free_memory
    return (f"Utilized: {utilized:.2f} GB / {total:.2f} GB \n Allocated by PyTorch: {allocated:.2f} GB / {total:.2f}")

print(f"Initial memory usage: {get_memory_usage()}")

def main():
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

    if not config.hf_token:
        print("Warning: Hugging Face token not found. Proceeding without authentication. This may fail for private models/datasets.")
    else:
        print("Hugging Face token found. Logging in.")
        login(token=config.hf_token)

    print("\n--- Initial Setup completed ---")

    # 2. --- Set up adaptive prompt architecture ---
    print("\n--- Initiating adaptive prompt architecture ---")
    def generatePrompt(input_text): #for inference not training
        system_prompt = config.system_prompt #Returns basic prompt if keywords do not match
        base_input = (input_text).lower()
        #Remove punctuation
        mapping_table = str.maketrans("","",".!?")
        
        decomposed_input = base_input.translate(mapping_table) #Removes punctuation
        decomposed_input = base_input.split(" ") #split into words
        
        #print(decomposed_input)

        classifications = [] #What categories should be added to prompt, just to keep track of what has been added and prevent redundant prompts
        #Check each category ("brute-force")
        for word in decomposed_input: #there may be a more efficient way to check than O(N) time complexity or look more elegant, I will work on this.
            if word in config.instrument_keywords and "instrument" not in classifications:
                classifications.append("instrument")
                system_prompt += """ If you are answering a question about an instrument, screen display, etc, mimic the following answer format: 
                Given a colonoscopy image and the prompt "What type of procedure is the image taken from? respond with "colonoscopy". If this is not
                a instrument question, just respond to the original question"""
            
            if word in config.determine_type_keywords and "type" not in classifications:
                classifications.append("type")
                system_prompt += """If you are asked the type of a feature in an image, focus on the type of anomalies or attributes in the image. 
                For example, when given an image of a colonoscopy with no polyps and the prompt "What type of polyp is present?", respond with "None" 
                If this is not a clasification question, just respond to the original question"""
            
            if word in config.measure_keywords and "measure" not in classifications:
                classifications.append("measure")
                system_prompt += """If the question asks about the attributes of an object, focus on the size, position, and quantitative attributes of the object you are being asked about. For example, given an image of 
                    a polyp and the prompt "What is the size of the polyp?", respond with "11-20mm", or whatever size-range the polyp could be.
                    If this is not a measurement question, just respond to the original question"""
            
            if word in config.image_keywords and "find" not in classifications:
                classifications.append("find")
                system_prompt += """If this question asks you to find an object in an image, find the target landmark or feature in the image, and say "none" if it does not exist. Ensure that you scan the image thoroughly. For example,
                    if given an image that contains 2 abnormalities and a prompt of "Where in the image is the abnormality?", respond with "center; center-left;", if that
                    is where the abnormality is. If this is not a 'find in image' question, just respond to the original question """
            
            if word in config.color_keywords and "color" not in classifications:
                classifications.append("color")
                system_prompt += """If this question asks you to find color, find the color of the target feature or landmark. Ensure that you scan the image thoroughly. For example,
                    if given an image that contains 2 abnormalities and a prompt of "What color is the abnormality? If more than one separate with ;", 
                    respond with "pink; red", if those are the colors of the abnormalities. If this is not a 'determine color' question, 
                    just respond to the original question"""
                
            if word in config.counting_keywords and "counting" not in classifications:
                classifications.append("counting")
                system_prompt += """If this question asks you to count the number of items, carefully count the number of target features in the image and reply concisely. For example, when given
                    an image of the GI tract with only ulcerative colitis and the prompt "How many findings are present?", respond with "1".
                    If this is not a counting question, just respond to the original question"""

            if word in config.landmark_keywords and "landmark" not in classifications:
                classifications.append("landmark")
                system_prompt += """If this question asks you to find an anatomical landmark, carefully find the location of all important anatomical features in the image. For example, when given
                    an image of the GI tract with no landmarks present and the prompt "Where in the image is the anatomical landmark?", reply with "none".
                    If this is not a 'find landmark' question, just respond to the original question"""
        #print(system_prompt)
        return str(system_prompt)
    

    # 3. --- Load dataset ---
    print("\n--- Loading Dataset ---")
    dataset = load_dataset(config.dataset_name, split=config.dataset_split)
    print(type(dataset))
    #Three columns in a dataset are created; default, with-prompt, and with-adaptive-prompt. This is not optimal and may change in the future.

    def format_and_resize_example(example: Dict) -> Dict:
        #Set up flag for removal
        example["remove"] = False
        #Set up blank messages in case of error
        example["messages"] = [
            {
                "role": "system",
                "content": [
                    {"type": "text","text":""}
                ]
            },
            {
                "role": "user",
                "content": [
                    {"type": "text","text":""},
                    {"type": "image"}, 
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text","text":""}
                ]
            }
        ] #This blank filler helps prevent dataset errors because the column still exists and has the same datatype
        example["messages-static"] = example["messages"]
        example["messages-adaptive"] = example["messages"] #If the script crashes early these columns would be undefined without this
        example["messages-answer"] = example["messages"]


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

        system_prompt_static = config.system_prompt
        system_prompt = generatePrompt(input_text = prompt)

        answer = example[config.answer_column]#for the last few examples in the Kvasir VQA dataset the 'answer' is the string "nan" but not actually nan, check for this?
        if answer == "nan": #Handles the "nan" case but not real nan
             print("Answer is Not a Number")
             example["remove"] = True
             return example

        example["messages"] = [
            {
                "role": "system",
                "content":[
                    {"type": "text","text":""}
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

        example["messages-static"] = [
            {
                "role": "system",
                "content": [
                    {"type": "text","text":system_prompt_static}
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

        example["messages-adaptive"] = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text":system_prompt}
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

        example["messages-answer"] = [
            {
                "role": "system",
                "content": [
                    {"type": "text","text":""}
                ]
            },
            {
                "role": "user",
                "content": [
                    {"type": "text","text":""},
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
        return example #Example has 3 versions of messages, one for each different model config.
    
    print("Mapping processing function to datasets (only resizing and message creation)...")
    original_features_to_remove = [col for col in [config.question_column, config.answer_column, "source", "img_id"] if col in dataset.features] #list comprehension to remove unnecessary columns

    #Load from dataset if the map is already completed:
    
    print("Existing, formatted dataset instance not found. Mapping dataset now.")    
    dataset = dataset.map(
        format_and_resize_example,
        num_proc=config.num_workers, #Use more than 1 worker for faster mapping, make sure memory is not overused
        remove_columns=original_features_to_remove,
        #keep_in_memory = True,

    )
        #Remove all rows for which remove = True
    dataset = dataset.filter(lambda example: example["remove"] is not True) #Only remove = False should remain
    print("Map completed. Dataset structure after initial processing:")
    print(dataset)
    print(f"Memory after initial data processing: {get_memory_usage()}")

    if config.train_split_name in dataset.features and config.eval_split_name in dataset.features: #Checks if dataset is already partitioned
        train_dataset = dataset[config.train_split_name]
        eval_dataset = dataset[config.eval_split_name]
        print("Dataset is already partitioned, skipping this step.")  
    else:
        dataset = dataset.train_test_split(test_size=config.test_size,shuffle = config.shuffle) #The train_test_split returns a partitioned dataset, with dictionary keys named "train" and "test"
        train_dataset = dataset["train"]
        eval_dataset = dataset["test"]
        print("Dataset partitioned into Train and Test splits.")
    
    torch.cuda.empty_cache()
    
    #Shrink train_dataset:
    train_dataset = train_dataset.select([0,1,2])

    print("\n--- Dataset loading completed ---")

    # 4. --- Load processor and model(s) ---
    print("\n--- Loading Processor ---")
    processor = AutoProcessor.from_pretrained(config.model_id, use_fast=True)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    
    print(f"EOS token: {processor.tokenizer.eos_token_id}")
    
    #Set up torch d_type
    if config.bf16 and config.fp16: 
        raise Warning("torch_dtype cannot be both bfloat16 and float 16, please adjust ScriptConfig values. torch_dtype has defaulted to bf16.")
        
    torch_dtype = torch.float32 #default, "else" case
    if config.bf16:
            torch_dtype = torch.bfloat16
    elif config.fp16:
            torch_dtype = torch.float16
    
    print("\n--- Configuring Quantization ---")
    quantization_config = BitsAndBytesConfig( 
        load_in_4bit = True, #essentially, load weights/tensors as 4 bit instead of 32, making training faster
        bnb_4bit_quant_type = "nf4", #more efficient version of 4 bit quantization, "normalized 4 bit floating point"
        bnb_4bit_compute_dtype = torch_dtype,
        bnb_4bit_use_double_quant = True, #improves memory usage by "0.5 bits/parameter", or ~0.25 GB total on MedGemma
    )

    
    print("\n--- Loading Models ---")
    def modelInit(tuned: bool):
        #set up model for medgemma 4b quantized
        model = AutoModelForImageTextToText.from_pretrained( #creates an instance of the model loaded with 4bit quantization
            "google/medgemma-4b-it", #model source
            quantization_config = quantization_config, #reduced memory usage by ~4.83 times
            torch_dtype = torch_dtype,
            device_map = "auto") 
        
        print("Model Loaded")
        
        if tuned:
            print(f"Loading and merging LoRA adapter from: {config.adapter_path}")
            model = PeftModel.from_pretrained(model, config.adapter_path)
            model = model.merge_and_unload()
            print("LoRA adapter merged successfully.")
        
        return model
        
    # 6. --- Configure Data Collator
    print("\n--- Initializing Data Collator ---")

    class CustomDataCollatorForVision2Seq:
        # Pass the max_sequence_length from config
        def __init__(self, processor, max_sequence_length: int,message_column_type): 
                self.processor = processor
                self.max_length = max_sequence_length # Use the fixed max_length from config
                self.message_column_type = message_column_type


        def __call__(self, example: Dict) -> Dict[str, torch.Tensor]: #Now designed for just batch size of 1 for eval for simplicity
            #print(f"message_column_type: {self.message_column_type}")
            inputs = self.processor.apply_chat_template(
                example[self.message_column_type],
                add_generation_prompt=False,
                tokenize=False,
            )
        
            batch_inputs = self.processor(
                text=inputs,
                images=[example[config.image_column]], # Pass the correctly formatted list of lists (with 1 element)
                return_tensors="pt",
                padding="longest",
                truncation=True, # Ensure truncation is enabled
                max_length=self.max_length # Explicitly set max_length from config
            )

            # Labels setup for causal LM
            labels = torch.full_like(batch_inputs["input_ids"], -100) #sets all labels to -100, a value not read by the model

            # Apply chat template specifically for the prompt part to determine its length
            # This ensures we mark only the assistant's response as targets for loss.
            answer_text = self.processor.tokenizer.apply_chat_template(
                example["messages-answer"], tokenize=False, add_generation_prompt=False
            )
            #print(answer_text)
            
            # Tokenize the prompt text to get its tokenized length
            labels = self.processor(
                text=answer_text,
                images=None, 
                return_tensors="pt",
                padding="longest",
                truncation=True, # Ensure truncation is enabled
                max_length=self.max_length
            )["input_ids"] #not same length as input, this is intentional
            
            batch_inputs["labels"] = labels
            gc.collect()
            return batch_inputs

    #7. --- Initialize Compute Metrics --- (If necessary)

    score = evaluate.load("bertscore")
    rouge = evaluate.load('rouge') #Rouge 2 is bigrams, or 2 words 
    accuracy = evaluate.load("accuracy")#accuracy is easier to visualize and more "meaningful" in the real world than loss

    def get_accuracy(inputs, labels):
        inputs = inputs.strip()
        labels = labels.strip()
        if inputs == labels:
            return 1.0
        else:
            return 0.0

    def averageScore(bert_precision,bert_recall,bert_f1,rouge_scores):
        #BertScore split into: precision, recall,f1
        average_precision = (sum(bert_precision))/(len(bert_precision))
        average_recall = (sum(bert_recall))/(len(bert_recall))
        average_f1 = (sum(bert_f1))/(len(bert_f1))
        average_rouge = (sum(rouge_scores))/(len(rouge_scores))
        return average_precision,average_recall,average_f1,average_rouge

    def evaluate_instance(tuned,eval_dataset, data_collator):
        model = modelInit(tuned=tuned)
        print("Evaluation Beginning")
        bert_precision = []
        bert_recall = []
        bert_f1 = []
        rouge_scores = []
        accuracies = []
        print(len(eval_dataset))
        for x in range (0,40):#len(eval_dataset)): #lowered for testing
            example = eval_dataset.select([x]) #get the row

            batch_inputs = data_collator(example)
            labels = batch_inputs["labels"]
            #print(f"labels {labels}")
            batch_inputs.pop("labels") #removes the answers
            model_inputs = batch_inputs
            #print(f"model_inputs, {model_inputs}")
            model_inputs.to(config.device)

            #Generate model output
            generation = model.generate(**model_inputs, max_new_tokens = 25, do_sample=False, repetition_penalty = 1.3)#MedGemma is verbose, but low tokens allow for faster evals

            #print(f"generation {generation}")
            prompt_length = model_inputs["input_ids"].shape[1]
            prediction = generation[:, prompt_length:] #Everything after prompt

            #print(f"Trimmed prediction: {prediction}")
            labels_list = labels.tolist()[0]
            prediction_list = prediction.tolist()[0]

            #Pad the end of labels to the same length
            if len(labels_list) < len(prediction_list):
                labels_list += [0]*(abs(len(prediction_list)-len(labels_list)))
            
            elif len(labels_list) > len(prediction_list): 
             prediction_list += [0]*(abs(len(prediction_list)-len(labels_list)))
            
            
            

            #BertScore requires strings not list of int.
            prediction = processor.batch_decode([prediction_list],skip_special_tokens=True)[0].strip()
            labels = processor.batch_decode([labels_list],skip_special_tokens=True)[0].strip()

            #remove "\n, ', and 'model' (which begins the label sequence)"
            prediction = (prediction.replace("model", "")).lower()
            labels = (labels.replace("user", ""))
            labels = (labels.replace("model", "")).lower()
            mapping_table = str.maketrans("","",".!?'`<>*\n")
        
            prediction = prediction.translate(mapping_table)
            labels = labels.translate(mapping_table)

            print(f"prediction {prediction},labels{labels}")

            rouge_score = rouge.compute(predictions=[prediction],
                references=[labels],                         
                rouge_types=['rougeL'],
                use_aggregator=True)
            
            rouge_scores.append(rouge_score["rougeL"])
            bert_score = score.compute(predictions=[prediction], references=[labels], lang="en")
            bert_precision.append(bert_score["precision"][0])
            bert_recall.append(bert_score["recall"][0])
            bert_f1.append(bert_score["f1"][0])

            eval_accuracy = get_accuracy(inputs = prediction, labels = labels)
            accuracies.append(eval_accuracy)

            print(bert_score,rouge_score)
            print(eval_accuracy)
            print("iter:",x)
            
            del prediction,labels,prediction_list,labels_list,generation
            gc.collect()
            
        average_precision,average_recall,average_f1,average_rouge = averageScore(bert_f1=bert_f1,bert_precision=bert_precision,bert_recall=bert_recall,rouge_scores=rouge_scores)
        average_accuracy = (sum(accuracies))/(len(accuracies))

        print(f"average_precision {average_precision},average_recall {average_recall},average_f1 {average_f1},average_accuracy {average_accuracy}, ROUGE {average_rouge}")
        del model
        gc.collect()
        torch.cuda.empty_cache()
        return average_precision,average_recall,average_f1,average_accuracy,average_rouge

    #8. --- Create trainer instances for model evaluation ---
    print("\n--- Initializing evaluation ---") #I may restrucure this into a for loop for clarity?
    
    average_precisions =[]
    average_recall_performances = []
    average_f1_scores = []
    accuracies = []
    average_rouge_scores = []
    message_column_type = "messages" #Set to default before evaluation
    data_collator = CustomDataCollatorForVision2Seq(processor, config.max_sequence_length,message_column_type=message_column_type)
    for i in range (0,4): #normally 6 but adaptive prompt not working yet
        torch.cuda.empty_cache()
        #Update column type at 2,4 for data collator
        if i == 2:
            del data_collator
            message_column_type = "messages-static" #This helps the data collator decide which column to grab data from
            data_collator = CustomDataCollatorForVision2Seq(processor, config.max_sequence_length,message_column_type=message_column_type)
        
        if i == 4:
            del data_collator
            message_column_type = "messages-adaptive" #This helps the data collator decide which column to grab data from
            data_collator = CustomDataCollatorForVision2Seq(processor, config.max_sequence_length,message_column_type=message_column_type)

        
        #If i is 1,3,5 set model to tuned
        if i % 2 == 0: #Divisible by 2 with no remainder, base
            print("Loading base model")
            average_precision,average_recall,average_f1,average_accuracy,average_rouge = evaluate_instance(eval_dataset=eval_dataset,data_collator=data_collator,tuned = False)
        
        else:
            print("Loading tuned model")
            average_precision,average_recall,average_f1,average_accuracy,average_rouge = evaluate_instance(eval_dataset=eval_dataset,data_collator=data_collator,tuned = True)

        #Evaluate then cleanup
        print(f"Model instance number {i} was evaluated. \nResults: {accuracies}")
        average_f1_scores.append(average_f1)
        average_rouge_scores.append(average_rouge)
        average_recall_performances.append(average_recall)
        average_precisions.append(average_precision)
        accuracies.append(average_accuracy)#List of accuracies

        torch.cuda.synchronize() #Ensures there is not some asynchronous mismatch where the trainer is deleted on CPU but not GPU or visa versa
        print(f"Memory after eval sequence {i}: {get_memory_usage()}")


    print("--- Evaluation Completed ---")

    # 9. --- Log and visualize results ---

    #Save results to a file for safekeeping
    with open("./medgemma-finetuned/eval.txt", "w") as model_evaluation:
        model_evaluation.write(f"Evaluation results: \nResults: {average_precisions,average_recall_performances,average_f1_scores,accuracies}")
        #As the other model configs are completed, add them to file writing process

    model_evaluation.close()
    #Set up keys
    models = ["Original","Tuned","Original-Static-Prompt","Tuned-Static-Prompt","Original-Adaptive-Prompt","Tuned-Adaptive-Prompt"]
    print(f"Model evaluation accuracy: {accuracies}, Model BERT scores (en) {average_precisions,average_recall_performances,average_f1_scores}")
    # data = {
    #     "Model":models,
    #     "Score":average_f1_scores
    # }

    print(data)
    # #Display results via seaborn
    # seaborn.set_style('darkgrid')
    # plot = seaborn.barplot(data = data, x = "Model", y = "Accuracy", fill = True, palette= "muted")
    # plot.set_title("Model configuration vs accuracy")
    # plt.show()
    # plt.savefig("ModelAccuracy.png")

if __name__ == "__main__":
    main()
