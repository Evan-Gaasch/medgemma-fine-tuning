#Load model, save model as combined for conversion to gguf format
from transformers import AutoModelForImageTextToText,AutoProcessor,BitsAndBytesConfig
from dataclasses import dataclass
from typing import Optional
from huggingface_hub import login,hf_hub_download
from peft import PeftModel
import torch,subprocess
from safetensors.torch import save_model
import shutil

@dataclass
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

print("Beginning GGUF conversion")

class converter:
    def __init__(self,config):
        self.config = config
    
    def modelInit(self, tuned: bool):
        #set up model for medgemma 4b quantized
        model = AutoModelForImageTextToText.from_pretrained( #creates an instance of the model loaded with 4bit quantization
            "google/medgemma-4b-it", #model source
            #quantization_config = quantization_config,#quantization_config, #reduced memory usage by ~4.83 times
            torch_dtype = torch.float16,
            #device_map="cpu",
            low_cpu_mem_usage=False,  
            _fast_init=False,
            #device_map = "auto") 
        )
        
        print("Model Loaded")
        
        if tuned:
            print(f"Loading and merging LoRA adapter from: {self.config.adapter_path}")
            model = PeftModel.from_pretrained(model, self.config.adapter_path)
            model = model.merge_and_unload()
            print("LoRA adapter merged successfully.")
            print(model)
        
        return model

    def convert(self):
        processor = AutoProcessor.from_pretrained(self.config.model_id, use_fast=True)
        if processor.tokenizer.pad_token is None:
            processor.tokenizer.pad_token = processor.tokenizer.eos_token


        model = self.modelInit(tuned = True).to("cpu") #prevents tied weights error

        if hasattr(model, 'tie_weights'):
            model.tie_weights()
            print("weights tied")
        else:
            print("attr not found")

        print("Model loaded")
        model.save_pretrained(self.config.output_dir, safe_serialization = True)
        print("Model Saved")
        processor.save_pretrained(self.config.processor_dir)
        print("Processor saved")

        # #Save tokenizer.model 
        cached_path = hf_hub_download(self.config.model_id,"tokenizer.model")
        cached_config = hf_hub_download(self.config.model_id,"config.json")
        print("Beginning copying")
        shutil.copyfile(cached_path,f"{self.config.processor_dir}/tokenizer.model")
        shutil.copyfile(cached_config,f"{self.config.processor_dir}/config.json") #Necessary for conversion

        print("Starting subprocess")
        result = subprocess.run(
            f"{self.config.venv_python_path} ./llama.cpp/convert_hf_to_gguf.py {self.config.output_dir} --outfile mmproj_{self.config.model_name}.gguf --mmproj --outtype f16 --verbose",
            capture_output=True,
            shell = True
        )
        print(result)
        result = subprocess.run(
            f"{self.config.venv_python_path} ./llama.cpp/convert_hf_to_gguf.py {self.config.output_dir} --outfile {self.config.model_name}.gguf --outtype f16 --verbose",
            capture_output=True,
            shell = True
        )
        print(result)
