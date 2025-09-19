# -*- coding: utf-8 -*-
"""
Inference Script for Fine-Tuned MedGemma Model.

This script loads a quantized MedGemma model (either the base model or a
fine-tuned LoRA adapter) and performs inference on a given image and text prompt.

Key Features:
- Loads 4-bit quantized model for memory efficiency.
- Merges LoRA adapters if a path is provided.
- Accepts command-line arguments for image path, prompt, and adapter path.
- Simple, clean, and focused on the core inference task.

Usage:
  # Base model inference
  python medgemma_inference_optimized.py --image_path /path/to/your/image.jpg --prompt "Your question here"

  # Inference with a fine-tuned LoRA adapter
  python medgemma_inference_optimized.py --image_path /path/to/your/image.jpg --prompt "Your question here" --adapter_path /path/to/your/lora_adapter
"""

import os
import torch
import PIL.Image
import numpy as np
import io
import argparse
from dataclasses import dataclass, field
from typing import Optional,List

# Dependency for loading .env files
from dotenv import load_dotenv

# Hugging Face Libraries
from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText, # Changed from PaliGemmaForConditionalGeneration
    BitsAndBytesConfig
)
from peft import PeftModel
from datasets import load_dataset
from huggingface_hub import login


# --- Configuration ---
@dataclass
class InferenceConfig:
    """Stores configuration settings for the inference script."""
    # Model and Tokenizer
    base_model_id: str = "google/medgemma-4b-it"
    adapter_path: str = "./medgemma-finetuned"  # Path to the fine-tuned LoRA adapter
    hf_token: Optional[str] = None  # Loaded from .env

    # Inference Parameters
    torch_dtype: torch.dtype = torch.bfloat16
    max_new_tokens: int = 150
    image_size = 896,896
    use_adaptive: bool = True
    use_prompt: bool = True

    # System Prompt
    system_prompt: str = (
        """You are a leading medical expert in the domain of colonoscopy and analyze medical imagery with high accuracy,
      identifying all anomalies and answering questions factually, and concisely. Note that 'finding' is generally code for abnormality.
      Respond in a concise manner; for example, when given an image of ulcerative colitis and a prompt "Are there any 
      abnormalities in the image?", respond with "Ulcerative Colitis" """
    )

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

# --- Core Components ---

class AdaptivePrompt:
    """ Handles adaptive prompting behaviors"""
    def __init__(self, config: InferenceConfig):
        self.config = config
    
    def generatePrompt(self,input_text: str):
        config = self.config
        system_prompt = config.system_prompt

        if not config.use_prompt:
            return ""

        if not config.use_adaptive:
            return system_prompt #(default)
        
        else: #Adaptive prompting: (keyword search)
            base_input = (input_text).lower()
            #Remove punctuation
            mapping_table = str.maketrans("","",".!?")
            
            decomposed_input = base_input.translate(mapping_table) #Removes punctuation
            decomposed_input = decomposed_input.split(" ") #split into words
            
            print(decomposed_input)

            classifications = [] #A list of what categories should be added to prompt, just to keep track of what has been added and prevent redundant prompts
            #Check each category ("brute-force")
            for word in decomposed_input: #there may be a more efficient way to check than O(N) time complexity or look more elegant, I will work on this.
                if word in config.instrument_keywords and "instrument" not in classifications:
                    classifications.append("instrument")
                    system_prompt += """ If you are answering a question about an instrument, screen display, etc, mimic the following answer format: Given a colonoscopy image and the prompt "What type of procedure is the image taken from? respond with "colonoscopy" """
                
                if word in config.determine_type_keywords and "type" not in classifications:
                    classifications.append("type")
                    system_prompt += """ Focus on the type of anomalies or attributes in the image. For example, when given an image of a colonoscopy with no polyps and the prompt
                        "What type of polyp is present?", respond with "None" """
                
                if word in config.measure_keywords and "measure" not in classifications:
                    classifications.append("measure")
                    system_prompt += """ Focus on the size, position, and quantitative attributes of the object you are being asked about. For example, given an image of 
                        a polyp and the prompt "What is the size of the polyp?", respond with "11-20mm", or whatever size-range the polyp could be. """
                
                if word in config.image_keywords and "find" not in classifications:
                    classifications.append("find")
                    system_prompt += """ Find the target landmark or feature in the image, and say "none" if it does not exist. Ensure that you scan the image thoroughly. For example,
                        if given an image that contains 2 abnormalities and a prompt of "Where in the image is the abnormality?", respond with "center; center-left;", if that
                        is where the abnormality is. """
                
                if word in config.color_keywords and "color" not in classifications:
                    classifications.append("color")
                    system_prompt += """ Find the color of the target feature or landmark. Ensure that you scan the image thoroughly. For example,
                        if given an image that contains 2 abnormalities and a prompt of "What color is the abnormality? If more than one separate with ;", 
                        respond with "pink; red", if those are the colors of the abnormalities"""
                    
                if word in config.counting_keywords and "counting" not in classifications:
                    classifications.append("counting")
                    system_prompt += """ Carefully count the number of target features in the image and reply concisely. For example, when given
                        an image of the GI tract with only ulcerative colitis and the prompt "How many findings are present?", respond with "1"."""

                if word in config.landmark_keywords and "landmark" not in classifications:
                    classifications.append("landmark")
                    system_prompt += """ Carefully find the location of all important anatomical features in the image. For example, when given
                        an image of the GI tract with no landmarks present and the prompt "Where in the image is the anatomical landmark?", reply with "none" """
            print(system_prompt)
        return system_prompt
        

class MedGemmaInference:
    """A class to handle MedGemma model loading and inference."""

    def __init__(self, config: InferenceConfig):
        """
        Initializes the inference engine.

        Args:
            config: An InferenceConfig object with all necessary settings.
        """
        self.config = config
        self.model = None
        self.processor = None
        self._setup()

    def _login_to_hub(self):
        """Logs into the Hugging Face Hub using the provided token."""
        if not self.config.hf_token:
            print("Warning: Hugging Face token not found. Proceeding without authentication.")
            return
        print("Hugging Face token found. Logging in.")
        login(token=self.config.hf_token)

    def _setup(self):
        """
        Sets up the tokenizer, and loads the quantized model and adapter.
        """
        self._login_to_hub()

        print("\n--- Loading Model and Processor ---")

        # Configure 4-bit quantization for memory efficiency
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=self.config.torch_dtype,
            bnb_4bit_use_double_quant=True,
        )

        # Load the processor
        self.processor = AutoProcessor.from_pretrained(
            self.config.base_model_id,
            use_fast=True,
            trust_remote_code=True
        )
        if self.processor.tokenizer.pad_token is None:
            self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token

        # Load the base model with quantization using AutoModelForImageTextToText
        self.model = AutoModelForImageTextToText.from_pretrained( # Changed here
            self.config.base_model_id,
            quantization_config=quantization_config,
            torch_dtype=self.config.torch_dtype,
            device_map="auto",
            trust_remote_code=True,
        )

        # If an adapter path is provided, load and merge the LoRA weights
        if self.config.adapter_path:
            print(f"Loading and merging LoRA adapter from: {self.config.adapter_path}")
            self.model = PeftModel.from_pretrained(self.model, self.config.adapter_path)
            self.model = self.model.merge_and_unload()
            print("LoRA adapter merged successfully.")

        print(f"Model '{self.config.base_model_id}' loaded successfully.")
        print(f"Memory Usage: {self.get_memory_usage()}")

    def run_inference(self, prompt: str, image, system_prompt: str) -> str:
        """
        Runs inference on a single image and text prompt.

        Args:
            prompt: The user's text question.
            image_path: The file path to the user's image.

        Returns:
            The model's generated response as a string.
        """
        if not self.model or not self.processor:
            raise RuntimeError("Model and processor must be initialized before running inference.")

        try:
            if not isinstance(image, PIL.Image.Image):
                if isinstance(image, dict) and "bytes" in image:
                    image = PIL.Image.open(io.BytesIO(image["bytes"])).convert("RGB") 
                else:
                    # Attempt to convert other formats to PIL Image
                    image = PIL.Image.fromarray(np.array(image)).convert("RGB") 
            
            image = image.convert("RGB").resize(InferenceConfig.image_size, resample=PIL.Image.Resampling.LANCZOS)

        except Exception as e:
            return f"Error: Could not open or process image. Details: {e}"

        # Correctly process inputs in a two-step manner for chat template
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image"}]},
        ]

        # Apply the chat template to get the formatted string.
        # Ensure add_generation_prompt is True for inference
        prompt_str = self.processor.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Pass the formatted string and image to the processor.
        # The processor expects images as a list
        model_inputs = self.processor(
            text=prompt_str,
            images=[image], # Changed image to [image]
            return_tensors="pt"
        ).to(self.model.device)

        # Generate the response
        print("\n--- Generating Response ---")
        input_len = model_inputs["input_ids"].shape[-1]
        with torch.no_grad():
            generation_output = self.model.generate(
                **model_inputs,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=False
            )
            
        # Decode only the newly generated tokens for a clean response
        generated_tokens = generation_output[0][input_len:]
        response = self.processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
        return response.strip()

    @staticmethod
    def get_memory_usage() -> str:
        """Returns the current GPU memory usage in GB."""
        if not torch.cuda.is_available():
            return "N/A (CUDA not available)"
        allocated = torch.cuda.memory_allocated(0) / (1024**3)
        total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        free_memory = (torch.cuda.mem_get_info(device=0)[0])/ (1024**3) #includes other processes

        utilized = total-free_memory
        return (f"Utilized: {utilized:.2f} GB / {total:.2f} GB \nAllocated by PyTorch: {allocated:.2f} GB / {total:.2f}")


def retrieve_arguments():
    #retrieve from frontend

    #Test:
    #For testing, used Kvasir VQA dataset image
    dataset = load_dataset("SimulaMet-HOST/Kvasir-VQA", split = "raw[0:10]") #only load 1 sample
    image = dataset["image"][4]
    user_prompt = dataset["question"][4]

    return user_prompt,image


def main():
    """Main function to execute the inference script."""
    args = retrieve_arguments()
    
    # Load environment variables from a .env file
    load_dotenv()

    # Initialize configuration
    config = InferenceConfig(
        adapter_path=InferenceConfig.adapter_path,
        hf_token=os.getenv("HF_TOKEN")
        
    )

    # Create inference engine and run
    try:
        inference_engine = MedGemmaInference(config)
        promptGen = AdaptivePrompt(config=config)
        prompt,image = retrieve_arguments()
        response = inference_engine.run_inference(prompt= prompt, image = image, system_prompt=promptGen.generatePrompt(input_text = prompt))
        
        print("\n--- Model Response ---")
        print(response)

    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")


if __name__ == "__main__":
    main()
