"""
Example code for medgemma tuner
"""
from image_text_to_text_tuner import tuner, evaluator, gguf_conversion

config = tuner.ScriptConfig()
config.dataset_name = "flaviagiammarino/vqa-rad" #Radiology Dataset (NOTE: MedGemma was trained on this field already, so>
config.epochs = 1 #Or any other setting change
config.tune_hyperparameters = False
config.use_pretrained_adapter = False
config.hf_token = "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"

fine_tuner = tuner.fine_tune(config)
fine_tuner.tune()

eval_config = evaluator.EvalConfig()
eval_config.medgemma_tuned_path = "./medgemma-finetuned/v3"
model_evaluator = evaluator.evaluate_model(eval_config)
model_evaluator.evaluate()

#Tuning complete, model saved to a file, eval saved to a file
#gguf_config = gguf_conversion.ConverterConfig()
#gguf_config.model_name = "test_model"
#gguf_config.adapter_path = "./medgemma-finetuned/v3"

#converter = gguf_conversion.converter(gguf_config)
#converter.convert()
