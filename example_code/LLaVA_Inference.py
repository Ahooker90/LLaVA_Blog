import torch
from transformers import AutoTokenizer, AutoProcessor, TrainingArguments, LlavaForConditionalGeneration, BitsAndBytesConfig
from trl import SFTTrainer
from peft import LoraConfig
import json
from PIL import Image 

model_id = "llava-hf/llava-1.5-7b-hf"
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
)
model = LlavaForConditionalGeneration.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)
processor = AutoProcessor.from_pretrained(model_id)
processor.tokenizer = tokenizer

test_file = open('test_data.json')
test_data = json.load(test_file)
print(len(test_data))
texts = test_data[0]['conversations'][0]['value']
images = Image.open(test_data[0]['image'][1:])
inputs = processor(texts, images, return_tensors="pt", padding=True, truncation=True)
out = model.generate(**inputs, max_new_tokens=3)
decoded_output = tokenizer.decode(out[0], skip_special_tokens=True)
print(decoded_output)







































