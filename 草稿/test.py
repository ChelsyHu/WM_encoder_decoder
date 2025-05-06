import torch
import transformers
from transformers import LlamaForCausalLM, LlamaTokenizer


model_dir = "/ssd-sata1/hqq/Llama-2-7b-hf"
model = LlamaForCausalLM.from_pretrained (model_dir)

tokenizer = LlamaTokenizer.from_pretrained (model_dir)

pipeline = transformers.pipeline (
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.float16,device_map="auto",
)

text1="Zoky"
vector1= tokenizer.encode(text1)

vector2=vector1 
text2=tokenizer.decode(vector2)


sequences = pipeline (
    'I have tomatoes, basil and cheese at home. What can I cook for dinner?\n',
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,max_length=80,
)
for seq in sequences:
    print (f"{seq ['generated_text']}")