import os_mirror.os_mirror 
from transformers import AutoTokenizer, AutoModelForCausalLM
import chunk_traditional as chunk_t
import embedding 



tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
LLM_MODEL = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B"
                                                 ,torch_dtype="auto",
                                                  device_map="auto")

query = "电动平衡车的安全要求是什么？"
chunks: list[str] = embedding.query_db(query)
prompt = "Please answer the question based on the following context:\n"
prompt += f"Query: {query}\n"
prompt += "Context:\n"   
for c in chunks:
    prompt += c + "\n"
    prompt += "-----\n"
       
messages = [{"role": "user", "content": prompt}]    
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True # Switches between thinking and non-thinking modes. Default is True.
)
# Generate text using the language model. LLM_MODEL.device is the device to use for the model.
model_inputs = tokenizer([text], return_tensors="pt").to(LLM_MODEL.device)
# conduct text completion
generated_ids = LLM_MODEL.generate(
    **model_inputs,
    max_new_tokens=32768
)
# Extract the generated token IDs, excluding the input prompt tokens
output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

# parsing thinking content
try:
    # rindex finding 151668 (</think>)
    index = len(output_ids) - output_ids[::-1].index(151668)
except ValueError:
    index = 0

thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

print(f"Thinking content: {thinking_content}")
print(f"Content: {content}")