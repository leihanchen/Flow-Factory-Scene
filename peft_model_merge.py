import torch
from diffusers import DiffusionPipeline
from peft import PeftModel

# 1. Define your paths
base_model_id = "Wan-AI/Wan2.1-I2V-14B-720P-Diffusers" # Change to your specific base model
adapter_dir = "./saves/wan2_i2v_lora_nft_20260322_130505/checkpoints/checkpoint-120"         # Folder with adapter_config.json
output_dir = "./saves/wan2_i2v_lora_nft_20260322_130505/checkpoints/checkpoint-120/full_model"         # Where the new full model will go

print("Loading the base pipeline...")
# Load pipeline (use bfloat16 or float16 depending on what you trained in)
pipe = DiffusionPipeline.from_pretrained(
    base_model_id, 
    torch_dtype=torch.bfloat16
)

print("Applying PEFT adapter to the Transformer...")
# Wan2.1 is a Diffusion Transformer (DiT). PEFT targets the transformer component.
# This wraps the base transformer with your LoRA weights.
transformer_with_peft = PeftModel.from_pretrained(pipe.transformer, adapter_dir)

print("Fusing weights (this might take a moment)...")
# merge_and_unload() permanently adds the LoRA matrix to the base weights 
# and returns a standard, un-wrapped PyTorch module.
merged_transformer = transformer_with_peft.merge_and_unload()

# Replace the pipeline's transformer with your newly merged one
pipe.transformer = merged_transformer

print(f"Saving the full merged model to {output_dir}...")
# Save the entire pipeline in the standard Diffusers folder structure
pipe.save_pretrained(output_dir)

print("Done! You now have a standalone Diffusers model.")
