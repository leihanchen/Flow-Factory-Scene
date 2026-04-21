import argparse
import os

import torch
from diffusers import DiffusionPipeline
from peft import PeftModel
from safetensors.torch import save_file as save_safetensors_file

def parse_args():
    parser = argparse.ArgumentParser(description="Merge PEFT LoRA weights into a full Diffusers model.")
    parser.add_argument(
        "--base-model-id",
        default="Wan-AI/Wan2.1-I2V-14B-720P-Diffusers",
        help="Base model ID or local path.",
    )
    parser.add_argument(
        "--adapter-dir",
        default="./saves/wan2_i2v_lora_nft_20260322_130505/checkpoints/checkpoint-120",
        help="Path to PEFT adapter directory (contains adapter_config.json).",
    )
    parser.add_argument(
        "--output-dir",
        default="./saves/wan2_i2v_lora_nft_20260322_130505/checkpoints/checkpoint-120/full_model",
        help="Output directory for merged full model.",
    )
    parser.add_argument(
        "--output-style",
        default="diffusers",
        choices=["diffusers", "weights", "both"],
        help=(
            "Export style: 'diffusers' saves full Diffusers pipeline; "
            "'weights' saves merged transformer as .pth and .safetensors; "
            "'both' saves both formats."
        ),
    )
    return parser.parse_args()


args = parse_args()
base_model_id = args.base_model_id
adapter_dir = args.adapter_dir
output_dir = args.output_dir
output_style = args.output_style

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

os.makedirs(output_dir, exist_ok=True)

if output_style in ["diffusers", "both"]:
    print(f"Saving Diffusers full merged model to {output_dir}...")
    # Save the entire pipeline in the standard Diffusers folder structure.
    pipe.save_pretrained(output_dir)

if output_style in ["weights", "both"]:
    print(f"Saving merged transformer weights (.pth + .safetensors) to {output_dir}...")
    # Persist standalone transformer weights in common checkpoint formats.
    state_dict = merged_transformer.state_dict()
    torch.save(state_dict, os.path.join(output_dir, "diffusion_pytorch_model.pth"))
    save_safetensors_file(state_dict, os.path.join(output_dir, "diffusion_pytorch_model.safetensors"))

print("Done! You now have a standalone Diffusers model.")
