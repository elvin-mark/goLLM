import torch
import os
import sys


def conv_gpt2_model():
    model_path = os.environ["GPT2_MODEL_PATH"]
    converted_model_path = os.environ["GPT2_CONVERTED_MODEL_PATH"]

    model = torch.load(model_path)
    n_layers = 12
    with open(converted_model_path, "wb") as f:
        f.write(model["wte.weight"].numpy().tobytes())
        f.write(model["wpe.weight"].numpy().tobytes())
        for i in range(n_layers):
            f.write(model[f"h.{i}.ln_1.weight"].numpy().tobytes())
            f.write(model[f"h.{i}.ln_1.bias"].numpy().tobytes())
            f.write(model[f"h.{i}.attn.c_attn.weight"].numpy().tobytes())
            f.write(model[f"h.{i}.attn.c_attn.bias"].numpy().tobytes())
            f.write(model[f"h.{i}.ln_2.weight"].numpy().tobytes())
            f.write(model[f"h.{i}.ln_2.bias"].numpy().tobytes())
            f.write(model[f"h.{i}.attn.c_proj.weight"].numpy().tobytes())
            f.write(model[f"h.{i}.attn.c_proj.bias"].numpy().tobytes())
            f.write(model[f"h.{i}.mlp.c_fc.weight"].numpy().tobytes())
            f.write(model[f"h.{i}.mlp.c_fc.bias"].numpy().tobytes())
            f.write(model[f"h.{i}.mlp.c_proj.weight"].numpy().tobytes())
            f.write(model[f"h.{i}.mlp.c_proj.bias"].numpy().tobytes())
        f.write(model["ln_f.weight"].numpy().tobytes())
        f.write(model["ln_f.bias"].numpy().tobytes())


def conv_rwkv_model():
    model_path = os.environ["RWKV_MODEL_PATH"]
    converted_model_path = os.environ["RWKV_CONVERTED_MODEL_PATH"]

    model = torch.load(model_path, map_location="cpu")
    n_layers = 12
    with open(converted_model_path, "wb") as f:
        f.write(model["emb.weight"].type(torch.float32).numpy().tobytes())
        f.write(model[f"blocks.0.ln0.weight"].type(
            torch.float32).numpy().tobytes())
        f.write(model[f"blocks.0.ln0.bias"].type(
            torch.float32).numpy().tobytes())
        for i in range(n_layers):
            f.write(model[f"blocks.{i}.ln1.weight"].type(
                torch.float32).numpy().tobytes())
            f.write(model[f"blocks.{i}.ln1.bias"].type(
                torch.float32).numpy().tobytes())
            f.write(model[f"blocks.{i}.att.time_decay"].type(
                torch.float32).numpy().tobytes())
            f.write(model[f"blocks.{i}.att.time_first"].type(
                torch.float32).numpy().tobytes())
            f.write(model[f"blocks.{i}.att.time_mix_k"].type(
                torch.float32).numpy().tobytes())
            f.write(model[f"blocks.{i}.att.time_mix_v"].type(
                torch.float32).numpy().tobytes())
            f.write(model[f"blocks.{i}.att.time_mix_r"].type(
                torch.float32).numpy().tobytes())
            f.write(model[f"blocks.{i}.att.key.weight"].type(
                torch.float32).numpy().tobytes())
            f.write(model[f"blocks.{i}.att.value.weight"].type(
                torch.float32).numpy().tobytes())
            f.write(
                model[f"blocks.{i}.att.receptance.weight"].type(torch.float32).numpy().tobytes())
            f.write(model[f"blocks.{i}.att.output.weight"].type(
                torch.float32).numpy().tobytes())
            f.write(model[f"blocks.{i}.ln2.weight"].type(
                torch.float32).numpy().tobytes())
            f.write(model[f"blocks.{i}.ln2.bias"].type(
                torch.float32).numpy().tobytes())
            f.write(model[f"blocks.{i}.ffn.time_mix_k"].type(
                torch.float32).numpy().tobytes())
            f.write(model[f"blocks.{i}.ffn.time_mix_r"].type(
                torch.float32).numpy().tobytes())
            f.write(model[f"blocks.{i}.ffn.key.weight"].type(
                torch.float32).numpy().tobytes())
            f.write(
                model[f"blocks.{i}.ffn.receptance.weight"].type(torch.float32).numpy().tobytes())
            f.write(model[f"blocks.{i}.ffn.value.weight"].type(
                torch.float32).numpy().tobytes())
        f.write(model["ln_out.weight"].type(torch.float32).numpy().tobytes())
        f.write(model["ln_out.bias"].type(torch.float32).numpy().tobytes())
        f.write(model["head.weight"].type(torch.float32).numpy().tobytes())


if __name__ == "__main__":
    if sys.argv[1] == "gpt2":
        conv_gpt2_model()
    elif sys.argv[1] == "rwkv":
        conv_rwkv_model()
