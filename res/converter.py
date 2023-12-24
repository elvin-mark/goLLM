import torch
import os


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


if __name__ == "__main__":
    conv_gpt2_model()
