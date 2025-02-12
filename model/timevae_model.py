from vae.vae_utils import instantiate_vae_model, load_vae_model
import os

def initialize_timevae_model(cfg, shape):
    if not cfg["pretrain_path"] is None and os.path.isdir(cfg["pretrain_path"]):
        print("loading vae model...")
        return load_vae_model("timeVAE", cfg["pretrain_path"])
    else:
        print("creating new vae model...")
        return instantiate_vae_model("timeVAE", shape[0], shape[1], 64, hidden_layer_sizes = [64, 32], latent_dim = 64)