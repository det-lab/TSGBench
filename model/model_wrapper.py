import os

class ModelWrapper:
    def __init__(self, model_type, model, cfg = None, data = None):
        self.model_type = model_type
        self.model = model
        self.cfg = cfg
        self.data = data
    def sample_data(self, shape):
        if self.model_type == "timeVAE":
            return self.model.get_prior_samples(shape)
        if self.model_type == "timeGAN":
            if self.cfg["load"]:
                return self.model(self.data, self.cfg, self.cfg['pretrain_path'])
            return self.model(self.data, self.cfg)

def initialize_timevae_model(cfg, shape):
    # imports moved into function, as they somehow conflict with timegan
    from vae.vae_utils import instantiate_vae_model, load_vae_model
    
    if not cfg["pretrain_path"] is None and os.path.isdir(cfg["pretrain_path"]):
        print("loading vae model...")
        return ModelWrapper("timeVAE", load_vae_model("timeVAE", cfg["pretrain_path"]))
    else:
        print("creating new vae model...")
        return ModelWrapper("timeVAE", instantiate_vae_model("timeVAE", shape[0], shape[1], 64, hidden_layer_sizes = cfg['hidden_layer'], latent_dim = cfg['latent_dim']))

def initialize_timegan_model(cfg, data):
    # import timegan functions; they conflict with vae somehow, so import in here
    from timegan.timegan import run_timegan_fresh
    from timegan.timegan_old import load_timegan
    
    # Get cfg paramaters
    cfg['hidden_dim'] = cfg['latent_dim']
    cfg['num_layer'] = len(cfg['hidden_layer'])
    cfg["iterations"] = 1000
    cfg["batch_size"] = 64
    cfg["module"] = "lstm"
    
    # TODO Loading pre-trained
    if not cfg["pretrain_path"] is None and not cfg["pretrain_path"] == "None":
        print("loading timeGAN model...")
        cfg["load"] = True
        model = load_timegan
    else:
        print("creating new timeGAN model...")
        cfg["load"] = False
        model = run_timegan_fresh
    return ModelWrapper("timeGAN", model, cfg, data)
    