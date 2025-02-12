from src.preprocess import preprocess_data, load_preprocessed_data
from vae.vae_utils import instantiate_vae_model, train_vae, save_vae_model

cfg = {"original_data_path":"./data/original_data/exchange_rate.csv", "seq_length":125}
data = preprocess_data((cfg))
print(data)
train_data = data[0]
print(train_data.shape)
model = instantiate_vae_model("timeVAE", data[0][0].shape[0], data[0][0].shape[1], 46, hidden_layer_sizes = [64, 32], latent_dim = 64)
print("begin training")
train_vae(model, data[0], 1000, verbose=1)
print("end training")
save_vae_model(model, "./saved_models/timevae")
print("model successfully saved")