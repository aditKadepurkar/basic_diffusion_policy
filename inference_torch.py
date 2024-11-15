import torch
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from eval_torch import eval_policy
from diffusion.data_loader import DataLoader


# Load the model
model_path = 'model.pt'
model = torch.load(model_path)

noise_scheduler = DDPMScheduler(
    num_train_timesteps=50,
    beta_schedule='squaredcos_cap_v2',
    clip_sample=True,
    prediction_type='epsilon'
)

dataloader = DataLoader(
    file_path="demonstrations/1731007425_4282627/demo.hdf5",
    dataset_name="data", 
    batch_size=128
)

eval_policy(model=model, noise_scheduler=noise_scheduler, stats=dataloader.stats)

print("Model loaded successfully from", model_path)