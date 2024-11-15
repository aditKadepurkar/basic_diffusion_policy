from diffusion.pytorch_everything import ConditionalUnet1D, get_resnet, replace_bn_with_gn
import torch
from diffusion.data_loader_new import DataLoaderNew
import numpy as np
import torch.nn as nn
import tqdm
from diffusers.optimization import get_scheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusion.data_loader import DataLoader
from eval_torch import eval_policy


def test_conditional_unet1d():
    dataloader = DataLoaderNew(batch_size=10)

    model = ConditionalUnet1D(8, 1054)
    model = model.to('cuda:0')

    vision_encoder = get_resnet('resnet18')
    vision_encoder = replace_bn_with_gn(vision_encoder)
    vision_encoder = vision_encoder.to('cuda:0')
    
    # x = torch.randn(2, 20, 8)
    timestep = torch.zeros((10,)).float()
    # print(timestep.shape)
    cond = torch.randn(2, 30)

    data = next(dataloader)

    print(data['states'].shape, data['actions'].shape, data['visual'].shape)

    dlpack_capsule = data['actions'].__dlpack__()
    x = torch.from_dlpack(dlpack_capsule).float()

    # send the visual(ie: the image) through a resnet
    dlpack_capsule = data['visual'].__dlpack__()
    im = torch.from_dlpack(dlpack_capsule).float()

    image_features = vision_encoder(im.flatten(end_dim=1))
    image_features = image_features.reshape(*im.shape[:2],-1)
    print(image_features.shape)

    dlpack_capsule = data['states'].__dlpack__()
    agent_pos = torch.from_dlpack(dlpack_capsule).float()
    obs = torch.cat([image_features, agent_pos],dim=-1)

    timestep = timestep.to('cuda:0')


    print(x.shape, obs.flatten(start_dim=1).shape, timestep.shape)
    out = model(x, timestep, obs.flatten(start_dim=1))
    assert out.shape == (10, 8, 8)

    print("Test passed")


def train():
    epochs = 100
    
    vision_encoder = get_resnet('resnet18')
    vision_encoder = replace_bn_with_gn(vision_encoder)
    vision_encoder = vision_encoder.to('cuda:0')

    noise_pred_net = ConditionalUnet1D(4, 64)
    noise_pred_net = noise_pred_net.to('cuda:0')

    
    nets = nn.ModuleDict({
        # 'vision_encoder': vision_encoder,
        'noise_pred_net': noise_pred_net
    })

    noise_scheduler = DDPMScheduler(
        num_train_timesteps=50,
        beta_schedule='squaredcos_cap_v2',
        clip_sample=True,
        prediction_type='epsilon'
    )

    # dataloader = DataLoaderNew(batch_size=10)
    dataloader = DataLoader(
        file_path="demonstrations/1731007425_4282627/demo.hdf5",
        dataset_name="data", 
        batch_size=128
    )
    
    stats = dataloader.stats

    optimizer = torch.optim.AdamW(
        params=nets.parameters(),
        lr=1e-4, weight_decay=1e-6
    )

    # Cosine LR schedule with linear warmup
    lr_scheduler = get_scheduler(
        name='cosine',
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=dataloader.get_batch_count() * epochs
    )


    with tqdm.tqdm(total=epochs, desc="Epoch") as poch:
        for e in range(epochs):
            epoch_loss = []
            # Iterate over batches
            with tqdm.tqdm(dataloader, desc="Batches", total=dataloader.get_batch_count(), leave=False) as data_iter:
                for data in data_iter:
            
                    agent_obs = data['states']
                    actions = data['actions']
                    # im = data['visual']

                    # print(agent_obs.shape, actions.shape, im.shape)

                    # convert to torch tensors
                    # agent_obs = torch.tensor(agent_obs).to('cuda:0')
                    # actions = torch.tensor(actions).to('cuda:0')
                    # im = torch.tensor(im).to('cuda:0')

                    dlpack_capsule = data['states'].__dlpack__()
                    agent_obs = torch.from_dlpack(dlpack_capsule).float().to('cuda:0')

                    dlpack_capsule = data['actions'].__dlpack__()
                    actions = torch.from_dlpack(dlpack_capsule).float().to('cuda:0')

                    # send the visual(ie: the image) through a resnet
                    # dlpack_capsule = data['visual'].__dlpack__()
                    # im = torch.from_dlpack(dlpack_capsule).float()

                    # image_features = vision_encoder(im.flatten(end_dim=1))
                    # image_features = image_features.reshape(*im.shape[:2],-1)

                    # obs_features = torch.cat([image_features, agent_obs], dim=-1)
                    # obs_cond = obs_features.flatten(start_dim=1)


                    noise = torch.randn(actions.shape, device="cuda:0")

                    B = actions.shape[0]

                    timesteps = torch.randint(
                        0, noise_scheduler.config.num_train_timesteps,
                        (B,), device="cuda:0"
                    ).long()

                    noisy_actions = noise_scheduler.add_noise(
                        actions, noise, timesteps)

                    # print(noisy_actions.shape, timesteps.shape, agent_obs.shape)
                    noise_pred = noise_pred_net(
                        noisy_actions, timesteps, global_cond=agent_obs)

                    loss = nn.functional.mse_loss(noise_pred, noise)

                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                    lr_scheduler.step()

                    loss_val = loss.item()
                    epoch_loss.append(loss_val)

                    data_iter.set_postfix(loss=loss_val)
            poch.update(1)
            poch.set_postfix(loss=np.mean(epoch_loss))
            dataloader.shuffle_data()
            # print(f"Epoch: {e+1}, Loss: {epoch_loss/dataloader.get_batch_count()}")
            # print(f"Learning Rate: {lr(step)}")

    print("Model saved")
    torch.save(nets, "model.pt")
    
    return stats


if __name__ == "__main__":
    # import jax
    # device = jax.devices("gpu")[1]
    # jax._src.config.update("jax_default_device", device)
    stats = train()

    noise_scheduler = DDPMScheduler(
        num_train_timesteps=50,
        beta_schedule='squaredcos_cap_v2',
        clip_sample=True,
        prediction_type='epsilon'
    )

    model = torch.load("model.pt")

    eval_policy(model=model, noise_scheduler=noise_scheduler, stats=stats)


