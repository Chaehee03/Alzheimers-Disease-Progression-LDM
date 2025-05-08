# import torch
# import yaml
# import argparse
# import os
# import numpy as np
# from tqdm import tqdm
# from torch.optim import Adam
# from dataset.mnist_dataset import MnistDataset
# # from dataset.adni_dataset import AdniDataset
# from torch.utils.data import DataLoader
# from models.unet import UNet
# from models.vae import VAE
# from scheduler.linear_noise_scheduler import LinearNoiseScheduler
#
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(device)
# print("torch.cuda.is_available() = ", torch.cuda.is_available())
# print(torch.cuda.get_device_name(0))
#
#
# def train(args):
#     # Read the config file #
#     with open(args.config_path, 'r') as file: # open config file
#         try:
#             config = yaml.safe_load(file) # YAML file -> python dictionary
#         except yaml.YAMLError as exc: # fail to open config file -> exception
#             print(exc)
#     print(config)
#     ########################
#
#     diffusion_config = config['diffusion_params']
#     dataset_config = config['dataset_params']
#     ldm_config = config['ldm_params']
#     vae_config = config['vae_params']
#     train_config = config['train_params']
#
#     # Create the noise scheduler
#     scheduler = LinearNoiseScheduler(num_timesteps=diffusion_config['num_timesteps'],
#                                      beta_start=diffusion_config['beta_start'],
#                                      beta_end=diffusion_config['beta_end'])
#
#     # Create the dataset
#     im_dataset_cls = {
#         'mnist': MnistDataset,
#         # 'adni': AdniDataset
#     }.get(dataset_config['name'])
#
#     im_dataset = im_dataset_cls(split = 'train',
#                                 im_path = dataset_config['im_path'],
#                                 im_size = dataset_config['im_size'],
#                                 im_channels = dataset_config['im_channels'],
#                                 use_latents=True,
#                                 latent_path=os.path.join(train_config['task_name'],
#                                                          train_config['vae_latent_dir_name'])
#                                 )
#
#     # load data by batch unit
#     # num_workers: the number of CPU threads
#     data_loader = DataLoader(im_dataset,
#                              batch_size=train_config['ldm_batch_size'],
#                              shuffle=True)
#
#     # Instantiate the model
#     unet = UNet(im_channels=vae_config['z_channels'],
#                  model_config=ldm_config).to(device) # load model to GPU/CPU
#     unet.train()
#
#     # Load VAE only if latents are not present. (use_latents = False)
#     if not im_dataset.use_latents:
#         print('Loading vae model as latents are not present.')
#         vae = VAE(im_channels=dataset_config['im_channels'],
#                   model_config=vae_config).to(device)
#         vae.eval() # don't need training, just create latents
#
#         # Load pretrained vae if checkpoint exists
#         vae_checkpoint = os.path.join(train_config['task_name'],
#                                        train_config['vae_autoencoder_ckpt_name'])
#         if os.path.exists(vae_checkpoint):
#             print('Loaded vae checkpoint')
#             # missing, unexpected = vae.load_state_dict(torch.load(vae_checkpoint, map_location = device)['model_state_dict'])
#             # print("Missing keys:", missing)
#             # print("Unexpected keys:", unexpected)
#             vae.load_state_dict(torch.load(vae_checkpoint, map_location = device)['model_state_dict'])
#
#     # Specify training parameters
#     num_epochs = train_config['ldm_epochs']
#     optimizer = Adam(unet.parameters(), lr=train_config['ldm_lr'])
#     criterion = torch.nn.MSELoss()
#
#     # Freeze VAE parameters
#     # VAE is pre-trained & only used for generating latents.
#     # It is relevant only when latents don't exist. (use_latents = False)
#     if not im_dataset.use_latents:
#         for param in vae.parameters():
#             param.requires_grad = False
#
#     # Run training
#     for epoch_idx in range(num_epochs): # repeat for epoch times
#         losses = []
#         for im in tqdm(data_loader): # train for batch unit
#             optimizer.zero_grad() # initialize gradient
#             im = im.float().to(device) # load data to GPU/CPU
#
#             # It is relevant only when latents don't exist.
#             if not im_dataset.use_latents:
#                 with torch.no_grad():
#                     z, _, _ = vae.encode(im) # z: latents
#
#             # Sample random noise
#             noise = torch.randn_like(z).to(device)
#
#             # Sample timestep
#             t= torch.randint(0, diffusion_config['num_timesteps'], (z.shape[0],)).to(device)
#
#             # Add noise to images according to timestep
#             noisy_im = scheduler.add_noise(z, noise, t)
#             noise_pred = unet(noisy_im, t) # U-Net predicts the noise
#
#             loss = criterion(noise_pred, noise) # calculate loss
#             losses.append(loss.item())
#             loss.backward() # backpropagation
#             optimizer.step() # update weights
#         print("Finished epoch: {} | Loss: {:.4f}".format(
#                 epoch_idx + 1,
#                 np.mean(losses)
#         ))
#         torch.save(unet.state_dict(), os.path.join(train_config['task_name'],
#                                                     train_config['ldm_ckpt_name'])) # save model
#
#     print('Done Training ...')
#
#
# if __name__ == '__main__': # runs only when this file script is run (don't run when imported)
#     parser = argparse.ArgumentParser(description='Arguments for ddpm training')
#     parser.add_argument('--config', dest='config_path',
#                         default='config/mnist.yaml', type=str)
#     args = parser.parse_args()
#     train(args)