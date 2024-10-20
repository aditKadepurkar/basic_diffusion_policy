"""
This file uses a huggingface dataset for images and trains the diffusion policy using the images
"""

import jax
import jax.numpy as jnp
import transformers
from datasets import load_dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image

class ImageDataLoader:
    def __init__(self, dataset_name="microsoft/cats_vs_dogs", batch_size=32, image_size=(128, 128)):
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.image_size = image_size
        self.transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor()
        ])
        self.dataset = self.load_and_transform_dataset()
        self.dataloader = self.create_dataloader()

    def load_and_transform_dataset(self):
        dataset = load_dataset(self.dataset_name)
        dataset = dataset.map(self.transform_example)
        return dataset

    def transform_example(self, example):
        example['image'] = self.transform(Image.fromarray(example['image']))
        return example

    def create_dataloader(self):
        return DataLoader(self.dataset['train'], batch_size=self.batch_size, shuffle=True)

    def get_dataloader(self):
        return self.dataloader

# Example usage
# image_data_loader = ImageDataLoader('huggingface/image_dataset_name')
# dataloader = image_data_loader.get_dataloader()
# for batch in dataloader:
#     images = batch['image']
    # Your training code here
