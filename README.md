# Diffusion Policy implementation in jax
This is just some code I wrote a few months ago that has a full pipeline of data collection -> training -> inference/evaluation

It includes the original code that was written in pytorch and my jax implementation that is made more modular and more tuned for the demonstrations I collected. Robosuite(mujoco) is used for data collection as well as some data augmentation with mimicgen. You will also see a bunch of test files I wrote, you should ignore them, but I left them in because they are useful for me to mess around with apis and whatnot.