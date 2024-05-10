### Colorization

To colorize dogs, run the following command:

```
./colorize.py --name colorize.dog.full --gray_im_path ./imgs/dog.full.png --prompt "a color photo of dogs on green grass" --num_samples 4 --guidance_scale 10.0 --num_inference_steps 30 --start_diffusion_step 7
```

To colorize the castle, run the following command:

```
./colorize.py --name colorize.castle --gray_im_path ./imgs/castle.aspectratio.png --prompt "a colorful photo of a scottish castle" --num_samples 4 --guidance_scale 10.0 --num_inference_steps 30 --start_diffusion_step 2
```

I've found that it is helpful to play with the `start_diffusion_step` flag, which controls which step of the diffusion process to start at. The `start_diffusion_step` is only applied to the first stage of deepfloyd. The second stage is conditioned on the colorized image from the first stage, so I don't think it's needed as much, but might help. The `guidance_scale` may also be useful to tune.

The code should work with any image size, as long as image dimensions are multiples of 32 (which is what the unet seems to take). There is no multidiffusion, just passing arbitrary sized images to the unet, which seems to work fine for colorization. 