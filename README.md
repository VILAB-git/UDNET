# UDNET (ECCV 2024)

Official repository for the ECCV 2024 paper, **"Towards Robust Event-based Networks for Nighttime via Unpaired Day-to-Night Event Translation."**

[[Paper](https://arxiv.org/abs/2407.10703)] 

I will soon be sharing more details on this pages.

## Training
We used a single GPU(RTX 3090) to train our model.
```
python train.py --dataroot {dataset path} --name {experiment name} --gpu_ids 0 --mode sb --netG unet --lambda_SB 1.0 --lambda_NCE 1.0 --input_nc 6 --output_nc 6 --ngf 72 --temp_nc 3 --temp_nce_T 0.11
```

## Inference
```
python inference_npy.py --dataroot {dataset path} --name {experiment name} --mode sb --netG unet --input_nc 6 --output_nc 6 --ngf 72 --temp_nc 3 --epoch {specific ckpt number} --phase test --eval
```

### Acknowledgement
Our source code is based on [UNSB](https://github.com/cyclomon/UNSB).
