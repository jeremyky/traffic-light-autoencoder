https://drive.google.com/file/d/10qpfSWbN761SKgAajw6fowlZ4WO_-ehg/view?usp=sharing

Autoencoder training dataset

# Overview

A U-Net-lite convolutional autoencoder designed for 256 x 256 RGB traffic-light frames. Balances a small bottleneck with high reconstruction quality via additive skip connections and a lightweight depthwise seperable convolutions

# Architecture

Input  B×3×256×256
    Encoder (downsample by 2× per stage; DW-Separable convs)
    Stem:  3 → 32,  k3 s2      →  B×32×128×128   (save s0)
    Block1:32 → 64,  k3 s2      →  B×64×64×64    (save s1)
    Block2:64 → 96,  k3 s2      →  B×96×32×32    (save s2)
    Block3:96 → 96,  k3 s2      →  B×96×16×16    (save s3)
    Flatten → FC: 96·16·16=24,576 → latent_dim (z)

Decoder (nearest-neighbor upsampling + DW-Separable; additive skips)
    FC: latent_dim → 24,576 → reshape to B×96×16×16
    16×16:   add match3(s3) → DW-Sep 96→96
    ↑ 2× → 32×32: add match2(s2) → DW-Sep 96→64
    ↑ 2× → 64×64: add match1(s1) → DW-Sep 64→48
    ↑ 2× →128×128: add match0(s0) → DW-Sep 48→32
    ↑ 2× →256×256: Head 1×1, 32→3 → **Sigmoid**

Output  B×3×256×256  (values in [0,1])
