Recognition rate = accuracy on all 10,000 test images before any attack. ASR = share of correctly classified images that the attack turns into a wrong answer, measured on the first 2,000 test images (10 iterations for I-FGSM / PGD / MI-FGSM).


**Attack success rate at ε = 0.1** (pixels in [0, 1]; 0.1 ≈ 26/255 grey levels)

| Model | Recognition rate (clean) | FGSM | I-FGSM | PGD | MI-FGSM |
|---|---:|---:|---:|---:|---:|
| MLP | 98.08% | 72.7% | 81.7% | 85.5% | 78.7% |
| CNN | 99.35% | 24.4% | 49.1% | 76.2% | 42.8% |
| Transformer | 98.13% | 86.5% | 99.1% | 100.0% | 97.2% |
| CNN (adv-trained) | 98.77% | 2.3% | 2.6% | 2.7% | 2.6% |

**Attack success rate at ε = 0.3** (pixels in [0, 1]; 0.3 ≈ 76/255 grey levels)

| Model | Recognition rate (clean) | FGSM | I-FGSM | PGD | MI-FGSM |
|---|---:|---:|---:|---:|---:|
| MLP | 98.08% | 99.7% | 100.0% | 100.0% | 100.0% |
| CNN | 99.35% | 86.3% | 100.0% | 100.0% | 99.8% |
| Transformer | 98.13% | 98.4% | 100.0% | 100.0% | 100.0% |
| CNN (adv-trained) | 98.77% | 7.1% | 12.6% | 17.6% | 10.7% |
