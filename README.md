# Fixup
A Re-implementation of Fixed-update Initialization (https://arxiv.org/abs/1901.09321) *(needs Pytorch 1.0)*.

**Cite as:**

*Hongyi Zhang, Yann N. Dauphin, Tengyu Ma. Fixup Initialization: Residual Learning Without Normalization. 7th International Conference on Learning Representations (ICLR 2019).*

----
## ResNet for CIFAR-10
The default arguments will train a ResNet-110 (https://arxiv.org/abs/1512.03385) with Fixup + Mixup (https://arxiv.org/abs/1710.09412).

*Example:*

The following script will train a ResNet-32 model (https://arxiv.org/abs/1512.03385) on GPU 0 with Fixup and no Mixup (alpha=0), with weight decay 5e-4 and (the default) learning rate 0.1 and batch size 128.
```
CUDA_VISIBLE_DEVICES=0 python cifar_train.py -a fixup_resnet32 --sess benchmark_a0d1e4lr01 --seed 11111 --alpha 0. --decay 5e-4
```

----
## ResNet for ImageNet
A re-implementation is available. However, I have not been able to test it. If you try it out, please feel free to contact me --- your feedback is very welcome!

----
## Transformer for machine translation
Stay tuned!
