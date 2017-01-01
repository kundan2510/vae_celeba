# pixelVAE

Code for the MNIST models: [PixelVAE: A Latent Variable Model for Natural Images](https://arxiv.org/abs/1611.05013)

## Training
```
THEANO_FLAGS='mode=FAST_RUN,device=gpu0,floatX=float32,lib.cnmem=.95' python models/mnist_pixelvae_train.py -L 12 -fs 5 -algo cond_z_bias -dpx 16 -ldim 16
```

## Evaluation
Take the weights of the model with best validation score from the above training procedure and then run

```
THEANO_FLAGS='mode=FAST_RUN,device=gpu0,floatX=float32,lib.cnmem=.95' python models/mnist_pixelvae_evaluate.py -L 12 -fs 5 -algo cond_z_bias -dpx 16 -ldim 16 -w path/to/weights.pkl
```


