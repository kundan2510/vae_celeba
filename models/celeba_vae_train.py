"""
Usage: THEANO_FLAGS='mode=FAST_RUN,device=gpu0,floatX=float32,lib.cnmem=.95' python -u models/celeba_vae_train.py -edim 32 -ddim 32 -name expname
"""

import os, sys
sys.path.append(os.getcwd())

import time

import argparse

import lib
import lib.train_loop
import lib.celeba
import lib.ops.kl_unit_gaussian
import lib.ops.conv2d
import lib.ops.deconv2d
import lib.ops.linear


import numpy as np
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import scipy.misc
import lasagne
import pickle

import functools

import sys
sys.setrecursionlimit(10000)

parser = argparse.ArgumentParser(description='Generating images pixel by pixel')
parser.add_argument('-ldim', '--latent_dim', required=False, default=64, type=int)
parser.add_argument('-name', '--name', required=True, help="Name of the experiment")
parser.add_argument('-edim', '--encoder_dim', required=True, type=int, help='Dimension of activations in encoder')
parser.add_argument('-ddim', '--decoder_dim', required=True, type=int, help='Dimension of activations in decoder')
parser.add_argument('-resume', '--resume', required=False, default=0, type=int, help='Whether to resume. Very naive.\
                                             It just loads last saved parameters from the same directory.')
parser.add_argument('-beta', '--beta', required=False, default=1., type=lib.floatX, help='Coefficient of KL, beta = 1, means VAE.')

args = parser.parse_args()


# assert args.decoder_algorithm in ['cond_z_bias', 'upsample_z_conv']

print args


LATENT_DIM = args.latent_dim
ENCODER_DIM = args.encoder_dim
DECODER_DIM = args.decoder_dim
N_CHANNELS = 3
BATCH_SIZE = 32
TEST_BATCH_SIZE = 64
ALPHA_ITERS = 20000

if os.path.exists("/Tmp"):
    OUT_PREFIX = "/Tmp/kumarkun"
else:
    OUT_PREFIX = "/home/kundan"

OUT_DIR_PARAMS = '{}/celeba/vae/{}/params'.format(OUT_PREFIX, args.name)
OUT_DIR_SAMPLES = '{}/celeba/vae/{}/samples'.format(OUT_PREFIX, args.name)


if not os.path.isdir(OUT_DIR_SAMPLES):
    os.makedirs(OUT_DIR_SAMPLES)
    print "Created directory {}".format(OUT_DIR_SAMPLES)

if not os.path.isdir(OUT_DIR_PARAMS):
    os.makedirs(OUT_DIR_PARAMS)
    print "Created directory {}".format(OUT_DIR_PARAMS)


lib.ops.conv2d.enable_default_weightnorm()
lib.ops.linear.enable_default_weightnorm()

floatX = lib.floatX
T.nnet.elu = lambda x: T.switch(x >= floatX(0.), x, T.exp(x) - floatX(1.))


TIMES = ('iters', 2000, 2000*400, 2000, 400*500, ALPHA_ITERS)

lib.print_model_settings(locals().copy())

theano_srng = RandomStreams(seed=234)

def PixCNNGate(x):
    a = x[:, ::2]
    b = x[:, 1::2]
    return T.tanh(a) * T.nnet.sigmoid(b)

def PixCNN_condGate(x, z, dim,  activation='tanh', name=""):
    a = x[:, ::2]
    b = x[:, 1::2]

    Z_to_tanh = lib.ops.linear.Linear(name+".tanh", input_dim=LATENT_DIM, output_dim=dim, inputs=z)
    Z_to_sigmoid = lib.ops.linear.Linear(name+".sigmoid", input_dim=LATENT_DIM, output_dim=dim, inputs=z)

    a = a + Z_to_tanh[:, :, None, None]
    b = b + Z_to_sigmoid[:, :, None, None]

    if activation == 'tanh':
        return T.tanh(a) * T.nnet.sigmoid(b)
    else:
        return T.nnet.elu(a) * T.nnet.sigmoid(b)

def subpixel_conv(x, input_dim, output_dim, name=""):
    # TODO
    pass

def downsample_block(x, z, input_dim, dim, activation='elu', name=""):
    assert( (input_dim % 4) == 0)

    """
    TODO
    """
def resnet_block(x, input_dim, output_dim, sampling='same', masking=False, activation='elu', name=""):
    if sampling == 'up':
        x = T.nnet.abstract_conv.bilinear_upsampling(x, 2)
    elif sampling == 'down':
        x = T.signal.pool.pool_2d(x, (2, 2), ignore_border=True, mode='average_exc_pad')

    x_1x1 = lib.ops.conv2d.Conv2D(
        name + ".1_1x1",
        input_dim=input_dim,
        output_dim=3*output_dim,
        filter_size=(1, 1),
        inputs=x
    )

    x_res_conn = x_1x1[:, :output_dim]
    x = PixCNNGate(x_1x1[:, output_dim:])

    x = lib.ops.conv2d.Conv2D(
        name + ".actual_conv",
        input_dim=output_dim,
        output_dim=2*output_dim,
        filter_size=(3, 3),
        inputs=x
    )

    x = PixCNNGate(x)

    x = lib.ops.conv2d.Conv2D(
        name + ".2_1x1",
        input_dim=output_dim,
        output_dim=2*output_dim,
        filter_size=(1, 1),
        inputs=x
    )
    x = PixCNNGate(x)

    return x+x_res_conn


def Encoder(images):
    output = images
    dims = [ENCODER_DIM*(2**i) if i < 4 else ENCODER_DIM*8 for i in range(6)]
    for i in range(6):
        output = resnet_block(output, 3 if i == 0 else dims[i-1], dims[i], sampling='down', name="Encoder.resnet_{}_1".format(i))

    for i in range(3):
        output = lib.ops.conv2d.Conv2D(
            "Encoder.out_{}".format(i+1),
            input_dim=dims[5],
            output_dim=3*dims[5],
            filter_size=(1, 1),
            inputs=output
        )
        output = PixCNNGate(output[:, :2*dims[5]]) + output[:, 2*dims[5]:]

    output = output.reshape((output.shape[0], -1))
    output = lib.ops.linear.Linear(
        "Encoder.final",
        input_dim=dims[5],
        output_dim=2*LATENT_DIM,
        inputs=output
    )

    return output[:, ::2], output[:, 1::2]

def Decoder(latent):
    dims = [DECODER_DIM*(2**i) if i < 4 else DECODER_DIM*8 for i in range(6)]
    output = latent
    output = lib.ops.linear.Linear("Decoder.inp.1", input_dim=LATENT_DIM, output_dim=3*dims[5], inputs=output)

    output = PixCNNGate(output[:, :2*dims[5]]) + output[:, 2*dims[5]:]
    output = lib.ops.linear.Linear("Decoder.inp.2", input_dim=dims[5], output_dim=3*dims[5], inputs=output)
    output = PixCNNGate(output[:, :2*dims[5]]) + output[:, 2*dims[5]:]

    output = lib.ops.linear.Linear("Decoder.inp.4", input_dim=dims[5], output_dim=3*4*dims[5], inputs=output)
    output = PixCNNGate(output[:, :8*dims[5]]) + output[:, 8*dims[5]:]
    output = output.reshape((output.shape[0], dims[5], 2, 2))

    for i in range(5):
        output = resnet_block(output, dims[5-i], dims[4-i], sampling='up', name="Decoder.resnet_{}_1".format(i+1))

    for i in range(2):
        output = lib.ops.conv2d.Conv2D("Decoder.out_{}".format(i+1), input_dim=dims[0], output_dim=3*dims[0], filter_size=(1, 1), inputs=output)
        output = PixCNNGate(output[:, :2*dims[0]]) + output[:, 2*dims[0]:]

    output = lib.ops.conv2d.Conv2D("Decoder.out.final", input_dim=dims[0], output_dim=N_CHANNELS*256, filter_size=(1, 1), inputs=output)
    output = output.reshape((output.shape[0], N_CHANNELS, 256, output.shape[2], output.shape[3]))

    return output

def compute_cross_entropy_cost(logits, ground_truth):
    logits_transposed = logits.dimshuffle(0, 3, 4, 1, 2)
    logits_reshaped = logits_transposed.reshape((-1, 256))

    outputs_transposed = ground_truth.dimshuffle(0, 2, 3, 1)
    outputs_reshaped = outputs_transposed.flatten()

    cost = T.nnet.categorical_crossentropy(T.nnet.softmax(logits_reshaped), outputs_reshaped)

    cost = cost.reshape(outputs_transposed.shape)

    return cost.mean(axis=0).sum()

def get_images_from_logits(logits):
    logits_transposed = logits.dimshuffle(0, 3, 4, 1, 2)
    logits_reshaped = logits_transposed.reshape((-1, 256))
    output = T.argmax(T.nnet.softmax(logits_reshaped), axis=1).reshape(logits_transposed.shape[:4]).dimshuffle(0, 3, 1, 2)
    return output




images = T.itensor4('images')

images_rescaled = T.cast(images/16. - 8., theano.config.floatX)
mu, log_sigma = Encoder(images_rescaled)

eps = T.cast(theano_srng.normal(mu.shape), theano.config.floatX)
latents = mu + (eps * T.exp(log_sigma))

sampled_latents = T.matrix('sampled_latents')

sample_fn = theano.function(
    [sampled_latents],
    get_images_from_logits(Decoder(sampled_latents))
)

logits = Decoder(latents)

reconst_cost = compute_cross_entropy_cost(logits, images)/(64.*64.*3.)
kl_cost = lib.ops.kl_unit_gaussian.kl_unit_gaussian(
    mu,
    log_sigma
).sum(axis=1).mean()/(64.*64.*3.)

total_iters = T.iscalar('total_iters')
alpha = T.minimum(
    args.beta,
    T.cast(total_iters, theano.config.floatX) / lib.floatX(ALPHA_ITERS)
)

cost = alpha*kl_cost + reconst_cost

def generate_and_save_samples(tag):
    """TODO: Implement this"""
    latents = lib.floatX((np.random.normal(size=(100, LATENT_DIM))))
    images = sample_fn(latents)

    def save_images(images, filename):
        """images.shape: (batch, n channels, height, width)"""
        new_tag = tag

        images = images.transpose(0, 2, 3, 1)
        images = images.reshape((10, 10, 64, 64, 3))

        images = images.transpose(0, 2, 1, 3, 4)
        images = images.reshape((10*64, 10*64, 3))

        image = scipy.misc.toimage(images, channel_axis=2)
        image.save('{}/{}_{}.jpg'.format(OUT_DIR_SAMPLES, filename, new_tag))

    save_images(images, "samples")

if args.resume:
    lib.load_params(os.path.join(OUT_DIR_PARAMS, "params_last.pkl"))

print "Creating data loader.."
train_data, dev_data, test_data = lib.celeba.load(
    BATCH_SIZE,
    TEST_BATCH_SIZE
)

generate_and_save_samples("initial")

print "Running Train loop.."

try:
    lib.train_loop.train_loop(
        inputs=[total_iters, images],
        inject_total_iters=True,
        cost=cost,
        prints=[
            ('KL', kl_cost),
            ('reconst', reconst_cost),
            ('alpha/beta', alpha)
        ],
        optimizer=functools.partial(lasagne.updates.adam, learning_rate=0.001),
        train_data=train_data,
        test_data=dev_data,
        callback=generate_and_save_samples,
        times=TIMES
    )
except:
    print "Training interrupted. Saving params.. Please wait..."
    lib.save_params(os.path.join(OUT_DIR_PARAMS, "params_last.pkl"))
