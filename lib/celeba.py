import os
import fuel
from fuel.datasets import CelebA

from fuel.schemes import ShuffledScheme, SequentialScheme
from fuel.streams import DataStream
import numpy as np

def _make_stream(stream, bs):
    def new_stream():
        result = np.empty((bs, 3, 64, 64), dtype = 'int32')
        for (features, targets) in stream.get_epoch_iterator():
            for i, img in enumerate(features):
                result[i] = img
            yield (result,)
    return new_stream

def load(batch_size, test_batch_size, format_ = '64'):
    tr_data = CelebA(format_, which_sets=('train',))
    val_data = CelebA(format_, which_sets=('valid',))
    test_data = CelebA(format_, which_sets=('test',))

    ntrain = tr_data.num_examples
    nval = val_data.num_examples
    ntest = test_data.num_examples

    tr_scheme = ShuffledScheme(examples=ntrain, batch_size=batch_size)
    tr_stream = DataStream(tr_data, iteration_scheme=tr_scheme)

    te_scheme = SequentialScheme(examples=ntest, batch_size=test_batch_size)
    te_stream = DataStream(test_data, iteration_scheme=te_scheme)

    val_scheme = SequentialScheme(examples=nval, batch_size=batch_size)
    val_stream = DataStream(val_data, iteration_scheme=val_scheme)

    return _make_stream(tr_stream, batch_size), \
           _make_stream(val_stream, batch_size), \
           _make_stream(te_stream, test_batch_size)

if __name__ == "__main__":
    train_data, dev_data, test_data = load(64, 64)
    import time
    start = time.time()
    i = 0
    for (data,) in train_data():
        assert(len(data) > 0)
        i += 1
        if i >= 999:
            break
    end = time.time()
    print "Loading time per batch is: {}".format((end - start)/1000.)
