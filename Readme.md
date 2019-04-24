# Sharing Is Caring

A library for private ensemble learning. Implemented for Columbia University privacy seminar.

## Install

Using pip:
```bash
$ git clone https://github.com/bgoodc/sharing-is-caring.git
$ cd sharing-is-caring
$ pip install .
```

## Introduction
Sharing Is Caring(SIC) is a framework for collaborative machine learning. The SIC
protocol allows a set of peers, or *local experts* to collaborate on classification
problems by privately training a *meta-model.* The protocol works as follows:

1. Each peer, *e* trains a local model *M_e* (*M_e* is assumed to have been trained 
with a privacy mechanism chosen by *e*).
2. Each peer broadcasts *M_e* to all other participating peers.
3. Peers collaboratively train a *meta-model*, *MM*, that maps inputs to local models as follows:
..a. A centralized *MM*, along with its loss function, is initialized and broadcast to all peers.
..b. For rounds *t* := 1,2,...,*T*:
....1) Each peer calculates the gradient *g_et* of *MM_t-1* using local data only available to *e*. Here, noise is added to ensure differential privacy of the meta-model training data.
....2) Each peer broadcasts *g_pt* to all other peers.
....3) Each per updates *MM_t-1* to *MM_t* by averaging the gradients of all participating peers.

We assume that all peers participate honestly in the protocol except that some peers may be interested in learning about the data that other peers have trained on.

## Prototype
Our prototype is a simple meta-model located in `sic/meta_model.py`. It takes as arguments to the constructor a list of peers' models, and, the peers' local training data. Note that 
in a real implementation, each peer would have its own version of the meta-model and would broadcast gradient updates to other peers, ensuring that no one meta-model has access to all
peers' local training data.

If we have a set of pre-trained local models saved in the variable `models`, we can create new meta-model as follows:
```python
...
mm = MetaModel(input_shape, models, flags)
```

Meta-model training occurs by calling the `compute_gradient` or `update_gradient` functions:
```python
# x_local_train and y_local_train are training data and labels local to each
# peer. In practice, these would be fully isolated.
epocs = 10
for i in range(epochs):
    for j in range(npeers):
        mm.compute_gradient(x_local_train[j], y_local_train[j], update=True)
```

Alternatively:
```python
for i in range(epochs):
    mm.update_gradient([
        mm.compute_gradient(x_local_train[j], y_local_train[j], update=False)
        for j in range(npeers)])
```

The `flags` parameter is used to control the addition of DP noise
```python
flags = dict()
flags['dpsgd'] = True               # add noise on/off
flags['learning_rate'] = 0.15       # size of each gradient step
flags['noise_multiplier'] = 1.01    # amount of noise added (propto l2_norm_clip)
flags['l2_norm_clip'] = 1.0         # the maximimum size of a gradient update
flags['batch_size'] = 250           # unused
flags['epochs'] = 3                 # unused
flags['microbatches'] = 1           # unused
flags['model_dir'] = None
```

Although the meta-model is centralized, in practice this would not be the case. Each peer is supposed to see only the gradient update at step *t*, and not the full local training data. 
It would look like this:

```python
for i in range(epochs):
    wait_for_peers(i) # does not wait when i==0

    updates = [mm.compute_gradient(x_local_train, y_local_train)]
    updates += get_gradient_updates_from_peers(...)
    mm.update_gradient(updates)

    notify_peers_round(i+1)
```

As long as the gradient descent algorithm being used is deterministic, all peers will collaboratively make steps towards an optimal solution.

## Red Team Instructions
For our red team, we will provide a fully-trained meta-model, along with the global training data used by all experts, and the history of all gradient steps calculated throughout the optimization process. Your goal should be to determine if particular training data was used by any of the participating peers to train the meta-model.

## Authors
* Geelon So 
* Shilin Ni
* Brian Goodchild
