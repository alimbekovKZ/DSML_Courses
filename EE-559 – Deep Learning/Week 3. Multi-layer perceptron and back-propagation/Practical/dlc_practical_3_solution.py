#!/usr/bin/env python

import math
import torch
from torch import Tensor

import dlc_practical_prologue as prologue

######################################################################

def sigma(x):
    return x.tanh()

def dsigma(x):
    return 4 * (x.exp() + x.mul(-1).exp()).pow(-2)

######################################################################

def loss(v, t):
    return (v - t).pow(2).sum()

def dloss(v, t):
    return 2 * (v - t)

######################################################################

def forward_pass(w1, b1, w2, b2, x):
    x0 = x
    s1 = w1.mv(x0) + b1
    x1 = sigma(s1)
    s2 = w2.mv(x1) + b2
    x2 = sigma(s2)

    return x0, s1, x1, s2, x2

def backward_pass(w1, b1, w2, b2,
                  t,
                  x, s1, x1, s2, x2,
                  dl_dw1, dl_db1, dl_dw2, dl_db2):
    x0 = x
    dl_dx2 = dloss(x2, t)
    dl_ds2 = dsigma(s2) * dl_dx2
    dl_dx1 = w2.t().mv(dl_ds2)
    dl_ds1 = dsigma(s1) * dl_dx1

    dl_dw2.add_(dl_ds2.view(-1, 1).mm(x1.view(1, -1)))
    dl_db2.add_(dl_ds2)
    dl_dw1.add_(dl_ds1.view(-1, 1).mm(x0.view(1, -1)))
    dl_db1.add_(dl_ds1)

######################################################################

train_input, train_target, test_input, test_target = prologue.load_data(one_hot_labels = True,
                                                                        normalize = True)

nb_classes = train_target.size(1)
nb_train_samples = train_input.size(0)

zeta = 0.90

train_input = train_input * zeta
test_input = test_input * zeta

nb_hidden = 50
eta = 1e-1 / nb_train_samples
epsilon = 1e-6

w1 = Tensor(nb_hidden, train_input.size(1)).normal_(0, epsilon)
b1 = Tensor(nb_hidden).normal_(0, epsilon)
w2 = Tensor(nb_classes, nb_hidden).normal_(0, epsilon)
b2 = Tensor(nb_classes).normal_(0, epsilon)

dl_dw1 = Tensor(w1.size())
dl_db1 = Tensor(b1.size())
dl_dw2 = Tensor(w2.size())
dl_db2 = Tensor(b2.size())

for k in range(0, 1000):

    # Back-prop

    acc_loss = 0
    nb_train_errors = 0

    dl_dw1.zero_()
    dl_db1.zero_()
    dl_dw2.zero_()
    dl_db2.zero_()

    for n in range(0, nb_train_samples):
        x0, s1, x1, s2, x2 = forward_pass(w1, b1, w2, b2, train_input[n])

        pred = x2.max(0)[1][0]
        if train_target[n, pred] < 0: nb_train_errors = nb_train_errors + 1
        acc_loss = acc_loss + loss(x2, train_target[n])

        backward_pass(w1, b1, w2, b2,
                      train_target[n],
                      x0, s1, x1, s2, x2,
                      dl_dw1, dl_db1, dl_dw2, dl_db2)

    # Gradient step

    w1 = w1 - eta * dl_dw1
    b1 = b1 - eta * dl_db1
    w2 = w2 - eta * dl_dw2
    b2 = b2 - eta * dl_db2

    # Test error

    nb_test_errors = 0

    for n in range(0, test_input.size(0)):
        _, _, _, _, x2 = forward_pass(w1, b1, w2, b2, test_input[n])

        pred = x2.max(0)[1][0]
        if test_target[n, pred] < 0: nb_test_errors = nb_test_errors + 1

    print('{:d} acc_train_loss {:.02f} acc_train_error {:.02f}% test_error {:.02f}%'
          .format(k,
                  acc_loss,
                  (100 * nb_train_errors) / train_input.size(0),
                  (100 * nb_test_errors) / test_input.size(0)))
