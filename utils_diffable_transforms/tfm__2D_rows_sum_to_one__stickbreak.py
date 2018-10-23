'''
Differentiable transform for 2D array with positive rows that sum to one

Use Stick-Breaking transform.

Function API
------------
* to_common_arr
    Transforms array of real values to pos. values & rows sum to one
    Input shape: K x V
    Output shape: K x V-1
    Differentiable via autograd.
* to_diffable_arr
    Transforms array of pos. values & rows sum to one to real values
    Input shape: K x V-1
    Output shape: K x V
    Differentiable via autograd.

Examples
--------

## Part 1) Simple test
#

>>> np.set_printoptions(precision=4, suppress=1)

>>> topics_KV = 10 * np.eye(3) + np.ones((3,3))
>>> topics_KV /= topics_KV.sum(axis=1)[:,np.newaxis]
>>> topics_KV
array([[0.8462, 0.0769, 0.0769],
       [0.0769, 0.8462, 0.0769],
       [0.0769, 0.0769, 0.8462]])

>>> to_diffable_arr(topics_KV)
array([[ 2.3979, -0.    ],
       [-0.    ,  2.3979],
       [-2.3979, -2.3979]])

>>> to_common_arr(to_diffable_arr(topics_KV))
array([[0.8462, 0.0769, 0.0769],
       [0.0769, 0.8462, 0.0769],
       [0.0769, 0.0769, 0.8462]])


Reference
---------
Stick-breaking transform explained in Sec. 35.6 "Unit Simplex"
of the Stan Reference v 2.17.0 document:
https://github.com/stan-dev/stan/releases/download/v2.17.0/stan-reference-2.17.0.pdf

'''
import autograd
import autograd.numpy as np
from tfm__unit_interval import (
    logistic_sigmoid,
    inv_logistic_sigmoid)
from tfm__2D_rows_sum_to_one__log import (
    MIN_EPS,
    to_safe_common_arr)

def to_common_arr(
        reals_KVm1,
        min_eps=MIN_EPS,
        **kwargs):
    ''' Convert unconstrained topic weights to proper normalized topics

    Should handle any non-nan, non-inf input without numerical problems.

    Args
    ----
    reals_KVm1 : 2D array, size K x V-1
        Contains real values.

    Returns
    -------
    proba_KV : 2D array, size K x V
        Minimum value of any entry will be min_eps
        Maximum value will be 1.0 - min_eps
        Each row will sum to 1.0 (+/- min_eps)

    Examples
    --------
    >>> reals_KVm1 = np.zeros((1, 3))
    >>> to_common_arr(reals_KVm1)
    array([[0.25, 0.25, 0.25, 0.25]])

    >>> ones_KVm1 = np.ones((1, 3))
    >>> to_common_arr(ones_KVm1)
    array([[0.47536689, 0.30225   , 0.16257509, 0.05980803]])

    '''
    assert reals_KVm1.ndim == 2
    K, Vm1 = reals_KVm1.shape
    V = Vm1 + 1
    offset_Vm1 = -1.0 * np.log(V - np.arange(1.0, V))

    fracs_KVm1 = logistic_sigmoid(reals_KVm1 + offset_Vm1[np.newaxis,:])

    # v1: Fast, but cumprod is not autograd-able yet
    #cumprod_KVm1 = np.cumprod(1.0 - fracs_KVm1, axis=1)

    # v2: This is autograd-able
    tmp_KVm1 = 1.0 - fracs_KVm1
    cumprod_KVm1 = np.hstack([
        np.prod(tmp_KVm1[:, :(vv+1)], axis=1)[:,np.newaxis]
            for vv in range(V-1)
        ])

    proba_KV = np.hstack([
        fracs_KVm1[:, :1],
        fracs_KVm1[:, 1:] * cumprod_KVm1[:, :-1],
        cumprod_KVm1[:, -1:],
        ])
    assert np.allclose(1.0, np.sum(proba_KV, axis=1))
    return proba_KV


def to_diffable_arr(proba_KV, min_eps=MIN_EPS, do_force_safe=False):
    ''' Transform normalized topics to unconstrained space.

    Args
    ----
    proba_KV : 2D array, size K x V
        minimum value of any entry must be min_eps
        each row should sum to 1.0

    Returns
    -------
    reals_KVm1 : 2D array, size K x (V-1)
        unconstrained real values

    Examples
    --------
    >>> np.set_printoptions(precision=3)
    >>> V = 4
    >>> unif_1V = np.ones((1,V)) / float(V)
    >>> to_diffable_arr(unif_1V)
    array([[ 2.22e-16, -1.11e-16,  0.00e+00]])

    >>> rand_1V = np.asarray([[ 0.11, 0.22, 0.33, 0.20, 0.14 ]])
    >>> to_diffable_arr(rand_1V)
    array([[-0.704, -0.015,  0.663,  0.357]])

    '''
    assert proba_KV.ndim == 2
    K, V = proba_KV.shape
    offset_Vm1 = -1.0 * np.log(V - np.arange(1.0, V))

    cumsum_KV1m = np.maximum(
        1e-100,
        1.0 - np.cumsum(proba_KV[:, :-1], axis=1)
        )
    fracs_KV = np.hstack([
        proba_KV[:, :1],
        proba_KV[:, 1:] / cumsum_KV1m
        ])
    reals_KVm1 = (
        inv_logistic_sigmoid(fracs_KV[:, :-1])
        - offset_Vm1)
    return reals_KVm1


if __name__ == '__main__':
    topics_KV = np.eye(3) + np.ones((3,3))
    topics_KV /= topics_KV.sum(axis=1)[:,np.newaxis]

    print("Attempting reconstruction of some toy topics_KV array")
    print('------ original topics_KV')
    print(topics_KV)
    print('------ reconstructed topics_KV')
    print(to_common_arr(to_diffable_arr(topics_KV)))


    print("Consider that each row of topics_KV is a Dirichlet random var.")
    print("We set concentration so that a MAP exists: alpha = 2.0")
    print("Verify that the MAP solution has gradient zero when we use autograd...")
    K = 2
    V = 4
    alpha_V = 2.0 * np.ones(V)
    def calc_loss_dirichlet_map(reals_KV):
        t_KV = to_common_arr(reals_KV)
        return np.sum(np.dot(np.log(t_KV), alpha_V - 1.0))

    calc_grad_of_loss_dirichlet_map = autograd.grad(
        calc_loss_dirichlet_map)

    true_topics_KV = np.tile(alpha_V - 1, (K,1))
    true_topics_KV /= np.sum(true_topics_KV, axis=1)[:,np.newaxis]

    print("Printing gradient at optimal topics_KV value")
    print("Should be very close to 0.0 because it's optimal")
    reals_KV = to_diffable_arr(true_topics_KV)
    print(calc_grad_of_loss_dirichlet_map(reals_KV))

