import os
import sys
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import time
import tqdm

import jax
from jax.config import config
config.update("jax_debug_nans", True)
import haiku as hk
from jax import numpy as jnp
from jax.experimental.callback import rewrite

from bnn_hmc_google_research.bnn_hmc.utils import data_utils
from bnn_hmc_google_research.bnn_hmc.utils import models
from bnn_hmc_google_research.bnn_hmc.utils import losses
from bnn_hmc_google_research.bnn_hmc.utils import checkpoint_utils
from bnn_hmc_google_research.bnn_hmc.utils import cmd_args_utils
from bnn_hmc_google_research.bnn_hmc.utils import logging_utils
from bnn_hmc_google_research.bnn_hmc.utils import train_utils
from bnn_hmc_google_research.bnn_hmc.utils import tree_utils
from bnn_hmc_google_research.bnn_hmc.utils import precision_utils
from bnn_hmc_google_research.bnn_hmc.utils import optim_utils


def make_model(output_size, layer_dims, invsp_noise_std):
    def forward(batch, is_training):
        x, _ = batch
        x = hk.Flatten()(x)
        for layer_dim in layer_dims:
            x = hk.Linear(layer_dim)(x)
            x = jax.nn.relu(x)
        x = hk.Linear(output_size)(x)
        x = jnp.concatenate([x, jnp.ones_like(x) * invsp_noise_std], -1)
        return x
    return forward

def inv_softplus(x):
    return jnp.log(jnp.exp(x) - 1)

def resample_params(seed, params, std=0.005):
    key = jax.random.PRNGKey(seed)
    num_leaves = len(jax.tree_leaves(params))
    normal_keys = list(jax.random.split(key, num_leaves))
    treedef = jax.tree_structure(params)
    normal_keys = jax.tree_unflatten(treedef, normal_keys)
    params = jax.tree_map(lambda p, k: jax.random.normal(k, p.shape) * std,
                               params, normal_keys)
    return params

class BNNTraining:
    def __init__(self, layer_dim_list, output_size, num_devices=None):
        self.output_size = output_size
        if num_devices is None:
            self.num_devices = len(jax.devices())
        else:
            self.num_devices = num_devices        
        print('Default backend: ', jax.lib.xla_bridge.get_backend().platform)
    
        noise_std = 0.05
        invsp_noise_std = inv_softplus(noise_std)
        self.net_fn = make_model(output_size=self.output_size, layer_dims=layer_dim_list, invsp_noise_std=invsp_noise_std)

        self.net = hk.transform_with_state(self.net_fn)
        self.net_apply, net_init = self.net.apply, self.net.init
        self.net_apply = precision_utils.rewrite_high_precision(self.net_apply)

        self.param_seed = 2
        data_info = {"y_scale": 1.}
        task = data_utils.Task("regression")
        (likelihood_factory, self.predict_fn, ensemble_upd_fn, self.metrics_fns, tabulate_metrics) = train_utils.get_task_specific_fns(task, data_info)


    def train_data_BNN(self, X_train_arr, Y_train_arr, max_epochs, folder):

        x = jnp.asarray(X_train_arr)
        y = jnp.asarray(Y_train_arr)

        train_set = (x, y)
        train_set = data_utils.pmap_dataset(train_set, self.num_devices)

        params, net_state = self.net.init(jax.random.PRNGKey(self.param_seed), (x, None), True)
        params = resample_params(self.param_seed, params)
        net_state = jax.pmap(lambda _: net_state)(jnp.arange(self.num_devices))

        prior_std = 0.1
        weight_decay = 1 / prior_std**2
        temperature = 1

        log_prior_fn, log_prior_diff_fn = losses.make_gaussian_log_prior(weight_decay, 1.)
        log_likelihood_fn = losses.make_gaussian_likelihood(1.)

        step_size = 1e-5
        trajectory_len = jnp.pi / 2 / jnp.sqrt(weight_decay)
        max_num_leapfrog_steps = int(trajectory_len // step_size + 1)
        print("Leapfrog steps per iteration:", max_num_leapfrog_steps)

        update, get_log_prob_and_grad = train_utils.make_hmc_update(
            self.net_apply, log_likelihood_fn, log_prior_fn, log_prior_diff_fn,
            max_num_leapfrog_steps, 1.0, 0.)

        # Initial log-prob and grad values
        log_prob, state_grad, log_likelihood, net_state = (
            get_log_prob_and_grad(train_set, params, net_state))

        all_test_preds = []
        all_train_preds = []
        key = jax.random.PRNGKey(0)
        log_likelihood_list = []
        mse_list = []
        params_list = []

        for iteration in tqdm.tqdm(range(max_epochs)):
            
            # in_burnin = (iteration < const.num_burn_in_iterations)
        #     do_mh_correction = (not args.no_mh) and (not in_burnin)

            (params, net_state, log_likelihood, state_grad, step_size, key,
            accept_prob, accepted) = (
                update(train_set, params, net_state, log_likelihood, state_grad,
                    key, step_size, trajectory_len, True))
            # # Evaluation
            # test_predictions = np.asarray(
            # self.predict_fn(self.net_apply, params, net_state, test_set))
            # if accepted:
            #     all_test_preds.append(test_predictions)
            train_predictions = np.asarray(
            self.predict_fn(self.net_apply, params, net_state, train_set))
            if accepted:
                all_train_preds.append(train_predictions)
                
            mse = self.metrics_fns['mse'](train_predictions[1][0], Y_train_arr)

            print("It: {} \t Accept P: {} \t Accepted {} \t Log-likelihood: {} \t MSE: {}".format(
                    iteration, accept_prob, accepted, log_likelihood))

            mse_list.append(mse)
            log_likelihood_list.append(-log_likelihood)

        return all_train_preds, net_state, params_list, log_likelihood_list, mse_list

    def predict_a_batch(self, net_state, params, x_in, actual_y):
        'todo check if this fcn works'
        f_test = jnp.asarray(x_in)
        y_test = jnp.asarray(actual_y)
        test_set = (f_test, y_test)
        test_set = data_utils.pmap_dataset(test_set, self.num_devices)
        predictions = np.asarray(self.predict_fn(self.net_apply, params, net_state, test_set))
        return predictions


def evaluate_BNN():

    input_data = pd.read_csv('tests/inputs_nuc.csv')
    label_data = pd.read_csv('tests/labels_nuc.csv')

    folder_results = 'results/'
    if not os.path.exists(folder_results):
        os.makedirs(folder_results)

    system_NN = BNNTraining(layer_dim_list=[100, 100], output_size=label_data.shape[1])
    size_training_data = int(input_data.shape[0] * 0.8)

    x_input_arr = input_data[:size_training_data].to_numpy()
    y_input_arr = label_data[:size_training_data].to_numpy()

    x_test_arr = input_data[size_training_data:].to_numpy()
    y_test_arr = label_data[size_training_data:].to_numpy()

    all_train_preds, net_state, params_list, log_likelihood_list, mse_list = system_NN.train_data_BNN(
                X_train_arr=x_input_arr,
                Y_train_arr=y_input_arr,
                max_epochs=40,
                folder=folder_results,
            )
    
    df_mse_list = pd.DataFrame(mse_list)
    df_mse_list.to_csv(folder_results + 'mse_list_bnn.csv')


if __name__ == '__main__':
    evaluate_BNN()
