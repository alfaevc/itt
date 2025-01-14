import numpy as np
import math
import network

def orthogonal_epsilons(N,dim):
    epsilons_N=np.zeros((math.ceil(N/dim)*dim,dim))
    for i in range(0,math.ceil(N/dim)):
        epsilons = np.random.standard_normal(size=(dim, dim))
        Q, _ = np.linalg.qr(epsilons)
        Q_normalize=np.copy(Q)
        fn = lambda x, y: np.linalg.norm(x) * y
        Q_normalize = np.array(list(map(fn, epsilons, Q_normalize)))
        epsilons_N[i*dim:(i+1)*dim] = Q_normalize@Q
    return epsilons_N[0:N]


def construct_hashing_table(theta,hyper_params):
    n_batch = hyper_params.n_batch
    batch_size = hyper_params.batch_size
    nA = hyper_params.nA
    n_proj_vec = hyper_params.n_proj_vec
    query_size = hyper_params.query_size
    action_net = network.itt_get_action_net(theta[:hyper_params.action_nn_dim],hyper_params)
    actions_arr = np.random.uniform(low=-1, high=1, size=(n_batch * batch_size, nA))
    latent_actions_arr = network.itt_action_forward_multiple_pass(action_net, actions_arr, hyper_params)

    projection_vector = orthogonal_epsilons(n_proj_vec, nA + 1)
    # add extra dim to make two norms equal. So max dot product is equivalent to max cosine similarity
    latent_actions_norm = np.linalg.norm(latent_actions_arr, axis=1)
    max_norm = max(latent_actions_norm)
    extra_dim = np.sqrt(max_norm - latent_actions_norm)
    aug_latent_actions_arr = np.hstack((latent_actions_arr, extra_dim.reshape((-1, 1))))
    medians = np.median(aug_latent_actions_arr @ projection_vector.T,axis=0)
    # generate keys
    binary_vecs = np.sign(aug_latent_actions_arr @ projection_vector.T-medians)  # shape is (num_actions,n_proj_vec)
    binary_vecs = (binary_vecs + 1) / 2
    powers = 2 ** np.arange(n_proj_vec)
    keys = np.dot(binary_vecs, powers)
    ind = np.argsort(keys)
    keys = keys[ind]
    # initialize hash table
    keys_val, idx_start, count = np.unique(keys, return_counts=True, return_index=True)
    arr = np.arange(2 ** n_proj_vec).astype(float)
    start_points = abs(arr[:, None] - keys_val[None, :]).argmin(axis=-1)
    start_points = idx_start[start_points]
    end_points = np.minimum(start_points + query_size, n_batch * batch_size - 1)
    start_points = np.minimum(start_points, n_batch * batch_size - query_size)
    # for ease of initialization, we always query the same amount of actions
    intervals_all = np.hstack(
        (start_points.reshape((-1, 1)), end_points.reshape((-1, 1))))
    intervals_all = intervals_all.astype(int)
    return latent_actions_arr, actions_arr, medians, intervals_all, projection_vector

def get_multiple_tables(epsilons,theta,hyper_params):
    sigma=hyper_params.sigma
    n_batch = hyper_params.n_batch
    batch_size = hyper_params.batch_size
    nA = hyper_params.nA
    n_proj_vec = hyper_params.n_proj_vec
    n_tables = epsilons.shape[0]
    actions_arr_all = np.zeros((2 * n_tables, n_batch * batch_size, nA))
    latent_actions_arr_all = np.zeros((2 * n_tables, n_batch * batch_size, nA))
    medians_all = [0] * n_tables * 2
    intervals_all = [0] * n_tables * 2
    projection_vector_all = np.zeros((2 * n_tables, n_proj_vec, nA + 1))
    for i in range(n_tables):
        latent_actions_arr_all[2*i], actions_arr_all[2*i], medians_all[2*i], intervals_all[2*i], projection_vector_all[2*i] \
            = construct_hashing_table(theta + sigma * epsilons[i],hyper_params)
        latent_actions_arr_all[2*i+1], actions_arr_all[2*i+1], medians_all[2*i+1] \
            , intervals_all[2*i+1], projection_vector_all[2*i+1] = construct_hashing_table(
            theta - sigma * epsilons[i],hyper_params)
    return actions_arr_all, latent_actions_arr_all, medians_all, intervals_all, projection_vector_all