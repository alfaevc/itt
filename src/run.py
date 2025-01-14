import numpy as np
import gym
import time
import multiprocessing as mp
import network
import utils
import f_eval
import re
import pybullet_envs
import argparse


def collect_result(result):
    result_list.append(result)

def AT_gradient_parallel(theta,hyper_params, update_action):
    numCPU = mp.cpu_count()
    pool = mp.Pool(numCPU)
    global result_list; result_list = []
    theta_dim = hyper_params.theta_dim
    # fully utilize the compute
    jobs=round(np.ceil(theta_dim/numCPU))
    if np.floor(theta_dim/numCPU) >= 0.8*theta_dim/numCPU:
        jobs = round(np.floor(theta_dim/numCPU))
    n_jobs=round(jobs*numCPU)
    
    epsilons = utils.orthogonal_epsilons(n_jobs, hyper_params.theta_dim)
    sigma = hyper_params.sigma
    if update_action==0:
        epsilons[:,:hyper_params.action_nn_dim]=np.zeros((epsilons.shape[0],hyper_params.action_nn_dim))
    if hyper_params.useHash==1:
        actions_arr_all, latent_actions_arr_all, medians_all, intervals_all, projection_vector_all \
            = utils.get_multiple_tables(epsilons,theta,hyper_params)              
        for i in range(numCPU):
            pool.apply_async(F_subprocess,args = (epsilons[i*jobs:(i+1)*jobs], theta, hyper_params,
                        actions_arr_all[2*i*jobs:(2*i+2)*jobs],
                        latent_actions_arr_all[2*i*jobs:(2*i+2)*jobs],
                        medians_all[2*i*jobs:(2*i+2)*jobs],
                        intervals_all[2*i*jobs:(2*i+2)*jobs],
                        projection_vector_all[2*i*jobs:(2*i+2)*jobs]),
                        callback=collect_result)
    else:
        for i in range(numCPU):
            pool.apply_async(F_subprocess,args = (epsilons[i*jobs:(i+1)*jobs], theta, hyper_params), callback=collect_result)
    pool.close()
    pool.join()
    grad = np.average(result_list, axis=0)
    if update_action == 0:
        # actions params are not updated
        grad[:hyper_params.action_nn_dim]=np.zeros(hyper_params.action_nn_dim)
    return grad


def F_subprocess(epsilons, theta, hyper_params, actions_arr_all=[], latent_actions_arr_all=[], medians_all=[], intervals_all=[], projection_vector_all=[]):
    if epsilons.ndim == 1:
        epsilons = epsilons.reshape(1,-1)
    grad = np.zeros(epsilons.shape)
    sigma = hyper_params.sigma
    if hyper_params.useHash==1:
        for i in range(epsilons.shape[0]):
            output1 = f_eval.F_hash(theta + sigma * epsilons[i], hyper_params, actions_arr_all[2*i],latent_actions_arr_all[2*i],medians_all[2*i],
                            intervals_all[2*i],projection_vector_all[2*i])
            output2 = f_eval.F_hash(theta - sigma * epsilons[i], hyper_params, actions_arr_all[2*i+1],latent_actions_arr_all[2*i+1],
                            medians_all[2*i+1],intervals_all[2*i+1],projection_vector_all[2*i+1])
            grad[i] = (output1 - output2) * epsilons[i]
    elif hyper_params.model_name=='itt':
        for i in range(epsilons.shape[0]):
            output1 = f_eval.F_itt(theta + sigma * epsilons[i], hyper_params)
            output2 = f_eval.F_itt(theta - sigma * epsilons[i], hyper_params)
            grad[i] = (output1 - output2) * epsilons[i]
    elif hyper_params.model_name=='iot':
        for i in range(epsilons.shape[0]):
            output1 = f_eval.F_iot(theta + sigma * epsilons[i], hyper_params)
            output2 = f_eval.F_iot(theta - sigma * epsilons[i], hyper_params)
            grad[i] = (output1 - output2) * epsilons[i]
    else: # explicit
        for i in range(epsilons.shape[0]):
            output1 = f_eval.F_explicit(theta + sigma * epsilons[i], hyper_params)
            output2 = f_eval.F_explicit(theta - sigma * epsilons[i], hyper_params)
            grad[i] = (output1 - output2) * epsilons[i]
    grad = np.average(grad,axis=0)/sigma/2
    return grad

def update_action_rules(i,hyper_params):
    if hyper_params.lazy_update==0:
        #for itt, update both towers every epoch
        return 1
    if i % 5 == 0:
        return 1
    else:
        return 0


def gradascent(theta0,hyper_params):
    theta = np.copy(theta0)
    accum_rewards = np.zeros(hyper_params.max_epoch)
    t1 = time.time()
    for i in range(hyper_params.max_epoch):
        accum_rewards[i] = f_eval.eval(theta,hyper_params)
        if i % 1 == 0:
            print("The return for epoch {0} is {1}".format(i, accum_rewards[i]))
            with open(hyper_params.outfile, "a") as f:
                f.write("%.d %.2f \n" % (i, accum_rewards[i]))
        if i % 1 == 0:
            print('runtime until now: ', time.time() - t1)
        theta += hyper_params.eta * AT_gradient_parallel(theta,hyper_params,update_action_rules(i,hyper_params))
        out_theta_file = hyper_params.model_name+'_theta_'+hyper_params.env_name+'.txt'
        np.savetxt(out_theta_file, theta, delimiter=' ', newline=' ')
    return theta, accum_rewards


##########################################################################

def parse_arguments():
    parser = argparse.ArgumentParser(description='ES Argument Parser')
    parser.add_argument('--method', dest='method', type=str, default='itt')
    parser.add_argument('--env', dest='env', type=str, default='HalfCheetah-v2')
    parser.add_argument('--import_theta', dest='import_theta', type=int, default=0)
    parser.add_argument('--n_batch', dest='n_batch', type=int, default=1)
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=1000)
    parser.add_argument('--max_epoch', dest='max_epoch', type=int, default=4000)
    parser.add_argument('--sigma', dest='sigma', type=float, default=1.0)
    parser.add_argument('--lr', dest='lr', type=float, default=1e-2)
    # variants
    parser.add_argument('--proj', dest='proj', type=int, default=6)
    parser.add_argument('--hash', dest='hash', type=int, default=0)
    parser.add_argument('--lazy', dest='lazy', type=int, default=0)
    return parser.parse_args()

class init_hyper_params:
    def __init__(self, args):
        self.model_name = args.method;self.useHash=args.hash;self.lazy_update=args.lazy
        self.eta=args.lr;self.max_epoch=args.max_epoch;self.sigma=args.sigma;
        self.env_name=args.env; self.n_batch=args.n_batch;self.batch_size=args.batch_size
        env = gym.make(args.env)
        state_dim = env.reset().size
        nA, = env.action_space.shape
        self.nA=nA;self.state_dim=state_dim
        outfile = "{}.txt".format(args.method + '_' + args.env + str(time.time()))
        self.outfile=outfile
        theta_dim, action_nn_dim, state_nn_dim = network.get_theta_dim(args.method, state_dim, nA)
        num_epsilons = theta_dim
        self.theta_dim=theta_dim;self.num_epsilons=num_epsilons
        #parameters for hash table.
        query_size = int(round(args.n_batch * args.batch_size / (2 ** args.proj)))
        self.n_proj_vec=args.proj;self.query_size=query_size
        #parameters for twin tower model
        self.action_nn_dim=action_nn_dim; self.state_nn_dim=state_nn_dim

if __name__ == '__main__':
    args = parse_arguments()
    hyper_params = init_hyper_params(args)

    import_theta = args.import_theta
    input_theta_file = args.method+'_theta_'+args.env+'.txt'
    with open(hyper_params.outfile, "w") as f:
        f.write("")

    t_start = time.time()
    theta0 = np.random.standard_normal(size=hyper_params.theta_dim)
    if import_theta:
        with open(input_theta_file, "r") as g:
            l = list(filter(len, re.split(' |\*|\n', g.readlines()[0])))
        for i in range(len(l)):
            theta0[i] = float(l[i])
    time_elapsed = int(round(time.time() - t_start))
    with open(hyper_params.outfile, "a") as f:
        f.write("Initialized \n")
    theta, accum_rewards = gradascent(theta0, hyper_params)