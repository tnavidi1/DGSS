import cvxpy as cp
import numpy as np

import argparse

import matplotlib.pyplot as plt


def parse_inputs(FLAGS):
    """
    Parse the file input arguments

    :param `FLAGS`: data object containing all specified arguments

    :return: `storagePen, solarPen, seed, evPen, dir_network, load_fname_orig, n_days`
    """

    storagePen = float(FLAGS.storagePen) / 10
    solarPen = float(FLAGS.solarPen) / 10
    evPen = float(FLAGS.evPen) / 10
    n_days = int(FLAGS.days)

    if FLAGS.network == 'iowa':
        dir_network = 'Iowa_feeder/'
        t_res = 24  # number of points in a day
    elif FLAGS.network == '123':
        dir_network = 'IEEE123-1ph-test/'
        t_res = 24 * 4  # number of points in a day for 15 min resolution
    else:
        print('reverting to default Iowa network')
        dir_network = 'Iowa_feeder/'
        t_res = 24  # number of points in a day

    return storagePen, solarPen, evPen, dir_network, n_days, t_res


def train_test_split(data, t_idx):
    data_train = data[:, 0:t_idx]
    data_test = data[:, t_idx:]

    return data_train, data_test


def stack_daily(data, time_resolution):
    # Input a single node time series
    # Get daily mean and cov from time series

    steps_per_day = time_resolution
    num_days = int(np.size(data) / time_resolution)
    one_day = np.zeros((np.size(data, 0), steps_per_day))

    X = np.zeros((steps_per_day, num_days)) # has shape of hours, days

    for b in range(np.size(one_day, 1)):
        sub_array = data[b::steps_per_day]
        # print(sub_array.shape)
        X[b, :] = np.reshape(sub_array, (1, num_days))

    return X


def get_demand_bounds(daily_demand, buffer = 1.1):

    upper = (np.max(daily_demand, axis=1)*buffer).clip(min=1)  # allow at least 1kW for upper demand
    lower = (np.min(daily_demand, axis=1)*buffer).clip(max=0)  # allow at least 0 consumption

    upper[upper - lower < 0.5] += 0.5  # make sure there is at least 0.5 kW difference between bounds

    return upper, lower


def global_optimization(upper_demand, lower_demand, nodes_storage, coefs_mat_v, intercept_mat_v,
                        coefs_mat_tr, intercept_mat_tr, coefs_mat_ti, intercept_mat_ti, coefs_upper_i, coefs_lower_i,
                        v_max, v_min, T_max):
    # coefs_upper_i - reactive power models, first term is intercept second is coefficient

    v_max = v_max * 100
    v_min = v_min * 100

    N = upper_demand.shape[0]
    storage_bool = np.zeros(N)
    storage_bool[nodes_storage] = 1
    storage_bool = storage_bool == 1
    not_storage_bool = np.logical_not(storage_bool)

    intercept_mat_v = intercept_mat_v.flatten()
    intercept_mat_ti = intercept_mat_ti.flatten()
    intercept_mat_tr = intercept_mat_tr.flatten()
    coefs_mat_v_p = coefs_mat_v.clip(min=0)
    coefs_mat_v_n = coefs_mat_v.clip(max=0)
    coefs_mat_tr_p = coefs_mat_tr.clip(min=0)
    coefs_mat_tr_n = coefs_mat_tr.clip(max=0)
    coefs_mat_ti_p = coefs_mat_ti.clip(min=0)
    coefs_mat_ti_n = coefs_mat_ti.clip(max=0)

    delta_u = cp.Variable(shape=nodes_storage.shape)
    delta_l = cp.Variable(shape=nodes_storage.shape)

    r_u = cp.Variable(shape=upper_demand.shape)
    r_l = cp.Variable(shape=lower_demand.shape)
    imag_u = cp.Variable(shape=upper_demand.shape)
    imag_l = cp.Variable(shape=lower_demand.shape)

    s_u = cp.hstack([r_u, imag_u]).T
    s_l = cp.hstack([r_l, imag_l]).T

    objective = cp.Minimize(-cp.sum(cp.log(upper_demand[nodes_storage] - delta_u - lower_demand[nodes_storage] - delta_l)))

    # Constraints to the minimization problem:
    constraints = [delta_u >= 0,
                   delta_l >= 0,
                   r_u[storage_bool] == upper_demand[storage_bool] - delta_u,
                   r_l[storage_bool] == lower_demand[storage_bool] + delta_l,
                   imag_u >= cp.multiply(coefs_upper_i[:, 1], r_u) + coefs_upper_i[:, 0],
                   imag_l <= cp.multiply(coefs_lower_i[:, 1], r_l) + coefs_lower_i[:, 0],
                   v_min <= coefs_mat_v_n @ s_u + coefs_mat_v_p @ s_l + intercept_mat_v,
                   v_max >= coefs_mat_v_p @ s_u + coefs_mat_v_n @ s_l + intercept_mat_v
                   #v_min <= coefs_mat_v @ s_l + intercept_mat_v,
                   #v_max >= coefs_mat_v @ s_l + intercept_mat_v,
                   #T_max**2 >= (coefs_mat_tr @ s_l + intercept_mat_tr) ** 2 + (coefs_mat_ti @ s_l + intercept_mat_ti) ** 2
                   ]
    constraints.append(T_max**2 >= (coefs_mat_tr_p @ s_u + coefs_mat_tr_n @ s_l + intercept_mat_tr)**2
                   + (coefs_mat_ti_p @ s_u + coefs_mat_ti_n @ s_l  + intercept_mat_ti)**2)

    if np.sum(not_storage_bool) > 0:
        constraints.append(r_u[not_storage_bool] == upper_demand[not_storage_bool])
        constraints.append(r_l[not_storage_bool] == lower_demand[not_storage_bool])

    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.MOSEK)  #, verbose=True)

    if problem.status != 'optimal':
        print('bounds optimization status is', problem.status)

    upper_supply = upper_demand.copy()
    upper_supply[nodes_storage] += -delta_u.value
    lower_supply = lower_demand.copy()
    lower_supply[nodes_storage] += delta_l.value

    return upper_supply, lower_supply, delta_u.value, delta_l.value


def plot_worst_bounds(upper_demand, lower_demand, upper_supply, lower_supply):
    node = np.unravel_index(upper_supply.argmin(), upper_supply.shape)[0]
    plt.figure()
    plt.plot(upper_demand[node, :].T)
    plt.plot(upper_supply[node, :].T)
    plt.plot(lower_demand[node, :].T)
    plt.plot(lower_supply[node, :].T)
    plt.legend(('upper demand', 'upper supply', 'lower demand', 'lower supply'))

    node = np.unravel_index(lower_supply.argmax(), lower_supply.shape)[0]
    plt.figure()
    plt.plot(upper_demand[node, :].T)
    plt.plot(upper_supply[node, :].T)
    plt.plot(lower_demand[node, :].T)
    plt.plot(lower_supply[node, :].T)
    plt.legend(('upper demand', 'upper supply', 'lower demand', 'lower supply'))
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train data driven models')
    parser.add_argument('--opt', default=1, help='Type of optimization 0 is perfect foresight, 1 is bounds')
    parser.add_argument('--storagePen', default=1, help='storage penetration percentage times 10')
    parser.add_argument('--solarPen', default=5, help='solar penetration percentage times 10')
    parser.add_argument('--evPen', default=5, help='EV penetration percentage times 10')
    parser.add_argument('--network', default='iowa', help='name of network to simulate')
    parser.add_argument('--days', default=30, help='number of days to simulate')

    FLAGS, unparsed = parser.parse_known_args()
    print('running with arguments: ({})'.format(FLAGS))

    path_networks = 'Networks/'
    path_absolute = './'

    # parse inputs
    storagePen, solarPen, evPen, dir_network, n_days, t_res = parse_inputs(FLAGS)

    # Define length of training set
    t_idx = t_res * 30  # 30 days of training data, t_res is number of points per day

    # load demand data
    demand = np.loadtxt(path_networks + dir_network + 'raw_demand.csv')
    # demand = np.loadtxt(path_networks + dir_network + 'demand_solar.csv')
    demand_solar_ev = np.loadtxt(path_networks + dir_network + 'demand_solar_ev.csv')
    demand_imag = np.loadtxt(path_networks + dir_network + 'raw_demand_imag.csv')

    # load storage data for perfect foresight controller
    data = np.load(path_networks + dir_network +
                   'storage_data' + str(solarPen) + str(storagePen) + str(evPen) + '.npz')
    sGenFull = data['sGenFull']
    nodes_storage = data['nodesStorage']
    qmin = data['qmin']
    qmax = data['qmax']
    umin = data['umin']
    umax = data['umax']

    # Split real and reactive power demand into training and test set
    demand_train, demand = train_test_split(demand, t_idx)
    demand_solar_ev_train, demand_solar_ev = train_test_split(demand_solar_ev, t_idx)
    demand_imag_train, demand_imag = train_test_split(demand_imag, t_idx)

    # load models
    data = np.load(path_networks + dir_network + 'model_coefs.npz')
    coefs_mat_v = data['coefs_mat_v']
    intercept_mat_v = data['intercept_mat_v']
    coefs_mat_tr = data['coefs_mat_tr']
    intercept_mat_tr = data['intercept_mat_tr']
    coefs_mat_ti = data['coefs_mat_ti']
    intercept_mat_ti = data['intercept_mat_ti']
    coefs_upper_i = data['coefs_upper_i']  # reactive power models
    coefs_lower_i = data['coefs_lower_i']  # first term is intercept second is coefficient

    # load voltage and transformer limits
    data = np.load(path_networks + dir_network + 'grid_data.npz')
    t_limits = data['t_limits']
    v_max = data['v_max']
    v_min = data['v_min']

    # initialize variables
    c_all = np.zeros((nodes_storage.size, demand.shape[1]))
    d_all = np.zeros((nodes_storage.size, demand.shape[1]))
    upper_supply = np.zeros((demand.shape[0], t_res * (n_days + 1)))
    lower_supply = np.zeros((demand.shape[0], t_res * (n_days + 1)))

    # daily loop
    for day in range(n_days):
        print('running day', day)

        # previous week historical data
        # use data with full solar and EV
        if day == 0:
            demand_prev = demand_solar_ev_train[:, -7 * t_res:]
        else:
            demand_prev = demand_solar_ev[:, (day - 1) * t_res:day * t_res]

        # extract perfect foresight data for current loop
        demand_day = demand[:, day * t_res:(day+1) * t_res]
        demand_solar_ev_day = demand_solar_ev[:, day * t_res:(day+1) * t_res]

        # Get demand bounds
        upper_demand = np.zeros((demand.shape[0], t_res))
        lower_demand = np.zeros((demand.shape[0], t_res))
        for node in range(demand.shape[0]):

            daily_demand = stack_daily(demand_prev[node, :], t_res)
            # demand bounds are max of past weeks data with solar and EV
            upper_demand[node, :], lower_demand[node, :] = get_demand_bounds(daily_demand)

        print(upper_demand.shape)
        #print(np.min(upper_demand-lower_demand))
        nodes_storage = np.arange(demand.shape[0])

        # get supply bounds
        upper_supply_d = np.zeros((demand.shape[0], t_res))
        lower_supply_d = np.zeros((demand.shape[0], t_res))
        for t in range(t_res):
            print('running time step:', t)
            # Global optimization
            upper_supply_t, lower_supply_t, delta_u, delta_l = global_optimization(upper_demand[:, t], lower_demand[:, t], nodes_storage,
                            coefs_mat_v, intercept_mat_v, coefs_mat_tr, intercept_mat_tr, coefs_mat_ti, intercept_mat_ti,
                            coefs_upper_i, coefs_lower_i, v_max, v_min, t_limits)

            upper_supply_d[:, t] = upper_supply_t
            lower_supply_d[:, t] = lower_supply_t

        upper_supply[:, t_res * day: t_res * (day + 1)] = upper_supply_d
        lower_supply[:, t_res * day: t_res * (day + 1)] = lower_supply_d

        np.savez(path_networks + dir_network + 'bound_data.npz', upper_supply=upper_supply, lower_supply=lower_supply)
        print('Saved bounds')

    plot_worst_bounds(upper_demand, lower_demand, demand_solar_ev[:, 0:n_days * t_res], demand_solar_ev[:, 0:n_days * t_res])




