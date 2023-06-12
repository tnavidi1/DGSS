import numpy as np
import pandas as pd
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
    metrics = int(FLAGS.metrics)

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

    return storagePen, solarPen, evPen, dir_network, n_days, t_res, metrics


def train_test_split(data, t_idx):
    data_train = data[:, 0:t_idx]
    data_test = data[:, t_idx:]

    return data_train, data_test


def get_t_vio_metric(t_profile_real, t_profile_imag, t_limits):
    """
    Calculate transformer violation metric as sum of deviations above the limits as percentage of limits
    :param t_profile_real:
    :param t_profile_imag:
    :param t_limits:
    :return: transformer violation metric and number of violations
    """
    limits_expanded = np.tile(t_limits.reshape((t_limits.size, 1)), (1, t_profile_real.shape[1]))
    vio_where = (t_profile_real ** 2 + t_profile_imag ** 2 - limits_expanded**2) > 0
    vio = np.sqrt(t_profile_real[vio_where] ** 2 + t_profile_imag[vio_where] ** 2) - limits_expanded[vio_where]
    vio = np.sum(vio / limits_expanded[vio_where] * 100)
    num_vio = np.sum(t_profile_real**2 + t_profile_imag**2 - np.tile(t_limits.reshape((535,1))**2, (1,720)) > 0)
    return vio, num_vio


def get_v_vio_metric(v_profile, v_max, v_min):
    max_expanded = np.tile(v_max.reshape((v_max.size, 1)), (1, v_profile.shape[1]))
    min_expanded = np.tile(v_min.reshape((v_min.size, 1)), (1, v_profile.shape[1]))
    vio = np.sum((v_profile - max_expanded).clip(min=0) + (min_expanded - v_profile).clip(min=0))
    num_vio = np.sum(v_profile > max_expanded) + np.sum(v_profile < min_expanded)
    return vio, num_vio

def plot_extreme_bounds(upper_supply, lower_supply):
    node = np.unravel_index(lower_supply.argmax(), lower_supply.shape)[0]
    plt.figure()
    plt.plot(upper_supply[node, :].T)
    plt.plot(lower_supply[node, :].T)

    node = np.unravel_index(upper_supply.argmin(), upper_supply.shape)[0]
    plt.figure()
    plt.plot(upper_supply[node, :].T)
    plt.plot(lower_supply[node, :].T)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train data driven models')
    parser.add_argument('--opt', default=1, help='Type of optimization 0 is perfect foresight, 1 is bounds')
    parser.add_argument('--storagePen', default=1, help='storage penetration percentage times 10')
    parser.add_argument('--solarPen', default=5, help='solar penetration percentage times 10')
    parser.add_argument('--evPen', default=5, help='EV penetration percentage times 10')
    parser.add_argument('--network', default='iowa', help='name of network to simulate')
    parser.add_argument('--days', default=30, help='number of days to simulate')
    parser.add_argument('--metrics', default=0, help='0=optimization results, 1=grid metrics')

    FLAGS, unparsed = parser.parse_known_args()
    print('running with arguments: ({})'.format(FLAGS))

    path_networks = 'Networks/'
    path_absolute = './'

    # parse inputs
    storagePen, solarPen, evPen, dir_network, n_days, t_res, metrics = parse_inputs(FLAGS)

    # Define length of training set
    t_idx = t_res * 90  # 30 days of training data, t_res is number of points per day

    # load data
    #demand = np.loadtxt(path_networks + dir_network + 'raw_demand.csv')
    #demand = np.loadtxt(path_networks + dir_network + 'demand_solar.csv')
    demand = np.loadtxt(path_networks + dir_network + 'demand_solar_ev.csv')
    demand_imag = np.loadtxt(path_networks + dir_network + 'raw_demand_imag.csv')

    # Split real and reactive power demand into training and test set
    demand_train, demand = train_test_split(demand, t_idx)
    #demand_solar_ev_train, demand_solar_ev = train_test_split(demand_solar_ev, t_idx)
    demand_imag_train, demand_imag = train_test_split(demand_imag, t_idx)

    data = np.load(path_networks + dir_network + 'bound_data.npz')
    upper_supply = data['upper_supply']
    lower_supply = data['lower_supply']

    lower_supply = lower_supply[:, 0: n_days * t_res]
    upper_supply = upper_supply[:, 0: n_days * t_res]

    # demand with LCs
    demand_new = demand[:,0: n_days*t_res].clip(lower_supply[:,0: n_days*t_res], upper_supply[:,0: n_days*t_res])

    if metrics == 0:
        np.savetxt(path_networks + dir_network + 'demand_storage.csv', demand_new)

    elif metrics == 1:
        # load grid data
        v_profile_0 = np.loadtxt(path_absolute + path_networks + dir_network + 'v_profile_stor.csv')
        v_profile = np.loadtxt(path_absolute + path_networks + dir_network + 'v_profile.csv')[:,0:720]

        t_profile_real_0 = np.loadtxt(path_absolute + path_networks + dir_network + 't_profile_real_stor.csv')
        t_profile_imag_0 = np.loadtxt(path_absolute + path_networks + dir_network + 't_profile_imag_stor.csv')
        t_profile_real = np.loadtxt(path_absolute + path_networks + dir_network + 't_profile_real.csv')[:,0:720]
        t_profile_imag = np.loadtxt(path_absolute + path_networks + dir_network + 't_profile_imag.csv')[:,0:720]

        taps_profile = np.loadtxt(path_absolute + path_networks + dir_network + 'taps_profile_stor.csv')

        # load voltage and transformer limits
        data = np.load(path_networks + dir_network + 'grid_data.npz')
        t_limits = data['t_limits']
        v_max = data['v_max']
        v_min = data['v_min']

        #plot v and t by node
        """
        limits_expanded = np.tile(t_limits.reshape((t_limits.size, 1)), (1, t_profile_real.shape[1]))
        t_profile_0 = (np.sqrt(t_profile_real_0 ** 2 + t_profile_imag_0 ** 2)/limits_expanded)#[400:,:]
        t_profile = (np.sqrt(t_profile_real ** 2 + t_profile_imag ** 2)/limits_expanded)#[400:,:]
        x = np.tile(np.arange(t_profile.shape[0]).reshape((t_profile.shape[0], 1)), (1, t_profile.shape[1]))

        plt.figure()
        plt.scatter(x, t_profile_0.clip(max=1), s=2)
        plt.scatter(x, 1.2*t_profile, s=2)
        plt.plot(np.arange(t_profile.shape[0]), 1*np.ones(t_profile.shape[0]))
        plt.ylabel('Transformer apparent power')
        plt.xlabel('Transformer ID')
        plt.legend(('transformer capacity', '2020', '2050 LCs'))
        plt.show()
        """
