import numpy as np
import pandas as pd
import argparse

import EV_Profiles as ev


def parse_inputs(FLAGS):
    """
    Parse the file input arguments

    :param `FLAGS`: data object containing all specified arguments

    :return: `storagePen, solarPen, seed, evPen, dir_network, load_fname_orig, n_days`
    """

    storagePen = float(FLAGS.storagePen) / 10
    solarPen = float(FLAGS.solarPen) / 10
    seed = int(FLAGS.seed)
    evPen = float(FLAGS.evPen) / 10
    n_days = int(FLAGS.days)

    if FLAGS.network == 'iowa':
        dir_network = 'Iowa_feeder/'
        load_fname_orig = 'Loads_orig.dss'
    elif FLAGS.network == '123':
        dir_network = 'IEEE123-1ph-test/'
        load_fname_orig = 'Load_orig.DSS'
    else:
        print('reverting to default Iowa network')
        dir_network = 'Iowa_feeder/'
        load_fname_orig = 'Loads_orig.dss'

    return storagePen, solarPen, seed, evPen, dir_network, load_fname_orig, n_days


def getLoads(path_network, load_fname):
    """
    Load the peak demand information from the OpenDSS network file

    :param
        `path_network`: path to network directory <br />
    :param
        `load_fname`:
            name of file containing load data <br />

    :return:
        `loads_real_np, loads_imag_np, load_size, load_nodes_np, load_volts_np`
    """

    loadfile = pd.read_csv(path_network + load_fname, sep=' ', header=None, usecols=range(9))

    load_volts = loadfile[5]  # 8 for SFO formatting, 6 for IEEE formatting, 5 for Iowa formatting
    loads_real = loadfile[6]
    loads_imag = loadfile[8]
    load_size = loads_real.shape
    load_nodes = loadfile[4]

    # remove units from load and bus from node name
    load_volts_np = np.zeros(load_size)
    loads_real_np = np.zeros(load_size)
    loads_imag_np = np.zeros(load_size)
    load_nodes_np = []
    for i in range(int(load_size[0])):
        load_volts_np[i] = str(load_volts[i])[3:]  # 3 for kV=
        loads_real_np[i] = str(loads_real[i])[3:]  # 3 for kW=
        loads_imag_np[i] = str(loads_imag[i])[5:]  # 5 for kvar=
        load_nodes_np.append(str(load_nodes[i])[5:])  # 5 for bus1=

    return loads_real_np, loads_imag_np, load_size, load_nodes_np, load_volts_np


def make_demand_data(data_path, peaks_real, n_days=60):
    """
    :param `data_path`: path to data directory <br />
    :param `peaks_real`: array of network peak demands <br />
    :param `n_days`: number of days in simulation

    """
    file_name = data_path

    t_res = 24 # points in day

    # stack the data from each sheet
    data = pd.read_excel(file_name, sheet_name='FeederA_P', header=0, index_col=0)
    dvs = data.values[0:t_res * n_days, :].T
    dvs = dvs[~np.all(dvs == 0, axis=1)]
    raw_demand = np.zeros((peaks_real.shape[0], t_res * n_days))
    raw_demand[0:dvs.shape[0], :] = dvs
    prev_idx = dvs.shape[0]

    data = pd.read_excel(file_name, sheet_name='FeederB_P', header=0, index_col=0)
    dvs = data.values[0:t_res * n_days, :].T
    dvs = dvs[~np.all(dvs == 0, axis=1)]
    raw_demand[prev_idx:prev_idx + dvs.shape[0], :] = dvs
    prev_idx += dvs.shape[0]

    data = pd.read_excel(file_name, sheet_name='FeederC_P', header=0, index_col=0)
    dvs = data.values[0:t_res * n_days, :].T
    dvs = dvs[~np.all(dvs == 0, axis=1)]
    raw_demand[prev_idx:prev_idx + dvs.shape[0], :] = dvs

    print(raw_demand)
    np.savetxt(data_path + 'raw_demand.csv', raw_demand)

    # repeat for reactive power
    data = pd.read_excel(file_name, sheet_name='FeederA_Q', header=0, index_col=0)
    dvs = data.values[0:t_res * n_days, :].T
    dvs = dvs[~np.all(dvs == 0, axis=1)]
    raw_demand_imag = np.zeros((peaks_real.shape[0], t_res * n_days))
    raw_demand_imag[0:dvs.shape[0], :] = dvs
    prev_idx = dvs.shape[0]

    data = pd.read_excel(file_name, sheet_name='FeederB_Q', header=0, index_col=0)
    dvs = data.values[0:t_res * n_days, :].T
    dvs = dvs[~np.all(dvs == 0, axis=1)]
    raw_demand_imag[prev_idx:prev_idx + dvs.shape[0], :] = dvs
    prev_idx += dvs.shape[0]

    data = pd.read_excel(file_name, sheet_name='FeederC_Q', header=0, index_col=0)
    dvs = data.values[0:t_res * n_days, :].T
    dvs = dvs[~np.all(dvs == 0, axis=1)]
    raw_demand_imag[prev_idx:prev_idx + dvs.shape[0], :] = dvs

    print(raw_demand_imag)
    np.savetxt(data_path + 'raw_demand_imag.csv', raw_demand_imag)

    return True

def load_demand_data_iowa(data_path):
    demand = np.loadtxt(data_path + 'raw_demand.csv')
    demand_imag = np.loadtxt(data_path + 'raw_demand_imag.csv')

    return demand, demand_imag

def setStorageSolar(pDemandFull, sNormFull, storagePen, solarPen, nodesPen, nodesLoad, n_days, garage_ids, daily_car_energy=0):
    """
    OLD:
    Inputs: pDemandFull - full matrix of real power demanded (nodes X time)
        sNormFull - full matrix of normalized solar data to be scaled and added to power demanded
        storagePen, solarPen, nodesPen - storage, solar, nodes penetration percentages
        rootIdx - index of the root node in the network
    Outputs: netDemandFull - full matrix of real net load
        sGenFull - full matrix of solar generation for storage nodes
        nodesLoad - list of nodes that have non-zero load
        nodesStorage - list of storage nodes
        qmin, qmax, umin, umax
    """

    # Pick storage nodes
    #nodesLoad = np.nonzero(pDemandFull[:, 0])[0]
    #if pDemandFull[rootIdx, 0] > 0:
    #    nodesLoad = np.delete(nodesLoad, np.argwhere(nodesLoad == rootIdx)) # remove substation node
    nodesStorage = np.random.choice(nodesLoad, int(np.rint(len(nodesLoad)*nodesPen)), replace=False)
    nodesStorage = np.append(nodesStorage, garage_ids)  # make sure the garage nodes have storage
    nodesStorage = np.unique(nodesStorage)

    # Assign solar
    loadSNodes = np.mean(pDemandFull[nodesStorage, :], 1)
    rawSolar = solarPen * (sum(np.mean(pDemandFull, 1)) + daily_car_energy/24)

    rawStorage = storagePen * (24*sum(np.mean(pDemandFull, 1)) + daily_car_energy)
    print('average network power', sum(np.mean(pDemandFull, 1)))
    alphas = loadSNodes/sum(loadSNodes)
    alphas = alphas.reshape(alphas.shape[0], 1)
    netDemandFull = pDemandFull

    # portionSolarPerNode represents the portion of solar in each node.
    #print(alphas.shape)

    # randomly increase or decrease solar generation at various nodes
    randomization = np.random.uniform(1-(1.0 - solarPen)/solarPen, 1+(1.0 - solarPen)/solarPen, size=alphas.shape)
    portionSolarPerNode = rawSolar * alphas * randomization
    #print(alphas)

    #np.reshape(portionSolarPerNode, (34, 0))
    #np.reshape(sNormFull, (0, 3673))

    # sGenFull is the full solar generation for all the nodes that were assigned storage.
    # sNormFull represents the shape of the solar generation over time.
    sGenFull = np.dot(portionSolarPerNode, sNormFull)
    # print(len(nodesLoad))
    # print(len(alphas))
    # print('s norm', sNormFull[:,5:20])
    # print(portionSolarPerNode)
    # We then select only the first two months of solar data.
    sGenFull = sGenFull[:, :(n_days) * 24]
    netDemandFull[nodesStorage, :] = netDemandFull[nodesStorage, :] - sGenFull

    # Assign storage
    qmax = np.multiply(rawStorage, alphas)
    qmin = np.zeros_like(qmax)
    umax = qmax/3 # it takes 3 hours to fully charge batteries
    umin = -umax

    return netDemandFull, sGenFull, nodesStorage, qmin, qmax, umin, umax


def test_solar_ratios(demand_ev, demand_solar_ev):
    print(np.mean(demand_solar_ev, axis=1) / np.mean(demand_ev, axis=1))

    return True


def make_EV_data(charging_power, demand, evPen, peaks_real, data_path, n_days):
    # total number of EVs in network at 100%
    total_EVs, n_EV = ev.get_total_EVs(demand, evPen)
    # get the ids for the number of EVs for each node
    EVs_per_node, garage_ids, home_ids = ev.assign_cars_to_nodes(peaks_real, total_EVs, garage_threshold=25)
    print('garage_ids', garage_ids)
    # initialize output dictionaries
    start_dict, end_dict, charge_dict = ev.initialize_dictionaries(nodes=len(EVs_per_node), days=n_days)
    # generate data
    start_dict, end_dict, charge_dict = ev.generate_EV_data(n_days, n_EV, start_dict, end_dict, charge_dict,
                                                            garage_ids, home_ids, EVs_per_node, charging_power)
    # save EV data
    np.savez(data_path + 'EV_charging_data' + str(evPen), start_dict=start_dict, end_dict=end_dict, charge_dict=charge_dict,
             charging_power=charging_power, evPen=evPen, garage_ids=garage_ids)
    print('saved EV data')
    return True


def add_ev_profiles(T, demand, start_dict, end_dict, garage_ids, charging_power_garage, charging_power_home, n_days):
    # T = hours in horizon
    ev_charge_profiles = np.zeros((demand.shape[0], demand.shape[1] + int(T / 2)))
    for i in range(n_days):
        for node in range(demand.shape[0]):
            t_arrival = start_dict[node][i]
            t_depart = end_dict[node][i]
            t_depart = np.array(np.round((t_depart - t_arrival) / 1.5), dtype=int) + t_arrival # remove LC buffer time of 1.5
            if node in garage_ids:
                t_depart = t_arrival + 1 # fast charging garages can fully charge within 1 hour
                total_charge = ev.get_EV_charging_window(t_arrival, t_depart, T, charging_power_garage)
            else:
                total_charge = ev.get_EV_charging_window(t_arrival, t_depart, T, charging_power_home)
            ev_charge_profiles[node, i * int(T / 2):i * int(T / 2) + T] = total_charge
    ev_charge_profiles = ev_charge_profiles[:, 0:-int(T / 2)]

    return demand + ev_charge_profiles

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate solar and storage and demand data')
    parser.add_argument('--seed', default=0, help='random seed')
    parser.add_argument('--storagePen', default=1, help='storage penetration percentage times 10')
    parser.add_argument('--solarPen', default=5, help='solar penetration percentage times 10')
    parser.add_argument('--aggregate', default=0, help='boolean for aggregating homes to network')
    parser.add_argument('--evPen', default=5, help='EV penetration percentage times 10')
    parser.add_argument('--network', default='iowa', help='name of network to simulate')
    parser.add_argument('--days', default=60, help='number of days to simulate')

    FLAGS, unparsed = parser.parse_known_args()
    print('running with arguments: ({})'.format(FLAGS))

    path_networks = './Networks/'

    # parse inputs
    storagePen, solarPen, seed, evPen, dir_network, load_fname_orig, n_days = parse_inputs(FLAGS)

    np.random.seed(seed)  # set random seed

    # load network peak demands
    peaks_real, peaks_imag, load_size, load_nodes, load_volts = \
        getLoads_Iowa(path_networks + dir_network, load_fname_orig)

    # make raw demand data
    if FLAGS.aggregate:
        raw_demand_flag = make_demand_data_iowa(path_networks + dir_network, peaks_real, n_days=n_days)

    # load raw demand data after it has been made the first time
    demand, demand_imag = load_demand_data_iowa(path_networks + dir_network)

    # Assign EV profiles
    charging_power_garage = 40 # for garages
    charging_power_home = 6 # for homes
    if FLAGS.aggregate:
        ev_data_flag = make_EV_data(charging_power_home, demand, evPen, peaks_real+0.01, path_networks + dir_network, n_days)

    # load EV data
    data = np.load(path_networks + dir_network + 'EV_charging_data' + str(evPen) + '.npz', allow_pickle=True)
    start_dict = data['start_dict'][()]
    end_dict = data['end_dict'][()]
    charge_dict = data['charge_dict'][()]
    garage_ids = data['garage_ids'][()]

    # Add uncontrolled EV profiles to demand
    T = 48 # 48 hours in horizon for hourly data like Iowa
    demand_ev = add_ev_profiles(T, demand,
                                start_dict, end_dict, garage_ids, charging_power_garage, charging_power_home, n_days)

    # Assign solar
    solar_per_node = 0.7  # average percentage of self power generation
    nodesPen = np.clip(np.max((solarPen, storagePen)) / solar_per_node, None, 1.0)
    solar_dict = np.load('./data/solar_ramp_data.npz')
    sr_NormFull = solar_dict['sNormFull'] # hourly resolution with mean power = 1
    demand_ev1 = np.copy(demand_ev)
    demand_solar_ev, sGenFull, nodesStorage, \
    qmin, qmax, umin, umax = setStorageSolar(demand_ev1, sr_NormFull, storagePen, solarPen,
                                             nodesPen, np.arange(peaks_real.size), n_days, garage_ids, daily_car_energy=0)

    demand_solar = np.zeros(demand.shape) + demand
    demand_solar[nodesStorage, :] = demand[nodesStorage, :] - sGenFull

    #test_solar_ratios(demand, demand_solar)

    # save data
    np.savez(path_networks + dir_network + 'storage_data' + str(solarPen) + str(storagePen) + str(evPen),
             sGenFull=sGenFull, nodesStorage=nodesStorage, qmin=qmin, qmax=qmax,
             umin=umin, umax=umax, storagePen=storagePen, solarPen=solarPen, nodesPen=nodesPen
             )

    np.savetxt(path_networks + dir_network + 'demand_solar_ev.csv', demand_solar_ev)
    np.savetxt(path_networks + dir_network + 'demand_solar.csv', demand_solar)

    print('saved data')

