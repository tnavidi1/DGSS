import numpy as np
import pandas as pd
import argparse

import EV_Profiles as ev


def get_network_type(network_name):
    if network_name == 'iowa/':
        network_type = 'iowa'
    elif network_name == 'sacramento/':
        network_type = 'IEEE'
    elif network_name == 'arizona/':
        network_type = 'IEEE'
    elif network_name == 'vermont/':
        network_type = 'vermont'
    else:
        network_type = 'SFO'

    return network_type


def get_load_bases(path_network, load_fname, network_type):
    if network_type == 'SFO':
        idx = 8
        sep = ' '
    elif network_type == 'IEEE':
        idx = 7
        sep = ' '
    elif network_type == 'iowa':
        idx = 6
        sep = ' '
    elif network_type == 'vermont':
        idx = 5
        sep = '\t'
    else:
        print('network type not recognized')
        idx = 8
        sep = ' '

    loadfile = pd.read_csv(path_network + load_fname, sep=sep, header=None, usecols=range(9), engine='python')
    loads_real_str = np.array(loadfile[idx], dtype=str)  # 6 for Iowa, 7 for IEEE, 8 for SFO, 5 for Vermont
    # loads_imag_str  = loadfile[8]  # 8 for Iowa and IEEE, 9 for SFO, 6 for Vermont

    loads = np.array([l[3:] for l in loads_real_str], dtype=float)

    return loads


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


def assign_load_profiles(network_base, network_path, network_type):
    # need to keep track of which node is commercial and which is residential

    loads = pd.read_csv(network_path + 'res_load_data.csv').values
    loads_c = pd.read_csv(network_path + 'com_load_data.csv').values
    t_res = 0.25  # hours per measurement

    spot_loads = network_base
    homeData = loads
    print(homeData.shape)

    maxsum = np.zeros((1, homeData.shape[0]))

    if network_type == 'agg':

        N = len(spot_loads)  # total number of buses
        Lbuses = np.where(spot_loads != 0)[0]  # indexes of load buses
        print('load bus indices', Lbuses)
        print('number of load buses', len(Lbuses))

        NHomes = np.zeros((N, 1))  # number of homes for each load bus
        pDemand = np.zeros((N, homeData.shape[1]))  # power demanded by each load bus
        load_type = np.zeros(N)  # type of load for each node either residential=0 or commercial=1

        for i in range(len(Lbuses)):
            currpeak = 0
            currload = np.zeros((1, homeData.shape[1]))
            nextload = np.zeros((1, homeData.shape[1]))

            while currpeak < spot_loads[Lbuses[i]]:
                if spot_loads[Lbuses[i]] > 63:
                    homeData = loads_c
                    load_type[i] = 1  # 1 for commercial 0 for residential
                else:
                    homeData = loads

                NHomes[i] = NHomes[i] + 1
                currload = nextload
                homeIdx = np.random.randint(homeData.shape[0])  # , size=(1,1))

                if np.isnan(np.sum(homeData[homeIdx, :])):
                    # check if data is valid.
                    # if it is Nan, then resample until it is valid
                    while np.isnan(np.sum(homeData[homeIdx, :])):
                        homeIdx = np.random.randint(homeData.shape[0])
                    # end while loop
                # end if
                nextload = currload + homeData[homeIdx, :]

                currpeak = average_daily_peak(nextload[0, :], t_res)
                currload = nextload
            # end while loop
            print('average daily peak for load bus', i, currpeak)
            pDemand[Lbuses[i], :] = currload / currpeak * spot_loads[Lbuses[i]]
            print('new average daily peak', average_daily_peak(pDemand[Lbuses[i], :], t_res))
        # end for loop
        print('max', np.max(pDemand, axis=1))

    return pDemand, load_type


def close_idx(lst, k):

    idx = (np.abs(lst - k)).argmin()
    return idx


def make_demand_data(name, demand_p):
    pf = np.random.uniform(0.9, 0.95)

    ccl = np.loadtxt(name + 'com_cool_load_data.csv', delimiter=',')
    chl = np.loadtxt(name + 'com_heat_load_data.csv', delimiter=',')
    ctl = np.loadtxt(name + 'com_total_load_data.csv', delimiter=',')
    cwl = np.loadtxt(name + 'com_wheat_load_data.csv', delimiter=',')
    rcl = np.loadtxt(name + 'res_cool_load_data.csv', delimiter=',')
    rhl = np.loadtxt(name + 'res_heat_load_data.csv', delimiter=',')
    rtl = np.loadtxt(name + 'res_total_load_data.csv', delimiter=',')
    rwl = np.loadtxt(name + 'res_wheat_load_data.csv', delimiter=',')

    nr = rtl.shape[0]
    nc = ctl.shape[0]

    total = np.vstack((rtl, ctl))
    total_p = np.max(total, axis=1)

    demand = np.zeros((demand_p.size, rtl.shape[1]))
    d_id = 0
    com_ids = []
    res_ids = []

    for peak in demand_p:
        idx = close_idx(total_p, peak)
        if idx > nr:
            com_ids.append(idx)
        else:
            res_ids.append(idx)
        demand[d_id, :] = total[idx, :] / total_p[idx] * peak
        d_id += 1

    demand_imag = np.tan(np.arccos(pf)) * demand

    np.savez(name + 'demand_data.npz', demand=demand, demand_imag=demand_imag, res_ids=res_ids, com_ids=com_ids)

    return demand, demand_imag, res_ids, com_ids


def load_demand_data(data_path):
    data = np.load(data_path + 'demand_data.npz')
    demand = data['demand']
    demand_imag = data['demand_imag']
    res_ids = data['res_ids']
    com_ids = data['com_ids']

    return demand, demand_imag, res_ids, com_ids


def load_demand_data_flex(data_path):
    data = np.load(data_path + 'demand_data.npz')
    demand = data['demand']
    demand_imag = data['demand_imag']
    res_ids = data['res_ids']
    com_ids = data['com_ids']
    demand_base = data['demand_base']
    demand_flex = data['demand_flex']

    return demand_base, demand, demand_imag, demand_flex, res_ids, com_ids


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
    #print('average network power', sum(np.mean(pDemandFull, 1)))
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
    netDemandFull[nodesStorage, :] = netDemandFull[nodesStorage, :] - sGenFull

    # Assign storage
    qmax = np.multiply(rawStorage, alphas)
    qmin = np.zeros_like(qmax)
    umax = qmax/2 # it takes 2 hours to fully charge batteries
    umin = -umax

    return netDemandFull, sGenFull, nodesStorage, qmin, qmax, umin, umax


def test_solar_ratios(demand_ev, demand_solar_ev):
    print(np.mean(demand_solar_ev, axis=1) / np.mean(demand_ev, axis=1))

    return True


def make_EV_data(charging_power, demand, evPen, peaks_real, data_path, n_days):
    # total number of EVs in network at 100%
    total_EVs, n_EV = ev.get_total_EVs(demand, evPen)
    print('total EVs', total_EVs)
    # get the ids for the number of EVs for each node
    EVs_per_node, garage_ids, home_ids = ev.assign_cars_to_nodes(peaks_real, total_EVs, garage_threshold=25)
    print('EVs per node', EVs_per_node)
    #print('garage_ids', garage_ids)
    # initialize output dictionaries
    start_dict, end_dict, charge_dict = ev.initialize_dictionaries(nodes=len(EVs_per_node), days=n_days)
    # generate data
    start_dict, end_dict, charge_dict = ev.generate_EV_data(n_days, n_EV, start_dict, end_dict, charge_dict,
                                                            garage_ids, home_ids, EVs_per_node, charging_power)
    # save EV data
    np.savez(data_path + 'EV_charging_data', start_dict=start_dict, end_dict=end_dict, charge_dict=charge_dict,
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


def make_flex_data(name, demand_p, res_pens, com_pens):
    spacePen = res_pens[res_pens.year == year]['space'].values
    waterPen = res_pens[res_pens.year == year]['water'].values
    otherPen = res_pens[res_pens.year == year]['other'].values
    spacePen_com = com_pens[res_pens.year == year]['space'].values
    waterPen_com = com_pens[res_pens.year == year]['water'].values
    otherPen_com = com_pens[res_pens.year == year]['other'].values
    eff = 0.81  # space heat
    cop = 3.3
    weff = 0.66  # water heat
    wcop = 3.45

    pf = np.random.uniform(0.9, 0.95)

    ccl = np.loadtxt(name + 'com_cool_load_data.csv', delimiter=',')
    chl = np.loadtxt(name + 'com_heat_load_data.csv', delimiter=',')
    ctl = np.loadtxt(name + 'com_total_load_data.csv', delimiter=',')
    cwl = np.loadtxt(name + 'com_wheat_load_data.csv', delimiter=',')
    rcl = np.loadtxt(name + 'res_cool_load_data.csv', delimiter=',')
    rhl = np.loadtxt(name + 'res_heat_load_data.csv', delimiter=',')
    rtl = np.loadtxt(name + 'res_total_load_data.csv', delimiter=',')
    rwl = np.loadtxt(name + 'res_wheat_load_data.csv', delimiter=',')
    chl = chl * eff / cop
    rhl = rhl * eff / cop
    cwl = cwl * weff / wcop
    rwl = rwl * weff / wcop

    nr = rtl.shape[0]
    nc = ctl.shape[0]

    total = np.vstack((rtl, ctl))
    total_p = np.max(total, axis=1)

    demand = np.zeros((demand_p.size, rtl.shape[1]))
    d_id = 0
    com_ids = []
    res_ids = []
    idxs = []
    scales = []

    for peak in demand_p:
        idx = close_idx(total_p, peak)
        idxs.append(idx)
        if idx > nr:
            com_ids.append(d_id)
        else:
            res_ids.append(d_id)
        scale = total_p[idx] * peak
        demand[d_id, :] = total[idx, :] / scale
        scales.append(scale)
        d_id += 1

    demand[res_ids, :] = demand[res_ids, :] * (1 + otherPen / 100)
    demand[com_ids, :] = demand[com_ids, :] * (1 + otherPen_com / 100)

    demand_base = np.copy(demand)

    #order = np.arange(demand_p.size)
    order = np.array(res_ids)
    np.random.shuffle(order)
    sp = 0
    i = 0
    while sp < spacePen/100:
        if i == order.size:
            break
        node = order[i]
        if np.sum(rhl[idxs[node], :]) > 0:
            demand_base[node, :] += rhl[idxs[node], :] / scales[node]
            sp += np.sum(rhl[idxs[node], :]) / scales[node] / np.sum(demand_base)
        i += 1
    print('sp', sp, spacePen)

    order = np.array(com_ids)
    np.random.shuffle(order)
    sp = 0
    i = 0
    while sp < spacePen_com/100:
        if i == order.size:
            break
        node = order[i]
        if np.sum(chl[idxs[node] - nr, :]) > 0:
            demand_base[node, :] += chl[idxs[node] - nr, :] / scales[node]
            sp += np.sum(chl[idxs[node] - nr, :]) / scales[node] / np.sum(demand_base)
        i += 1
    print('sp', sp, spacePen_com)

    order = np.array(res_ids)
    np.random.shuffle(order)
    wp = 0
    i = 0
    while wp < waterPen/100:
        if i == order.size:
            break
        node = order[i]
        if np.sum(rwl[idxs[node], :]) > 0:
            demand_base[node, :] += rwl[idxs[node], :] / scales[node]
            wp += np.sum(rwl[idxs[node], :]) / scales[node] / np.sum(demand_base)
        i += 1
    print('wp', wp, waterPen)

    order = np.array(com_ids)
    np.random.shuffle(order)
    wp = 0
    i = 0
    while wp < waterPen_com/100:
        if i == order.size:
            break
        node = order[i]
        if np.sum(cwl[idxs[node] - nr, :]) > 0:
            demand_base[node, :] += cwl[idxs[node] - nr, :] / scales[node]
            wp += np.sum(cwl[idxs[node] - nr, :]) / scales[node] / np.sum(demand_base)
        i += 1
    print('wp', wp, waterPen_com)

    for node in res_ids:
        demand[node, :] += -rcl[idxs[node]] / scales[node]

    for node in com_ids:
        demand[node, :] += -ccl[idxs[node] - nr] / scales[node]

    demand_imag = np.tan(np.arccos(pf)) * demand

    demand_flex = demand_base - demand

    np.savez(name + 'demand_data.npz', demand_base=demand_base, demand=demand, demand_imag=demand_imag,
             demand_flex=demand_flex, res_ids=res_ids, com_ids=com_ids)

    return demand_base, demand, demand_imag, demand_flex, res_ids, com_ids


def make_tariffs(pp, ppp, op, pd, ppd1, ppd2, ps, t_res):
    if ps - ppd1 < 0:
        tariff = np.concatenate((op * np.ones(int(24 / t_res + ps - ppd1)), ppp * np.ones(ppd1),
                                 pp * np.ones(pd), ppp * np.ones(ppd2)))
        tariff2 = np.concatenate((op * np.ones(int(24 / t_res - pd - ppd2 - ppd1)), ppp * np.ones(ppd1),
                                  pp * np.ones(pd), ppp * np.ones(ppd2)))
        tariff = np.concatenate((tariff, tariff2))
        tariff = tariff[int(24 / t_res):int(2 * (24 / t_res))]
    elif 24 / t_res - ps - pd - ppd2 < 0:
        tariff = np.concatenate((op * np.ones(ps - ppd1), ppp * np.ones(ppd1), pp * np.ones(pd),
                                 ppp * np.ones(ppd2)))
        tariff2 = np.concatenate((op * np.ones(int(24 / t_res - pd - ppd2 - ppd1)), ppp * np.ones(ppd1),
                                  pp * np.ones(pd), ppp * np.ones(ppd2)))
        tariff = np.concatenate((tariff, tariff2))
        tariff = tariff[int(24 / t_res):int(2 * (24 / t_res))]
    else:
        tariff = np.concatenate((op * np.ones(ps - ppd1), ppp * np.ones(ppd1), pp * np.ones(pd),
                        ppp * np.ones(ppd2), op * np.ones(int(np.maximum(24 / t_res - ps - pd - ppd2, 0)))))

    return tariff


def define_tariffs(peak_hour, t_res):
    # residential price
    pp = 0.42  # peak price
    ppp = 0
    op = 0.36  # of peak price
    pd = int(5 / t_res)  # peak duration
    ppd1 = 0
    ppd2 = 0
    ps = int(peak_hour / t_res - pd/2)  # peak start
    res = make_tariffs(pp, ppp, op, pd, ppd1, ppd2, ps, t_res)

    # residential EV
    pp = 0.5  # peak price
    ppp = 0.39  # part peak price
    op = 0.19  # of peak price
    pd = int(5 / t_res)  # peak duration
    ppd1 = int(1 / t_res)  # first part peak duration
    ppd2 = int(3 / t_res)  # second part peak duration
    ps = int(peak_hour / t_res - pd/2)  # peak start
    res_ev = make_tariffs(pp, ppp, op, pd, ppd1, ppd2, ps, t_res)

    # commercial
    pp = 0.37  # peak price
    ppp = 0  # part peak price
    op = 0.35  # of peak price
    pd = int(5 / t_res)  # peak duration
    ppd1 = 0  # first part peak duration
    ppd2 = 0  # second part peak duration
    ps = int(peak_hour / t_res - pd / 2)  # peak start
    comm = make_tariffs(pp, ppp, op, pd, ppd1, ppd2, ps, t_res)

    # commercial > 75kW
    pp = 0.16  # peak price
    ppp = 0.13  # part peak price
    op = 0.11  # of peak price
    pd = int(5 / t_res)  # peak duration
    ppd1 = int(2 / t_res)  # first part peak duration
    ppd2 = int(2 / t_res)  # second part peak duration
    ps = int(peak_hour / t_res - pd / 2)  # peak start
    comm_75 = make_tariffs(pp, ppp, op, pd, ppd1, ppd2, ps, t_res)

    # commercial EV
    pp = 0.33  # peak price
    ppp = 0.13  # part peak price
    op = 0.11  # of peak price
    pd = int(5 / t_res)  # peak duration
    ppd1 = int(2 / t_res)  # first part peak duration
    ppd2 = int(12 / t_res)  # second part peak duration
    ps = int(peak_hour / t_res - pd / 2)  # peak start
    comm_ev = make_tariffs(pp, ppp, op, pd, ppd1, ppd2, ps, t_res)

    tariffs = np.stack((res, res_ev, comm, comm_75, comm_ev))
    #print(tariffs.shape)
    return tariffs


if __name__ == '__main__':
    name = 'rural_san_benito/'
    year = 2050
    seed = int(np.random.uniform(0, 100000))
    network_name = name + 'network/'
    network_type = get_network_type(name)
    load_fname_orig = 'Loads_orig.dss'
    make_loads = False  # should be true when running for the first time
    solar_fname = 'solar_15min.csv'
    make_EV = False  # should be true when running for the first time
    peak_hour = 19
    t_res = 0.25  # hours

    res_pens = pd.read_csv(name + 'Residential_penetrations.csv')
    com_pens = pd.read_csv(name + 'Commercial_penetrations.csv')
    solarPen = res_pens[res_pens.year == year]['solar'].values / 100
    storagePen = res_pens[res_pens.year == year]['storage'].values / 100
    evPen = res_pens[res_pens.year == year]['ev'].values / 100
    flexPen = res_pens[res_pens.year == year]['flexibility enhanced'].values / 100

    np.random.seed(seed)  # set random seed

    # load network peak demands
    demand_p = get_load_bases(network_name, load_fname_orig, network_type)
    #print(demand_p)

    # make demand data
    if make_loads:
        # demand, demand_imag, res_ids, com_ids = make_demand_data(name, demand_p)
        # Assign flex load profiles
        demand, demand_raw, demand_imag, demand_flex, res_ids, com_ids = make_flex_data(
            name, demand_p, res_pens, com_pens)
    else:
        # demand, demand_imag, res_ids, com_ids = load_demand_data(name)
        demand, demand_raw, demand_imag, demand_flex, res_ids, com_ids = load_demand_data_flex(name)
    #print(np.mean(demand, axis=1))

    # Assign EV profiles
    charging_power_garage = 6.3 # for garages
    charging_power_home = 6.3 # for homes
    n_days = 365
    if make_EV:
        ev_data_flag = make_EV_data(charging_power_home, demand, evPen, demand_p+0.01, name, n_days)

    # load EV data
    data = np.load(name + 'EV_charging_data' + '.npz', allow_pickle=True)
    start_dict = data['start_dict'][()]
    end_dict = data['end_dict'][()]
    charge_dict = data['charge_dict'][()]
    garage_ids = data['garage_ids'][()]
    # keys are node index 0-:
    # values are list of arrays with each array containing info for a single day
    # array of start times = start_dict[node][day]

    # Add uncontrolled EV profiles to demand for purpose of determining extra energy from EVs
    T = 48 * 4
    demand_ev = add_ev_profiles(T, demand,
                                start_dict, end_dict, garage_ids, charging_power_garage, charging_power_home, n_days)

    flex_energy = flexPen * np.sum(demand_ev, axis=1)
    phis = np.clip(flex_energy / np.sum(demand_flex, axis=1), None, 1)
    #print(phis.shape)

    # Assign solar
    solar_per_node = np.random.uniform(0.4, 0.9)  # average percentage of self power generation
    nodesPen = np.clip(np.max((solarPen, storagePen)) / solar_per_node, None, 1.0)
    sr_NormFull = np.loadtxt(name + solar_fname)
    sr_NormFull = (sr_NormFull / np.mean(sr_NormFull)).reshape((1, sr_NormFull.size))
    demand_ev1 = np.copy(demand_ev)
    demand_solar_ev, sGenFull, nodesSolar, \
    qmin, qmax, umin, umax = setStorageSolar(demand_ev1, sr_NormFull, storagePen, solarPen,
                                             nodesPen, np.arange(demand_p.size), n_days, garage_ids, daily_car_energy=0)

    demand_solar = np.zeros(demand.shape) + demand_raw
    demand_solar[nodesSolar, :] = demand_raw[nodesSolar, :] - sGenFull
    nodesStorage = np.random.choice(nodesSolar, int(nodesSolar.size * storagePen), replace=False)
    #print(nodesStorage)
    #print(nodesSolar)

    #test_solar_ratios(demand, demand_solar)

    tariffs = define_tariffs(peak_hour, t_res)  # tariffs[0:end,:] = res, res ev[1,:], comm, comm75, comm ev

    # save data
    np.savez(name + 'storage_data',
             sGenFull=sGenFull, nodesStorage=nodesStorage, qmin=qmin, qmax=qmax, phis=phis,
             umin=umin, umax=umax, storagePen=storagePen, nodesSolar=nodesSolar, solarPen=solarPen, nodesPen=nodesPen
             )

    np.savetxt(name + 'demand_solar_ev.csv', demand_solar_ev)
    np.savetxt(name + 'demand_solar.csv', demand_solar)
    np.savetxt(name + 'demand_flex.csv', demand_flex)
    np.savetxt(name + 'tariffs.csv', tariffs)

    print('saved data')

