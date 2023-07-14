import numpy as np
import pandas as pd
import os

from opendss_interface import *


def load_demand_data_flex(data_path):
    data = np.load(data_path + 'demand_data.npz')
    demand = data['demand']
    demand_imag = data['demand_imag']
    res_ids = data['res_ids']
    com_ids = data['com_ids']
    demand_base = data['demand_base']
    demand_flex = data['demand_flex']

    return demand_base, demand, demand_imag, demand_flex, res_ids, com_ids


def train_test_split(data, t_idx):
    data_train = data[:, 0:t_idx]
    data_test = data[:, t_idx:]

    return data_train, data_test


def assign_transformer_limits(trans_base):
    transformer_limits = np.max(trans_base, axis=1) * 1.2
    #print(transformer_limits)
    #print(transformer_limits.shape)

    return transformer_limits


def assign_voltage_limits(v_base):
    v_max = np.ones(v_base.shape[0]) * 1.05
    v_min = np.ones(v_base.shape[0]) * 0.95

    return v_max, v_min


def get_t_vio_metric(t_profile, t_limits, t_res=0.25):
    window = int(2 / t_res)
    vios = np.zeros(t_profile.shape[0])
    for t in range(t_profile.shape[1] - window):
        vio = np.mean(t_profile[:, t:t+window], axis=1) > t_limits
        vios += vio

    return vios


def get_v_vio_metric(v_profile, v_max, v_min, t_res=0.25):
    v_profile, v_max, v_min = clean_voltages(v_profile, v_max, v_min)
    window = int(1 / t_res)
    vios = np.zeros(v_profile.shape[0])
    for t in range(v_profile.shape[1] - window):
        vio = np.mean(v_profile[:, t:t + window], axis=1) > v_max
        vios += vio
        vio = np.mean(v_profile[:, t:t + window], axis=1) < v_min
        vios += vio

    return vios


def clean_voltages(v_profile, v_max, v_min):
    connected = np.min(v_profile, axis=1) > 0.01
    v_profile = v_profile[connected, :]
    v_max = v_max[connected]
    v_min = v_min[connected]

    # align substation transformer tap
    v_profile = v_profile - np.mean(v_profile[0, :]) + 1

    return v_profile, v_max, v_min


if __name__ == '__main__':
    # name = 'sacramento/'
    # name = 'iowa/'
    # name = 'central_SF/'
    # name = 'commercial_SF/'
    # name = 'tracy/'
    name = 'rural_san_benito/'
    # name = 'los_banos/'
    # name = 'vermont/'
    # name = 'arizona/'
    # name = 'marin/'
    # name = 'oakland/'
    # year = 2020
    year = 2050
    controller = 'local'
    # controller = 'central'

    cwd = os.getcwd()  # get directory before going into network
    network_name = name + 'network/'
    network_type = get_network_type(name)
    load_fname_orig = 'Loads_orig.dss'
    load_fname = 'Loads.dss'
    master_fname = 'Master.dss'

    if controller == 'local':
        data = np.load(name + 'LC_data.npz')
        c_network = data['c_network']
        d_network = data['d_network']
        u_network = data['u_network']
        ev_network = data['ev_network']
        cost_network = data['cost_network']
        _, _, demand_imag, _, res_ids, com_ids = load_demand_data_flex(name)
        demand_ev = np.loadtxt(name + 'demand_solar_ev.csv')
        demand = np.loadtxt(name + 'demand_solar.csv')
        net_demand = demand_ev + c_network - d_network + u_network
        pf = 0.92
        net_demand_imag = demand_imag + np.tan(np.arccos(pf)) * u_network
    elif controller == 'central':
        data = np.load(name + 'GC_data.npz')
        c_network = data['c_network']
        d_network = data['d_network']
        u_network = data['u_network']
        ev_network = data['ev_network']
        cost_network = data['cost_network']
        _, _, demand_imag, _, res_ids, com_ids = load_demand_data_flex(name)
        demand_ev = np.loadtxt(name + 'demand_solar_ev.csv')
        demand = np.loadtxt(name + 'demand_solar.csv')
        net_demand = demand_ev + c_network - d_network + u_network
        pf = 0.92
        net_demand_imag = demand_imag

    # get default network loads
    copy_loadfile(network_name + load_fname_orig, network_name + load_fname)
    #demand = get_load_bases(network_name, load_fname, network_type)
    # print('demand shape', demand.shape)

    # Initialize openDSS
    dssObj = dss.DSS
    dssCircuit = dssObj.ActiveCircuit
    # Run PF with initial values
    runPF(network_name, master_fname, dssObj)
    network_name = os.getcwd() + '/'

    # export_taps writes the data files that contain info about transformer power and tap changes
    export_taps(network_name, master_fname, dssObj)
    tap_fname = get_tap_changes_filename(network_name)
    t_fname = get_t_power_filename(network_name)

    # read tap changes
    taps_prev = np.nan
    tap_changes, taps_prev = get_tap_changes(tap_fname, taps_prev=np.nan)

    # read transformer powers only the input power
    t_real, t_reac, t_apparent_power = get_t_power(t_fname)
    t_real, t_reac, t_apparent_power = t_input_powers(t_real, t_reac, t_apparent_power)
    # print('transformers shape', t_real.shape)

    # get voltages
    v_mags = get_VmagsPu(dssCircuit)

    # Print network characteristics
    print('N = ', v_mags.shape)
    print('Nc = ', demand.shape)
    print('N transformers = ', t_apparent_power.shape)

    ### make and test new loads
    T = demand.shape[1]
    t_real_all = np.zeros((t_apparent_power.size, T))
    t_reac_all = np.zeros((t_apparent_power.size, T))
    t_powers_all = np.zeros((t_apparent_power.size, T))
    v_mags_all = np.zeros((v_mags.size, T))
    for t in range(T):
        print('time step:', t)
        loads_real_new = net_demand[:, t]
        loads_imag_new = net_demand_imag[:, t]

        # update openDSS load file
        updateLoads_3ph(network_name, load_fname, loads_real_new, loads_imag_new, network_type)

        runPF(network_name, master_fname, dssObj)
        export_taps(network_name, master_fname, dssObj)
        t_real, t_reac, t_apparent_power = get_t_power(t_fname)
        t_real, t_reac, t_apparent_power = t_input_powers(t_real, t_reac, t_apparent_power)
        v_mags = get_VmagsPu(dssCircuit)
        t_real_all[:, t] = t_real
        t_reac_all[:, t] = t_reac
        t_powers_all[:, t] = t_apparent_power
        v_mags_all[:, t] = v_mags

    # evaluate binary metrics
    if year == 2020 and controller == 'local':
        transformer_limits = assign_transformer_limits(t_powers_all)
        #np.savetxt('/Users/tom/Dropbox/Pycharm3/DERCS_x/rural_san_benito/' + 'transformer_limits.csv', transformer_limits)
        np.savetxt(cwd + '/' + name + 'transformer_limits.csv', transformer_limits)
        v_max, v_min = assign_voltage_limits(v_mags_all)
    else:
        transformer_limits = np.loadtxt(cwd + '/' + name + 'transformer_limits.csv')
        v_max, v_min = assign_voltage_limits(v_mags_all)

    t_vios = get_t_vio_metric(t_powers_all, transformer_limits)
    v_vios = get_v_vio_metric(v_mags_all, v_max, v_min)

    print(np.sum(t_vios > 1) / t_vios.size)
    print(np.sum(v_vios > 1) / v_vios.size)

    np.savez(cwd + '/' + name + controller + '_metrics.npz', t_real_all=t_real_all, t_reac_all=t_reac_all,
             t_powers_all=t_powers_all, v_mags_all=v_mags_all,
             cost_network=cost_network, t_vios=t_vios, v_vios=v_vios)
    print('SAVED METRICS')

