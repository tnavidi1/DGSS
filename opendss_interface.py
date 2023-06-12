import os

import dss
import numpy as np
import pandas as pd

#pip install dss-python, numpy, pandas


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


def copy_loadfile(file_orig, file_copy):
    from shutil import copyfile
    copyfile(file_orig, file_copy)
    return True


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


def runPF(path_network, master_fname, dssObj):
    # Run the PF to initialize values
    dssObj.Text.Command = "compile " + path_network + master_fname
    dssObj.Text.Command = 'calcv'
    dssObj.ActiveCircuit.Solution.Solve()

    return True


def export_taps(path_network, master_fname, dssObj):
    dssObj.Text.Command = "Export Taps"
    dssObj.Text.Command = "Export powers"
    #dssObj.Text.Command = "Export voltages"

    return True


def get_tap_changes_filename(name):
    import fnmatch
    path = name
    for filename in os.listdir(path):
        if fnmatch.fnmatch(filename, '*EXP_Taps.csv'):
            #print('found file with name', filename)
            return name + filename
        elif fnmatch.fnmatch(filename, '*EXP_Taps.CSV'):
            # print('found file with name', filename)
            return name + filename
    print('failed to find file')
    return False


def get_tap_changes(fname, taps_prev=np.nan):
    loadfile = pd.read_csv(fname, sep=',')
    taps = loadfile[' Position']
    if np.any(np.isnan(taps_prev)):
        tap_change = np.zeros(taps.shape)
    else:
        tap_change = np.abs(taps - taps_prev)
    taps_prev = taps

    return tap_change, taps_prev


def get_t_power(fname):
    loadfile = pd.read_csv(fname, sep=',')
    ts = loadfile[loadfile['Element'].str.contains("Transformer")]
    t_apparent_power = np.sqrt(ts[' P(kW)'] ** 2 + ts[' Q(kvar)'] ** 2)

    # make sign of apparent power equal to sign of real power component to show direction of real power flow
    t_apparent_power = t_apparent_power * np.sign(ts[' P(kW)'])

    return ts[' P(kW)'], ts[' Q(kvar)'], t_apparent_power


def t_input_powers(t_real, t_reac, t_apparent_power):
    return t_real[::2], t_reac[::2], t_apparent_power[::2]


def get_t_power_filename(name):
    import fnmatch
    path = name
    for filename in os.listdir(path):
        if fnmatch.fnmatch(filename, '*EXP_POWERS.csv'):
            #print('found file with name', filename)
            return name + filename
        elif fnmatch.fnmatch(filename, '*EXP_POWERS.CSV'):
            # print('found file with name', filename)
            return name + filename
    print('failed to find file')
    return False


def get_VmagsPu(dssCircuit):
    # Units are PU not Volts
    Vmag = dssCircuit.AllBusVmagPu

    return Vmag


def updateLoads_3ph(path_network, load_fname, loads_real_new, loads_imag_new, network_type):
    if network_type == 'SFO':
        idx_r = 8
        idx_i = 9
        sep = ' '
    elif network_type == 'IEEE':
        idx_r = 7
        idx_i = 8
        sep = ' '
    elif network_type == 'iowa':
        idx_r = 6
        idx_i = 8
        sep = ' '
    elif network_type == 'vermont':
        idx_r = 5
        idx_i = 6
        sep = '\t'
    else:
        print('network type not recognized')
        idx_r = 8
        idx_i = 9
        sep = ' '

    # updates the kW and kvar value of existing loads
    loadfile = pd.read_csv(path_network + load_fname, sep=sep, header=None, usecols=range(9), engine='python')

    loads_real_str = ['kW=' + item for item in np.array(loads_real_new, dtype=str)]
    loads_imag_str = ['kvar=' + item for item in np.array(loads_imag_new, dtype=str)]

    name_str = ['"' + item + '"' for item in np.array(loadfile[1], dtype=str)]
    name_str = np.array(name_str, dtype=str)

    loadfile[1] = name_str  # 1 for Iowa and IEEE and SFO
    loadfile[idx_r] = loads_real_str  # 6 for Iowa, 7 for IEEE, 8 for SFO, 5 for Vermont
    loadfile[idx_i] = loads_imag_str  # 8 for Iowa and IEEE, 9 for SFO, 6 for Vermont

    loadfile.to_csv(path_network + load_fname, sep=sep, header=None, index=False,
                    quoting=3, quotechar="", escapechar="\\")

    return True


if __name__ == '__main__':
    name = 'sacramento/'
    #name = 'iowa/'
    #name = 'central_SF/'
    #name = 'commercial_SF/'
    #name = 'tracy/'
    #name = 'rural_san_benito/'
    #name = 'los_banos/'
    #name = 'vermont/'
    #name = 'arizona/'
    #name = 'marin/'
    #name = 'oakland/'
    network_name  = name + 'network/'
    network_type = get_network_type(name)
    load_fname_orig = 'Loads_orig.dss'
    load_fname = 'Loads.dss'
    master_fname = 'Master.dss'
    #print(network_name)

    # get default network loads
    copy_loadfile(network_name + load_fname_orig, network_name + load_fname)
    demand = get_load_bases(network_name, load_fname, network_type)
    #print('demand shape', demand.shape)

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
    #print('transformers shape', t_real.shape)

    # get voltages
    v_mags = get_VmagsPu(dssCircuit)

    # Print network characteristics
    print('N = ', v_mags.shape)
    print('Nc = ', demand.shape)
    print('N transformers = ', t_apparent_power.shape)

    ### make and test new loads
    # print(t_apparent_power)
    # print(v_mags)
    loads_real_new = demand * 1.9
    loads_imag_new = demand * 0.3

    # update openDSS load file
    updateLoads_3ph(network_name, load_fname, loads_real_new, loads_imag_new, network_type)

    ### test that changes were successful
    runPF(network_name, master_fname, dssObj)
    export_taps(network_name, master_fname, dssObj)
    t_real, t_reac, t_apparent_power = get_t_power(t_fname)
    t_real, t_reac, t_apparent_power = t_input_powers(t_real, t_reac, t_apparent_power)
    v_mags = get_VmagsPu(dssCircuit)
    #print(t_apparent_power)
    #print(v_mags)



