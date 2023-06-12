import dss
import numpy as np
import pandas as pd
import argparse

# To standardize the network files make sure all networks use the file structure:
# name\
#   data files
#   network\
#       'Loads_orig.dss'
#       'Loads.dss'
#       'Master.dss'
#       other network files
# Rename the original files to the standard names
# Make a copy of the original load file to Loads_orig.dss
# Some original network files have unrealistic voltage regulation and tap changing control so comment out as necessary

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
    DER = int(FLAGS.DER)

    if FLAGS.network == 'iowa':
        dir_network = 'Iowa_feeder/'
        load_fname_orig = 'Loads_orig.dss'
        load_fname = 'Load.dss'
        master_fname = 'Master.dss'
        t_res = 24 * 4
    elif FLAGS.network == '123':
        dir_network = 'IEEE123-1ph-test/'
        load_fname_orig = 'Load_orig.DSS'
        load_fname = 'Load.dss'
        master_fname = 'Master.dss'
        t_res = 24 * 4
    elif FLAGS.network == '123-3':
        dir_network = 'IEEE123/'
        load_fname_orig = 'Load_orig.DSS'
        load_fname = 'Load.DSS'
        master_fname = 'Run_IEEE123Bus.DSS'
        t_res = 24 * 4
    elif FLAGS.network == '34':
        dir_network = '34Bus/'
        load_fname_orig = 'ieee34Mod2.dss'
        load_fname = 'Load.DSS'
        master_fname = 'Run_IEEE34Mod2.dss'
        t_res = 24 * 4
    elif FLAGS.network == '34-2':
        dir_network = '34network/'
        load_fname_orig = 'Loads.dss'
        load_fname = 'Load.DSS'
        master_fname = 'Master.dss'
        t_res = 24 * 4
    else:
        print('reverting to default network format')
        dir_network = FLAGS.network + '/'
        load_fname_orig = 'Loads_orig.dss'
        load_fname = 'Loads.dss'
        master_fname = 'Master.dss'
        t_res = 24 * 4

    return storagePen, solarPen, evPen, dir_network, load_fname_orig, load_fname, master_fname, n_days, DER, t_res


def copyLoadFile(file_orig, file_copy):
    from shutil import copyfile
    copyfile(file_orig, file_copy)
    return True


def updateLoads_3ph(path_network, load_fname, loads_real_new, loads_imag_new):
    # updates the kW and kvar value of existing loads
    loadfile = pd.read_csv(path_network + load_fname, sep=' ', header=None, usecols=range(9), engine='python')

    loads_real_str = ['kW=' + item for item in np.array(loads_real_new, dtype=str)]
    loads_imag_str = ['kvar=' + item for item in np.array(loads_imag_new, dtype=str)]

    name_str = ['"' + item + '"' for item in np.array(loadfile[1], dtype=str)]
    name_str = np.array(name_str, dtype=str)

    loadfile[1] = name_str  # 1 for Iowa and IEEE
    loadfile[6] = loads_real_str  # 6 for Iowa, 7 for IEEE
    loadfile[8] = loads_imag_str  # 8 for Iowa and IEEE

    loadfile.to_csv(path_network + load_fname, sep=' ', header=None, index=False,
                    quoting=3, quotechar="", escapechar="\\")

    return True


def runPF(path_network, master_fname, dssObj):
    # Run the PF to initialize values
    dssObj.Text.Command = "compile " + path_network + master_fname
    dssObj.ActiveCircuit.Solution.Solve()

    return True


def getVmagsPu(dssCircuit):
    # Units are in Volts
    Vmag = dssCircuit.AllBusVmagPu

    return Vmag


def export_taps(path_network, master_fname, dssObj):
    dssObj.Text.Command = "Export Taps"
    dssObj.Text.Command = "Export powers"

    return True


def get_t_power(fname):
    loadfile = pd.read_csv(fname, sep=',')
    ts = loadfile[loadfile['Element'].str.contains("Transformer")]
    t_apparent_power = np.sqrt(ts[' P(kW)'] ** 2 + ts[' Q(kvar)'] ** 2)

    # make sign of apparent power equal to sign of real power component to show direction of real power flow
    t_apparent_power = t_apparent_power * np.sign(ts[' P(kW)'])

    return ts[' P(kW)'], ts[' Q(kvar)'], t_apparent_power


def get_tap_changes(fname, taps_prev=np.nan):
    loadfile = pd.read_csv(fname, sep=',')
    taps = loadfile[' Position']
    if np.any(np.isnan(taps_prev)):
        tap_change = np.zeros(taps.shape)
    else:
        tap_change = np.abs(taps - taps_prev)
    taps_prev = taps

    return tap_change, taps_prev


def get_t_limits(t_mag, fixed=1):
    if fixed:
        # typical transformer sizes from palo alto utility (units in kVA):
        typical_transformers = [5, 7.5, 10, 15, 25, 30, 37.5, 45, 50, 75, 100, 112.5, 150, 167, 225, 300, 500, 750, 1000, 1500,
                        2000, 2500]
        transformers_copy = [5, 7.5, 10, 15, 25, 30, 37.5, 45, 50, 75, 100, 112.5, 150, 167, 225, 300, 500, 750, 1000, 1500,
                             2000, 2500]

        max_leafs = np.max(np.abs(t_mag), axis=1)
        t_limits = []
        for c in max_leafs:
            if c < 1e-3:
                print('transformer too small')
                t_limits.append(5)
                continue
            transformers = [5, 7.5, 10, 15, 25, 30, 37.5, 45, 50, 75, 100, 112.5, 150, 167, 225, 300, 500, 750, 1000, 1500,
                        2000, 2500, 3000]
            transformers.append(c)
            transformers.sort()
            idx = np.where(transformers == c)[0]
            # rating.append(transformers[(int(idx) + 1) if (idx.size > 0) else 1])
            try:
                t_limits.append(transformers[(int(idx) + 2) if (idx.size > 0) else 1])
            except:
                t_limits.append(c * 1.2)
                print('t capacity + 0.20', c)
            # print('Current transformers vector: ', transformers)
        t_limits = np.array(t_limits)

    else:
        # transformer limits are 20% greater than base case
        t_limits = (np.abs(t_mag) + 1) * 1.2

    return t_limits


def get_v_limits(v_mags, fixed=1):
    if fixed:
        v_max = np.ones(v_mags.shape) * 1.05
        v_min = np.ones(v_mags.shape) * 0.95
    else:
        # voltage limits are 20% greater than base case
        v_max = np.ones(v_mags.shape) * ((np.max(v_mags) - 1) * 1.2 + 1)
        v_min = np.ones(v_mags.shape) * ((np.min(v_mags) - 1) * 1.2 + 1)

    return v_max, v_min


def makeSinglePhase(dssObj, path_networks, dir_name):
    posSeq_command = 'MakePosSeq'
    save_command = 'Save Circuit dir=' + path_networks + dir_name
    print('starting pos sequence model calculation')
    dssObj.Text.Command = posSeq_command
    dssObj.Text.Command = save_command
    print('finished saving')
    return True


def main_convertSinglePhase(path_networks, dir_network, master_fname, new_dir):
    # defining DSS objects
    dssObj = dss.DSS

    # Run PF with initial values to initialize
    runPF(path_networks + dir_network, master_fname, dssObj)

    makeSinglePhase(dssObj, path_networks, new_dir)
    return True


def train_test_split(data, t_idx):
    data_train = data[:, 0:t_idx]
    data_test = data[:, t_idx:]

    return data_train, data_test


if __name__ == '__main__':
    FLAGS = np.loadtxt('network_chars.csv')
    parse_inputs(FLAGS)


