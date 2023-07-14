import dask.dataframe as dd
import pandas as pd
import numpy as np
import os

import matplotlib.pyplot as plt

#s3://oedi-data-lake/nrel-pds-building-stock/end-use-load-profiles-for-us-building-stock/2021/comstock_amy2018_release_1/timeseries_individual_buildings/by_state/upgrade=0/state=CA/*.parquet'
#s3://oedi-data-lake/nrel-pds-building-stock/end-use-load-profiles-for-us-building-stock/2021/resstock_amy2018_release_1/timeseries_individual_buildings/by_state/upgrade=0/state=CA/*.parquet'
#df = dd.read_parquet('s3://oedi-data-lake/nrel-pds-building-stock/end-use-load-profiles-for-us-building-stock/2021/comstock_amy2018_release_1/timeseries_individual_buildings/by_state/upgrade=0/state=CA/100422-0.parquet',
#                     storage_options={'anon': True})

# column: out.electricity.total.energy_consumption
# units kWh
# Multiply by 4 to convert 15-min resolution kWh into kW

def dl_NREL_Load_data(name, category='res', device='total', directory=None):
    res_header = 's3://oedi-data-lake/nrel-pds-building-stock/end-use-load-profiles-for-us-building-stock/2021/resstock_amy2018_release_1/timeseries_individual_buildings/'
    com_header = 's3://oedi-data-lake/nrel-pds-building-stock/end-use-load-profiles-for-us-building-stock/2021/comstock_amy2018_release_1/timeseries_individual_buildings/'

    known_names = {
    'vermont': "by_puma_northeast/upgrade=0/puma=G50000100/",  # northwest vermont (Burlington city)
    'arizona': 'by_puma_west/upgrade=0/puma=G04000121/',  # southwest phoenix
    'central_SF': 'by_puma_west/upgrade=0/puma=G06007503/',  # south of market and potrero
    'tracy': 'by_puma_west/upgrade=0/puma=G06007703/',  #
    'commercial_SF': 'by_puma_west/upgrade=0/puma=G06007502/',  # commercial area in north shore
    'oakland': 'by_puma_west/upgrade=0/puma=G06000104/', # south central
    'los_banos': 'by_puma_west/upgrade=0/puma=G06004701/',
    'marin': 'by_puma_west/upgrade=0/puma=G06004102/', # sausalito
    'rural_san_benito': 'by_puma_west/upgrade=0/puma=G06001901/',
    'sacramento' : 'by_puma_west/upgrade=0/puma=G06006708/' # south east sacramento
    }

    # peak_name = 'out.electricity.peak_demand.energy_consumption'
    # 'out.natural_gas.total.energy_consumption'
    # 'out.propane.total.energy_consumption'
    if device == 'total':
        col_name = 'out.electricity.total.energy_consumption'
    elif device == 'heat':
        col_name = 'out.natural_gas.heating.energy_consumption'
    elif device == 'wheat':
        col_name = 'out.natural_gas.water_systems.energy_consumption'
    elif device == 'cool':
        col_name = 'out.electricity.cooling.energy_consumption'
    else:
        print('device name not recognized', device)
        return False

    if category == 'res':
        header = res_header
    elif category == 'com':
        header = com_header
    else:
        print('category is either res or com for residential or commercial loads')
        return False

    if name in known_names.keys():
        name_path = known_names[name]
    else:
        print('using input name as file path')
        name_path = name

    if directory == None:
        directory = './' + name + '/'

    size_year = 35040 # number of 15-minute intervals in a year

    # Quit if file already exists otherwise, download
    if os.path.exists(directory + category + '_' + device + '_load_data.csv'):
        print('file already exists')
        return False

    df = dd.read_parquet(header + name_path, storage_options={"anon": True}, columns=[col_name])
    print('read parquet')
    df = df.values.compute()
    print('computed values')
    print(df.shape)
    df = df.reshape((int(df.size/size_year), size_year))
    power = df * 4 # multiply by 4 to convert kWh per 15 minute period to kW

    # shape of load file is (consumers, time)
    #print(power.shape)
    #print(np.max(power, axis=1))

    make_directories(directory)
    np.savetxt(directory + category + '_' + device + '_load_data.csv', power, delimiter=',')
    print('downloaded data to:', directory + category + '_' + device + '_load_data.csv')
    return True


class Resampling:  # resamples data to proper time resolution
    def __init__(self, data, t, tnew):
        self.data = data
        self.t = t
        self.tnew = tnew

    def sample(self, data, t, tnew):
        if t > tnew:
            new_num_col = int(data.shape[1] * t / tnew)
            ret = self.upsampling(data, t, tnew, new_num_col)
        else:
            ret = self.downsampling(data, t, tnew)
        return ret

    def downsampling(self, data, t, tnew):
        t_ratio = int(tnew / t)
        downsampled = np.zeros((np.size(data, 0), int(np.size(data, 1) / t_ratio)))
        for i in range(np.size(downsampled, 0)):
            for j in range(np.size(downsampled, 1)):
                downsampled[i][j] = np.average(data[i][j * t_ratio:(j + 1) * t_ratio])
        return downsampled

    def upsampling(self, data, t, tnew, new_num_col):
        steps_per_day = int(1440 / t)
        # num_days = int((np.size(data) * t) / 1440)
        one_day = np.zeros((np.size(data, 0), steps_per_day))
        one_day_std = np.zeros((np.size(data, 0), steps_per_day))

        for a in range(np.size(one_day, 0)):
            for b in range(np.size(one_day, 1)):
                sub_array = data[a, b::steps_per_day]
                one_day[a, b] = np.mean(sub_array)
                one_day_std[a, b] = np.std(sub_array)

        t_ratio = int(t / tnew)
        upsampled = np.zeros((np.size(data, 0), new_num_col))
        for i in range(np.size(upsampled, 0)):
            for j in range(np.size(upsampled, 1)):
                # upsampled[i][j] = np.random.normal(one_day[i,(j/t_ratio)%24], one_day_std[i,(j/t_ratio)%24])
                upsampled[i][j] = np.random.normal(data[i, int(j / t_ratio)], one_day_std[i, int(j / t_ratio) % 24])
        return upsampled


def solar_data_format(fname):
    # extract power data
    # downsample from 5min resolution to 15min
    # load file
    solar = pd.read_csv(fname, delimiter=',')
    print(solar.shape)
    solar = solar.values[:,1]

    # resample data to 15 min resolution
    resolution_current = 5  # units is minutes
    resolution_new = 15
    solar = solar.reshape((1, solar.size))  # make array 2D
    sampler = Resampling(solar, resolution_current, resolution_new)
    solar = sampler.sample(solar, resolution_current, resolution_new)
    solar = solar[0, :]  # make 1D again

    # convert units to kW from MW
    solar = solar * 1000

    return solar


def load_data(name, category, device):
    directory = './' + name + '/'
    power = np.loadtxt(directory + category + '_' + device + '_load_data.csv', delimiter=',')
    print(power.shape)
    print(np.max(power))
    print(power[4, 4*24:8*24])
    print(power[4 * 24:8 * 24, 4])


def make_directories(path):

    if not os.path.exists(path):
        os.makedirs(path)
        print('Created directory:', path)

    return True


def check_directories(path):
    # return True if path exists False otherwise
    if os.path.exists(path):
        return True
    else:
        return False


def get_winter_months(data):
    t_res = 4  # data points per hour
    data_n = np.hstack((data[:, -31*24*t_res:], data[:, 0:59*24*t_res]))
    print(data_n.shape)
    return data_n


def get_summer_months(data):
    t_res = 4  # data points per hour
    data_n = data[:, 151*24*t_res:243*24*t_res]
    print(data_n.shape)
    return data_n


def sort_results(fname='2040_results.csv'):
    df = pd.read_csv(fname, names=['name','control','t_vio','v_vio'])
    print(df[df['control'] == 'uncoordinated'])
    print(df[df['control'] == 'coordinated price'])
    print(df[df['control'] == 'coordinated grid'])


def sort_results_other():
    df = pd.read_csv('other_results.csv', names=['name','control','t_vio','v_vio'])
    print(df[df['control'] == '12kW EV charging'])
    #control_name = '100% storage co-adoption'
    #control_name = '0% EV pricing'
    #control_name = '12kW EV charging'


def plot_years(name):
    years = [2020, 2025, 2030, 2035, 2040, 2045, 2050]

    df = pd.read_csv('years_results.csv', names=['name', 'year', 'control', 't_vio', 'v_vio'])
    df = df[df['name'] == name]

    plt.figure()
    plt.plot(years, np.clip(df[df['control'] == 'uncoordinated']['t_vio'] - 0.073, 0, None))
    plt.plot(years, np.clip(df[df['control'] == 'coordinated price']['t_vio'] - 0.073, 0, None))
    plt.plot(years, np.clip(df[df['control'] == 'coordinated grid']['t_vio'] - 0.073, 0, None))
    plt.legend(('uncoordinated', 'coordinated price', 'coordinated grid'))
    plt.xlabel('years')
    plt.ylabel('% overloaded transformers')
    plt.grid(axis='y')
    plt.ylim(0, 1)
    plt.savefig('figs/'+name+'_years_t')

    plt.figure()
    plt.plot(years, np.clip(df[df['control'] == 'uncoordinated']['v_vio']/1.5 - 0.073, 0, .9))
    plt.plot(years, np.clip(df[df['control'] == 'coordinated price']['v_vio']/1.5 - 0.073, 0, .9))
    plt.plot(years, np.clip(df[df['control'] == 'coordinated grid']['v_vio']/1.5 - 0.073, 0, .9))
    plt.legend(('uncoordinated', 'coordinated price', 'coordinated grid'))
    plt.xlabel('years')
    plt.ylabel('% nodes with voltage violation')
    plt.grid(axis='y')
    plt.ylim(0, 1)
    plt.savefig('figs/'+name + '_years_v')
    #plt.show()


def print_years(name):
    years = [2020, 2025, 2030, 2035, 2040, 2045, 2050]

    df = pd.read_csv('years_results.csv', names=['name', 'year', 'control', 't_vio', 'v_vio'])
    df = df[df['name'] == name]

    plt.figure()
    print(np.clip(df[df['control'] == 'uncoordinated']['t_vio'] - 0.073, 0, None))
    print(np.clip(df[df['control'] == 'coordinated price']['t_vio'] - 0.073, 0, None))
    print(np.clip(df[df['control'] == 'coordinated grid']['t_vio'] - 0.073, 0, None))

    print('voltage')
    print(np.clip(df[df['control'] == 'uncoordinated']['v_vio']/1.5 - 0.073, 0, .9))
    print(np.clip(df[df['control'] == 'coordinated price']['v_vio']/1.5 - 0.073, 0, .9))
    print(np.clip(df[df['control'] == 'coordinated grid']['v_vio']/1.5 - 0.073, 0, .9))


def download_all_data(name):
    dl_NREL_Load_data(name, category='res', device='total')
    dl_NREL_Load_data(name, category='res', device='heat')
    dl_NREL_Load_data(name, category='res', device='wheat')
    dl_NREL_Load_data(name, category='res', device='cool')
    dl_NREL_Load_data(name, category='com', device='total')
    dl_NREL_Load_data(name, category='com', device='heat')
    dl_NREL_Load_data(name, category='com', device='wheat')
    dl_NREL_Load_data(name, category='com', device='cool')
    return


if __name__ == '__main__':
    name = 'tracy'

    # download load data for selected network
    download_all_data(name)

    ###
    # reformat solar profile file
    fname = name + '/Actual_37.65_-121.75_2006_DPV_9MW_5_Min.csv'
    solar_15min = solar_data_format(fname)
    #print(solar_15min[92*4*24:93*4*24])
    np.savetxt(name + '/solar_15min.csv', solar_15min)

    ###
    #sort_results_2040(fname='commercial_results.csv')

    #name = 'los_banos'
    names = ['tracy', 'vermont', 'arizona', 'central_SF', 'commercial_SF',
             'oakland', 'los_banos', 'marin', 'rural_san_benito', 'sacramento']

    #print_years(name)

    categories = ['res', 'com']
    devices = ['total', 'heat', 'wheat', 'cool']
    #control = ['un', 'ceb', 'cg']
    #control = ['un', 'sca', 'evp', 'evc']
    years = [2020, 2024, 2030, 2035, 2040, 2045, 2050]


