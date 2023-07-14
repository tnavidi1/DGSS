import numpy as np
from sklearn.mixture import GaussianMixture as GMM
import argparse


def get_total_EVs(demand, evPen):
    total_EVs = np.sum(np.mean(demand, axis=1)) / 1  # 1 EV per kW of average demand is 1 EV per home on average
    # total number of EVs with our given penetration
    n_EV = int(total_EVs * evPen)

    return total_EVs, n_EV


def assign_cars_to_nodes(spot_loads, total_EVs=1200, garage_threshold=17):
    # garage and home ids are ids for load_nodes array
    # In load_nodes, we ignore the intermediate relay nodes, where current only flows through and isn't drawn.
    load_nodes = np.where(spot_loads > 0)[0]

    alphas = spot_loads[load_nodes]/np.sum(spot_loads)
    EVs_per_node = total_EVs * alphas
    EVs_per_node = np.array(EVs_per_node, dtype=int)
    while np.sum(EVs_per_node) < total_EVs:
        id = np.random.choice(np.arange(EVs_per_node.size), 1)
        EVs_per_node[id] += 1

    # print('number of load nodes', len(EVs_per_node))
    # print('EVs per node', EVs_per_node)
    # print(np.sum(EVs_per_node))

    # If there are more than x EVs associated to a node, we assume that the node is a garage.
    garage_ids = np.where(EVs_per_node >= garage_threshold)[0]
    # print('garage node ids', garage_ids)

    # If there are less than x EVs associated to a node, we assume that the node is a home.
    home_ids = np.where(EVs_per_node <= garage_threshold)[0]
    # print('home node ids', home_ids)

    #print('cars in work chargers', np.sum(EVs_per_node[garage_ids]))
    #print('cars in home chargers', np.sum(EVs_per_node[home_ids]))

    return EVs_per_node, garage_ids, home_ids


def initialize_dictionaries(nodes=85, days=31):
    # IDs are for load_nodes array not full 123 bus
    # final desired dictionaries contain lists of numpy arrays
    # each value of dictionary is for a node
    # each element of list is the day

    start_dict = {}
    end_dict = {}
    charge_dict = {}

    for i in range(nodes):
        start_dict[i] = []
        end_dict[i] = []
        charge_dict[i] = []

    return start_dict, end_dict, charge_dict


def remove_invalid(np_array):
    np_array = np_array[np_array < 1000]
    return np_array


def assign_data_to_nodes(garage_ids, home_ids, EVs_per_node, start_times, end_times, charge, start_dict, end_dict,
                         charge_dict, fast_charge_buffer=3.1):
    # split EV samples to garages and homes
    # We assume that cars that start being charged before 10 am are in garages.

    h_ids = list(np.where(start_times > 10)[0])  # which cars are charging in homes
    g_ids = list(np.where(start_times < 10)[0])  # which cars are charging in garages


    if len(g_ids) > np.sum(EVs_per_node[garage_ids]):
        # if there are more morning charging cars than there is space, move the extras to homes
        g_ids = g_ids[0:np.sum(EVs_per_node[garage_ids])]
        h_ids.extend(g_ids[np.sum(EVs_per_node[garage_ids]):])

    # print(h_ids)
    # print(g_ids)

    start_times_g = start_times[g_ids]
    end_times_g = end_times[g_ids]
    charge_g = fast_charge_buffer * charge[g_ids]
    # print('garage charge', charge_g)

    start_times_h = start_times[h_ids]
    end_times_h = end_times[h_ids]
    charge_h = charge[h_ids]
    # print('home charge', charge_h)

    total_energy = 0

    EV_count = 0
    for id in garage_ids:
        start_dict[id].append( remove_invalid(start_times_g[EV_count:EV_count + EVs_per_node[id]]) )
        end_dict[id].append( remove_invalid(end_times_g[EV_count:EV_count + EVs_per_node[id]]) )
        charge_dict[id].append( remove_invalid(charge_g[EV_count:EV_count + EVs_per_node[id]]) )
        EV_count += len(remove_invalid(charge_g[EV_count:EV_count + EVs_per_node[id]]))
        total_energy += np.sum(remove_invalid(charge_g[EV_count:EV_count + EVs_per_node[id]]))
    #print('evs in garages', EV_count)
    # print('garage charge', charge_dict[id])

    EV_count = 0
    for id in home_ids:
        start_dict[id].append( remove_invalid(start_times_h[EV_count:EV_count + EVs_per_node[id]]) )
        end_dict[id].append( remove_invalid(end_times_h[EV_count:EV_count + EVs_per_node[id]]) )
        charge_dict[id].append( remove_invalid(charge_h[EV_count:EV_count + EVs_per_node[id]]) )
        EV_count += len(remove_invalid(charge_h[EV_count:EV_count + EVs_per_node[id]]))
        total_energy += np.sum(remove_invalid(charge_h[EV_count:EV_count + EVs_per_node[id]]))
    #print('evs in homes', EV_count)
    # print('home charge', charge_dict[id])
    #print('total daily energy', total_energy)

    return start_dict, end_dict, charge_dict


def get_number_of_events(GMM_cars, n_EV, p=.3):
    # calculate the number of charge events when 2 events occur with probability p=.3
    # probability of 2 charges in one day = 30%

    percent_charge = GMM_cars.sample(1)[0].flatten()
    percent_charge = percent_charge.clip(min=10)
    n_charge = n_EV * percent_charge / 100

    #print('number of cars charging today', n_charge)

    cars = np.random.choice(np.arange(n_EV), size=int(n_charge))

    # cars = np.random.choice(np.arange(n_EV))
    # cars = np.random.choice(n_charge, np.arange(n_EV))

    # calculate how many charging events for each car per day
    # n_events = np.random.binomial(1, p, n_c) + 1

    return cars

def get_start_times(GMM_start, n_EV, cars):
    start_times = np.zeros(n_EV) + 10000  # initialize values to large so automatically excluded

    if cars.size > 0:
        # generate samples from GMM
        start_times[cars] = GMM_start.sample(cars.size)[0].flatten()
        #print(type(start_times))

        start_times = np.abs(start_times) # remove occasional negatives in a way that doesnt stack 0

    if np.any(start_times < 0):
        print('negative start time', start_times[start_times < 0])

    return start_times


def get_initial_soc(GMM_init_soc, n_EV, cars):
    # same thing as get_start_times, but with initial soc
    initialSoC = np.zeros(n_EV) + 10000

    if cars.size > 0:
        # generate samples from GMM
        initialSoC[cars] = GMM_init_soc.sample(cars.size)[0].flatten()

    return initialSoC


def get_final_soc(GMM_final_soc, n_EV, cars):
    # same thing as get_start_times, but with final soc
    finalSoC = np.zeros(n_EV) + 10000

    if cars.size > 0:
        # generate samples from GMM
        finalSoC[cars] = GMM_final_soc.sample(cars.size)[0].flatten()

    return finalSoC

def get_charge_duration(inital_soc, final_soc, n_EV, cars, t_res, power=8, buffer=1.5, future_charge_buffer=2.2, fast_charge_buffer=3.1):
    # calculate charge duration for each car
    charge = np.zeros(n_EV) + 10000

    charge[cars] = future_charge_buffer * np.abs(final_soc[cars] - inital_soc[cars])
    # print(charge)

    duration = np.array(np.ceil(fast_charge_buffer * charge / power * buffer), dtype=int)
    # duration = np.array(np.ceil(charge / power / t_res * buffer), dtype=int)

    if np.any(duration < 1):
        print('duration is too short', duration)

    return charge, duration

def get_end_times(start_times, duration, n_EV, cars):
    # calculate the end time based on how much needs to be charged
    end_times = np.zeros(n_EV) + 10000

    end_times[cars] = start_times[cars] + duration[cars]

    return end_times

def generate_EV_data(days, n_EV, start_dict, end_dict, charge_dict, garage_ids, home_ids, EVs_per_node,
                     charging_power):
    # calculate number of EVs

    # define GMM parameters
    t_res = 15/60.  # time resolution is 15 minutes for GMMs, but we want a resolution of 1h

    # Start time stats (Table I)
    n_components = 5
    weights_start_1 = np.array([.3509, .1061, .0364, .0387, .4678])
    means_start_1 = np.array([42.91, 32.1896, 2.2375, 94.0939, 73.6893]).reshape((n_components, 1))
    covs_start_1 = np.array([283.9475, 11.9877, 1.5401, 2.0585, 76.3613])

    GMM_start = GMM(n_components=n_components, covariance_type='spherical')
    GMM_start.weights_ = weights_start_1
    GMM_start.means_ = means_start_1
    GMM_start.covariances_ = covs_start_1
    GMM_start.precisions_cholesky_ = None


    # Percentage of cars charging per day (Table VII)
    weights_cars = np.array([.0605, .0530, .2161, .4049, .2655])
    means_cars = np.array([1.1760, 100.8095, 58.7235, 79.9056, 64.0442]).reshape((n_components, 1))
    covs_cars = np.array([2.1031, 5.3675, 236.0372, 92.7873, 87.7866])

    GMM_cars = GMM(n_components=n_components, covariance_type='spherical')
    GMM_cars.weights_ = weights_cars
    GMM_cars.means_ = means_cars
    GMM_cars.covariances_ = covs_cars
    GMM_cars.precisions_cholesky_ = None


    # Initial SoC per charging event (considering only weekday, whole days) (Table IV)
    weights_initialSoC = np.array([.1983, .3103, .2174, .0967, .1773])
    means_initialSoC = np.array([5.7377, 7.6143, 2.4397, 10.8346, 4.3333]).reshape((n_components, 1))
    covs_initialSoC = np.array([2.7184, 2.2484, 1.1016, .8860, 1.4895])

    GMM_initialSoC = GMM(n_components=n_components, covariance_type='spherical')
    GMM_initialSoC.weights_ = weights_initialSoC
    GMM_initialSoC.means_ = means_initialSoC
    GMM_initialSoC.covariances_ = covs_initialSoC
    GMM_initialSoC.precisions_cholesky_ = None


    # Final SoC per charging event (considering only weekday, whole days) (Table V)
    weights_finallSoC = np.array([.4290, .1063, .2085, .1515, .1047])
    means_finallSoC = np.array([11.9730, 6.5755, 9.7931, 12.3320, 11.6428]).reshape((n_components, 1))
    covs_finallSoC = np.array([.0362, 4.9691, 1.2780, .0057, .0034])

    GMM_finallSoC = GMM(n_components=n_components, covariance_type='spherical')
    GMM_finallSoC.weights_ = weights_finallSoC
    GMM_finallSoC.means_ = means_finallSoC
    GMM_finallSoC.covariances_ = covs_finallSoC
    GMM_finallSoC.precisions_cholesky_ = None

    for i in range(days):
        # sample all cars and distribute first
        # cars = np.arange(n_EV)
        cars = get_number_of_events(GMM_cars, n_EV)

        # Get initial and final states of charge
        initial_soc = get_initial_soc(GMM_initialSoC, n_EV, cars)
        final_soc = get_final_soc(GMM_finallSoC, n_EV, cars)

        # Get charge and duration of charge
        charge, duration = get_charge_duration(initial_soc, final_soc, n_EV, cars, t_res,
                                               power=charging_power, buffer=2.0, future_charge_buffer=1, fast_charge_buffer=1)

        # Get start and end times
        start_times = get_start_times(GMM_start, n_EV, cars=cars)
        start_times = np.array(start_times * t_res, dtype=int)
        end_times = np.array(get_end_times(start_times, duration, n_EV, cars), dtype=int)
    
        # Divide the times in the start and end times arrays by 4 and round to the nearest whole number.
        # start_times = np.array(start_times * t_res, dtype=int)
        # end_times = np.array(end_times * t_res, dtype=int)

        #print('sum of charge', remove_invalid(charge))
        #print('sum of charge', charge)
        #print(i)
        
        # add the daily samples to dictionaries
        start_dict, end_dict, charge_dict = assign_data_to_nodes(garage_ids, home_ids, EVs_per_node, start_times,
                                                                 end_times, charge, start_dict, end_dict, charge_dict,
                                                                 fast_charge_buffer=1)
    # end for loop

    return start_dict, end_dict, charge_dict


def get_EV_charging_window(t_arrival, t_depart, T, charge_power):
    # t_arrival is a 1D array of arrival times of many EVs
    # same for t_depart, but it is the departure times

    # initialize array to have size T
    total_charge = np.zeros(T)

    # this function needs to add up one array for each element in the input arrays
    for i in range(len(t_arrival)):

        # create array with number of elements = t_depart - t_arrival

        if t_depart[i] < t_arrival[i]:
            print(t_depart[i], t_arrival[i])
        charge = np.zeros(t_depart[i] - t_arrival[i])

        charge[:] = charge_power
        # Add new array to total_charge starting at index t_arrival and ending at index_t_depart
        #print(total_charge)
        total_charge[t_arrival[i] : t_depart[i]] += charge

    # end for loop

    return total_charge


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate solar and storage demand data')
    # parser.add_argument('--storagePen', default=4, help='storage penetration percentage times 10')
    # parser.add_argument('--solarPen', default=6, help='solar penetration percentage times 10')
    parser.add_argument('--evPen', default=4, help='EV penetration percentage times 10')

    FLAGS, unparsed = parser.parse_known_args()
    print('running with arguments: ({})'.format(FLAGS))

    # define EV penetration and info
    EV_pen = float(FLAGS.evPen) / 10
    max_capacity = 24  # kWh
    charging_power = 40  # kW

    # define nodes and spot loads
    ppc = np.load('data/case123_ppc_reg_pq-0.79.npy', allow_pickle=True).item()
    spot_loads = ppc['bus'][:, 2] * 1000  # convert to kW from MW

    # total number of EVs in network at 100%
    total_EVs = 1200
    # total number of EVs with our given penetration
    n_EV = int(total_EVs * EV_pen)

    # total number of days to generate data
    days = 61

    # get the ids for the number of EVs for each node
    EVs_per_node, garage_ids, home_ids = assign_cars_to_nodes(spot_loads, total_EVs)

    print('garage_ids', garage_ids)

    # initialize output dictionaries
    start_dict, end_dict, charge_dict = initialize_dictionaries(nodes=len(EVs_per_node), days=days)

    # generate data
    start_dict, end_dict, charge_dict = generate_EV_data(days, n_EV, start_dict, end_dict, charge_dict,
                                                         garage_ids, home_ids, EVs_per_node, charging_power)

    np.savez('data/EV_charging_data'+str(EV_pen), start_dict=start_dict, end_dict=end_dict, charge_dict=charge_dict,
             charging_power=charging_power, EV_pen=EV_pen, garage_ids=garage_ids)
    print('saved data')

    """ # test loading of data
    data = np.load('EV_charging_data' + str(EV_pen)+'.npz', allow_pickle=True)
    start_dict = data['start_dict']
    print(start_dict)
    """
