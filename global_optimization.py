import cvxpy as cp
import numpy as np

#import matplotlib.pyplot as plt
#plt.rcParams["font.family"] = "Times New Roman"
#plt.rcParams.update({'font.size': 18})
#plt.rcParams["figure.figsize"] = (7,6)


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


def global_optimization_alt(upper_demand, lower_demand, nodes_storage, coefs_mat_v, intercept_mat_v,
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


def load_demand_data_flex(data_path):
    data = np.load(data_path + 'demand_data.npz')
    demand = data['demand']
    demand_imag = data['demand_imag']
    res_ids = data['res_ids']
    com_ids = data['com_ids']
    demand_base = data['demand_base']
    demand_flex = data['demand_flex']

    return demand_base, demand, demand_imag, demand_flex, res_ids, com_ids


def global_optimization(P, imag, lambda_b, lambda_grid, Qmin, Qmax, Q0, cmax, dmax, y_l, y_d, y_c,
                        lambda_e, ubase, umax, phi,
                        start_time_network, end_time_network, charge_network, charging_power,
                        coefs_mat_v, intercept_mat_v, coefs_mat_tr, intercept_mat_tr, coefs_mat_ti, intercept_mat_ti,
                        v_max, v_min, T_max,
                        t_res = 0.25):

    T = P.shape[1]
    Nc = P.shape[0]
    N = v_max.shape[0]
    Nt = T_max.shape[0]

    T_max = T_max ** 2

    ev_c_dict = {}
    ev_q0 = 0
    ev_q_dict = {}
    ev_c_all_dict = {}

    for j in range(start_time_network):
        start_time = start_time_network[j]
        for i in range(len(start_time)):
            ev_c_dict[(j,i)] = cp.Variable(T)
            ev_q_dict[(j,i)] = cp.Variable(T+1)
    # end for loop

    Q = cp.Variable(shape=(Nc, T + 1))
    # c is charging variable
    c = cp.Variable(shape=(Nc, T + 1))
    # d is discharging variable
    d = cp.Variable(shape=(Nc, T + 1))
    # u is flexible load
    u = cp.Variable(shape=(Nc, T + 1))
    t_power = cp.Variable(shape=(Nt, T))
    v_mags = cp.Variable(shape=(N, T))
    ev_c_all = cp.Variable(shape=(Nc, T))

    s = cp.hstack([P + c - d + ev_c_all + u, imag]).T

    soc_constraints = [
        c >= 0,
        c <= np.tile(cmax, T),
        d >= 0,
        d <= np.tile(dmax, T),
        u >= 0,
        u <= np.tile(umax, T),
        Q >= Qmin,
        Q <= Qmax,
        Q[:, 0] == Q0,
        Q[:, 1:(T+1)] == y_l * Q[:, 0:T] + y_c * t_res * c[:, 0:T] - y_d * t_res * d[:, 0:T],
        cp.sum(u, axis=1) == np.sum(ubase, axis=1),
        0.5 * cp.sum(cp.abs(u - ubase), axis=1) <= phi * np.sum(ubase, axis=1)
    ]
    soc_constraints.append(t_power == (coefs_mat_tr @ s + intercept_mat_tr) ** 2
                       + (coefs_mat_ti @ s + intercept_mat_ti) ** 2)
    soc_constraints.append(v_mags == coefs_mat_v @ s + intercept_mat_v)
    for j in range(len(start_time_network)):
        start_time = start_time_network[j]
        end_time = end_time_network[j]
        charge = charge_network[j]
        for i in range(len(start_time)):
            ev_times_not = np.ones(T, dtype=int)
            ev_times_not[start_time[i]:end_time[i]] = 0
            ev_times_not = ev_times_not == 1
            soc_constraints.append(ev_c_dict[(j,i)][ev_times_not] == 0)
            soc_constraints.append(ev_c_dict[(j,i)] >= 0)
            soc_constraints.append(ev_q_dict[(j,i)][start_time[i]] == ev_q0)
            #print(charge[i])
            #print(ev_q_dict[i][-1])
            soc_constraints.append(ev_q_dict[(j,i)][end_time[i]+1] == charge[i])
            # add constraints where each variable in ev_c_dict is between 0 and ev_cmax = charging_power
            soc_constraints.append(ev_c_dict[(j,i)] >= 0)
            soc_constraints.append(ev_c_dict[(j,i)] <= charging_power)
            soc_constraints.append(ev_q_dict[(j,i)][1:] == y_l * ev_q_dict[(j,i)][0:-1] + y_c * t_res * ev_c_dict[(j,i)])
        # end for loop
        if len(start_time) > 0:
            ev_c_all_dict[j] = ev_c_dict[(j,0)]
            if len(start_time) > 1:
                for i in range(1,len(start_time)):
                    ev_c_all_dict[j] += ev_c_dict[(j,i)]
        else:
            ev_c_all_dict[j] = cp.Variable(1)
            soc_constraints.append(ev_c_all_dict[j] == 0)
        ev_c_all[j, :] = ev_c_all_dict[j]
    # end for loop

    e_cost = lambda_e @ cp.transpose(cp.pos(P + c - d + ev_c_all + u))
    v_cost = lambda_grid * (cp.sum_squares(cp.pos(v_mags - v_min) + cp.pos(v_max - v_mags)))
    t_cost = lambda_grid * (cp.sum_squares(cp.pos(T_max - t_power)))
    objective = cp.Minimize( v_cost + t_cost + e_cost
            + lambda_b * cp.sum_squares(c + d)
            )

    prob = cp.Problem(objective, soc_constraints)
    prob.solve(solver=cp.MOSEK)

    print('status', prob.status)

    cost = lambda_e @ cp.transpose(cp.pos(P + c.value - d.value + ev_c_all.value + u.value))

    return c.value, d.value, Q.value, u.value, ev_c_all.value, ev_c_dict, ev_q_dict, cost, prob.status


def close_idx(lst, k):
    idx = (np.abs(lst - k)).argmin()
    return idx


def expand_ev_times(times, t_res=0.25):
    nt = np.array([], dtype=int)
    day = 0
    for arr in times:
        nt = np.append(nt, arr + day / t_res * 24)
        day += 1

    nt = np.array(nt, dtype=int)
    return nt


def expand_inputs(qmins, qmaxs, Q0s, cmaxs, dmaxs, nodes_storage, nodes_solar, tariffs, res_ids, com_ids, Nc, T):
    qmins_ex = np.zeros(Nc)
    qmaxs_ex = np.zeros(Nc)
    Q0s_ex = np.zeros(Nc)
    cmaxs_ex = np.zeros(Nc)
    dmaxs_ex = np.zeros(Nc)
    lambda_e = np.zeros((Nc, T))

    for node in range(Nc):
        if node in nodes_storage:
            idx = close_idx(nodes_solar, node)
            qmins_ex[idx] = qmins[idx]
            qmaxs_ex[idx] = qmaxs[idx]
            Q0s_ex[idx] = Q0s[idx]
            cmaxs_ex[idx] = cmaxs[idx]
            dmaxs_ex[idx] = dmaxs[idx]
        if node in res_ids:
            lambda_e[node, :] = tariffs[0, :]
        elif node in com_ids:
            lambda_e[node, :] = tariffs[2, :]
        else:
            lambda_e[node, :] = tariffs[0, :]

    return qmins_ex, qmaxs_ex, Q0s_ex, cmaxs_ex, dmaxs_ex, lambda_e


def expand_ev_inputs(start_dict, end_dict, charge_dict, day, Nc):
    start_time_network = []
    end_time_network = []
    charge_network = []

    for node in range(Nc):
        # array of start times = start_dict[node][day]
        start_time1 = start_dict[node][day]
        end_time1 = end_dict[node][day]
        charge1 = charge_dict[node][day]
        start_time2 = start_dict[node][day+1]
        end_time2 = end_dict[node][day+1]
        charge2 = charge_dict[node][day+1]

        # expand data to 2 day horizon
        start_time = expand_ev_times([start_time1, start_time2])
        end_time = expand_ev_times([end_time1, end_time2])
        # print(end_time - start_time)
        end_time = end_time + (end_time - start_time) * 1.5
        end_time = np.array(end_time, dtype=int)
        # print(end_time - start_time)
        charge = np.concatenate([charge1, charge2])

        start_time_network.append(start_time)
        end_time_network.append(end_time)
        charge_network.append(charge)

    return start_time_network, end_time_network, charge_network


if __name__ == '__main__':
    name = 'rural_san_benito/'
    year = 2050

    lambda_b = 0.001
    lambda_grid = 125
    T = 24 * 4 * 2  # 15 minute resolution 2 day overlapping windows
    t_res = 15.0 / 60.0
    gamma_l = 1
    gamma_d = 0.93
    gamma_c = 0.93
    charging_power = 6.3

    data = np.load(name + 'storage_data.npz')
    nodes_storage = data['nodesStorage']
    nodes_solar = data['nodesSolar']
    qmins = data['qmin']
    qmaxs = data['qmax']
    cmaxs = -data['umin']
    dmaxs = data['umax']
    Q0s = qmaxs * 0.5
    phis = data['phis']

    data = np.load(name + 'demand_data.npz')
    res_ids = data['res_ids']
    com_ids = data['com_ids']
    demand_imag = data['demand_imag']
    # demand_ev = np.loadtxt(name + 'demand_solar_ev.csv')
    demand = np.loadtxt(name + 'demand_solar.csv')
    demand_flex = np.loadtxt(name + 'demand_flex.csv')
    tariffs = np.loadtxt(name + 'tariffs.csv')

    data = np.load(name + 'EV_charging_data' + '.npz', allow_pickle=True)
    start_dict = data['start_dict'][()]
    end_dict = data['end_dict'][()]
    charge_dict = data['charge_dict'][()]

    # load models
    data = np.load(name + 'model_coefs.npz')
    coefs_mat_v = data['coefs_mat_v']
    intercept_mat_v = data['intercept_mat_v']
    coefs_mat_tr = data['coefs_mat_tr']
    intercept_mat_tr = data['intercept_mat_tr']
    coefs_mat_ti = data['coefs_mat_ti']
    intercept_mat_ti = data['intercept_mat_ti']
    #coefs_upper_i = data['coefs_upper_i']  # reactive power models
    #coefs_lower_i = data['coefs_lower_i']  # first term is intercept second is coefficient

    # load voltage and transformer limits
    T_max = np.loadtxt(name + 'transformer_limits.csv')
    data = np.load(name + 'local' + '_metrics.npz')
    v_mags_all = data['v_mags_all']
    v_max, v_min = assign_voltage_limits(v_mags_all)
    _, v_max, v_min = clean_voltages(v_mags_all, v_max, v_min)

    n_days = int(demand.shape[1] * t_res / 24)
    t_day = int(24 / t_res)

    # initialize variables
    c_network = np.zeros(demand.shape)
    d_network = np.zeros(demand.shape)
    u_network = np.zeros(demand.shape)
    ev_network = np.zeros(demand.shape)
    cost_network = np.zeros(n_days)

    ubase = demand_flex
    umax = np.max(ubase, axis=1)

    Nc = demand.shape[0]
    #T_all = demand.shape[1]

    qmins, qmaxs, Q0s, cmaxs, dmaxs, lambda_e = \
        expand_inputs(qmins, qmaxs, Q0s, cmaxs, dmaxs, nodes_storage, nodes_solar, tariffs, res_ids, com_ids, Nc, T)

    if 'MOSEK' in cp.installed_solvers():
        print('MOSEK is installed')
    else:
        raise ValueError('MOSEK is not installed. Cannot run optimization')

    # daily loop
    for day in range(n_days-1):
        print('running day', day)

        # extract perfect foresight data for current loop
        demand_day = demand[:, day * t_day:(day+1) * t_day]
        demand_imag_day = demand_imag[:, day * t_day:(day+1) * t_day]

        start_time_network, end_time_network, charge_network = \
            expand_ev_inputs(start_dict, end_dict, charge_dict, day, Nc)

        # Global optimization
        c, d, Q, u, ev_c_all, ev_c_dict, ev_q_dict, cost, status = \
            global_optimization(demand_day, demand_imag_day, lambda_b, lambda_grid, qmins, qmaxs, Q0s, cmaxs, dmaxs, y_l, y_d, y_c,
                            lambda_e, ubase, umax, phis,
                            start_time_network, end_time_network, charge_network, charging_power,
                            coefs_mat_v, intercept_mat_v, coefs_mat_tr, intercept_mat_tr, coefs_mat_ti, intercept_mat_ti,
                            v_max, v_min, T_max)

        c_network[:, day * t_day:(day+1) * t_day] = c
        d_network[:, day * t_day:(day + 1) * t_day] = d
        u_network[:, day * t_day:(day + 1) * t_day] = u
        ev_network[:, day * t_day:(day + 1) * t_day] = ev_c_all
        cost_network[day] = cost

    np.savez(name + 'GC_data.npz', c_network=c_network, d_network=d_network, u_network=u_network,
             ev_network=ev_network, cost_network=cost_network)
    print('SAVED GC DATA')