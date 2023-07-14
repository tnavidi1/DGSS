import cvxpy as cp
import numpy as np

import matplotlib.pyplot as plt


def local_controller_EV_flex_solve(P, lambda_b, T, Qmin, Qmax, Q0,
                       cmax, dmax, y_l, y_d, y_c,
                       lambda_e, ubase, umax, phi,
                       start_time, end_time, charge, charging_power, t_res = 0.25):

    ev_c_dict = {}

    ev_q0 = 0
    ev_q_dict = {}

    for i in range(len(start_time)):
        ev_c_dict[i] = cp.Variable(T)
        ev_q_dict[i] = cp.Variable(T+1)
    # end for loop

    Q = cp.Variable(T + 1)
    # c is charging variable
    c = cp.Variable(T)
    # d is discharging variable
    d = cp.Variable(T)
    # u is flexible load
    u = cp.Variable(T)

    soc_constraints = [
        c >= 0,
        c <= np.tile(cmax, T),
        d >= 0,
        d <= np.tile(dmax, T),
        u >= 0,
        u <= np.tile(umax, T),
        Q >= Qmin,
        Q <= Qmax,
        Q[0] == Q0,
        Q[1:(T+1)] == y_l * Q[0:T] + y_c * t_res * c[0:T] - y_d * t_res * d[0:T],
        cp.sum(u) == np.sum(ubase),
        0.5 * cp.sum(cp.abs(u - ubase)) <= phi * np.sum(ubase)
    ]

    for i in range(len(start_time)):
        ev_times_not = np.ones(T, dtype=int)
        ev_times_not[start_time[i]:end_time[i]] = 0
        ev_times_not = ev_times_not == 1
        soc_constraints.append(ev_c_dict[i][ev_times_not] == 0)
        soc_constraints.append(ev_c_dict[i] >= 0)
        soc_constraints.append(ev_q_dict[i][start_time[i]] == ev_q0)
        #print(charge[i])
        #print(ev_q_dict[i][-1])
        soc_constraints.append(ev_q_dict[i][end_time[i]+1] == charge[i])
        # add constraints where each variable in ev_c_dict is between 0 and ev_cmax = charging_power
        soc_constraints.append(ev_c_dict[i] >= 0)
        soc_constraints.append(ev_c_dict[i] <= charging_power)
        soc_constraints.append(ev_q_dict[i][1:] == y_l * ev_q_dict[i][0:-1] + y_c * t_res * ev_c_dict[i])

    # end for loop

    if len(start_time) > 0:
        ev_c_all = ev_c_dict[0]
        if len(start_time) > 1:
            for i in range(1,len(start_time)):
                ev_c_all += ev_c_dict[i]
    else:
        ev_c_all = cp.Variable(1)
        soc_constraints.append(ev_c_all == 0)

    #print(P.size)
    #print(ev_c_all.size)

    objective = cp.Minimize(
            lambda_e.reshape((1, lambda_e.size)) @ cp.reshape(cp.pos(P + c - d + ev_c_all + u), (T, 1))
            + lambda_b * cp.sum_squares(c + d)
            )

    prob = cp.Problem(objective, soc_constraints)
    prob.solve(solver=cp.ECOS)

    print('status', prob.status)

    cost = lambda_e.reshape((1, lambda_e.size)) @ np.maximum(P + c.value - d.value + ev_c_all.value, 0).reshape((T, 1))

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


def local_controller_outer(lambda_b, T, gamma_l, gamma_d, gamma_c, charging_power, nodes_storage, nodes_solar,
                           qmins, qmaxs, cmaxs, dmaxs, Q0s, demand, demand_flex, phis, tariffs, res_ids, com_ids,
                           start_dict, end_dict, charge_dict,
                           t_res= 0.25):
    # run LC for each node
    c_network = np.zeros(demand.shape)
    d_network = np.zeros(demand.shape)
    u_network = np.zeros(demand.shape)
    ev_network = np.zeros(demand.shape)
    cost_network = np.zeros(demand.shape[0])

    for node in range(demand.shape[0]):
        print('node:', node)
        P = demand[node, :]
        ubase = demand_flex[node, :]
        umax = np.max(ubase)
        phi = phis[node]
        if node in nodes_storage:
            idx = close_idx(nodes_solar, node)
            Qmin = qmins[idx]
            Qmax = qmaxs[idx]
            cmax = cmaxs[idx]
            dmax = dmaxs[idx]
            Q0 = Q0s[idx]
        else:
            Qmin = 0
            Qmax = 0
            cmax = 0
            dmax = 0
            Q0 = 0

        if node in res_ids:
            lambda_e = tariffs[0, :]
        elif node in com_ids:
            lambda_e = tariffs[2, :]
        else:
            lambda_e = tariffs[0, :]

        # array of start times = start_dict[node][day]
        start_time = start_dict[node]
        end_time = end_dict[node]
        charge = charge_dict[node]

        # expand data to full horizon
        start_time = expand_ev_times(start_time)
        end_time = expand_ev_times(end_time)
        #print(end_time - start_time)
        end_time = end_time + (end_time - start_time) * 1.5
        end_time = np.array(end_time, dtype=int)
        #print(end_time - start_time)
        charge = np.concatenate(charge)

        lambda_e = np.tile(lambda_e, 365)

        c, d, Q, u, ev_c_all, ev_c_dict, ev_q_dict, cost, status = \
            local_controller_outer_node(P, lambda_b, Qmin, Qmax, Q0, cmax, dmax, gamma_l,
                                       gamma_d, gamma_c, lambda_e, ubase, umax, phi, start_time, end_time,
                                       charge, charging_power,
                                    t_res= 0.25)

        c_network[node, :] = c
        d_network[node, :] = d
        u_network[node, :] = u
        ev_network[node, :] = ev_c_all
        cost_network[node] = cost

    return c_network, d_network, u_network, ev_network, cost_network


def local_controller_outer_node(P, lambda_b, Qmin, Qmax, Q0, cmax, dmax, gamma_l,
                                       gamma_d, gamma_c, lambda_e, ubase, umax, phi, start_time, end_time,
                                       charge, charging_power,
                                    t_res= 0.25):
    # Within node run LC for T period over whole year
    #for i in range(int(P.size * t_res / 24)):

    T = P.size
    #print('T', T)

    c, d, Q, u, ev_c_all, ev_c_dict, ev_q_dict, cost, status = \
        local_controller_EV_flex_solve(P, lambda_b, T, Qmin, Qmax, Q0, cmax, dmax, gamma_l,
                                       gamma_d, gamma_c, lambda_e, ubase, umax, phi, start_time, end_time,
                                       charge, charging_power)

    print('cost', cost)

    return c, d, Q, u, ev_c_all, ev_c_dict, ev_q_dict, cost, status


if __name__ == '__main__':
    name = 'rural_san_benito/'
    year = 2050

    lambda_b = 0.001
    lambda_grid = 125
    T = 24 * 4 * 2  # 15 minute resolution 2 day overlapping windows
    t_res = 15.0/60.0
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
    #demand_ev = np.loadtxt(name + 'demand_solar_ev.csv')
    demand = np.loadtxt(name + 'demand_solar.csv')
    demand_flex = np.loadtxt(name + 'demand_flex.csv')
    tariffs = np.loadtxt(name + 'tariffs.csv')

    data = np.load(name + 'EV_charging_data' + '.npz', allow_pickle=True)
    start_dict = data['start_dict'][()]
    end_dict = data['end_dict'][()]
    charge_dict = data['charge_dict'][()]

    c_network, d_network, u_network, ev_network, cost_network = \
        local_controller_outer(lambda_b, T, gamma_l, gamma_d, gamma_c, charging_power, nodes_storage, nodes_solar,
                           qmins, qmaxs, cmaxs, dmaxs, Q0s, demand, demand_flex, phis, tariffs, res_ids, com_ids,
                           start_dict, end_dict, charge_dict,
                           t_res= 0.25)

    np.savez(name + 'LC_data.npz', c_network=c_network, d_network=d_network, u_network=u_network,
             ev_network=ev_network, cost_network=cost_network)
    print('SAVED LC DATA')



