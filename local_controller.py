import cvxpy as cp
import numpy as np

import matplotlib.pyplot as plt


def local_controller_solve(P, lambda_b, lambda_cap, T, t_res, Qmin, Qmax, Q0,
                       cmax, dmax, gamma_l, gamma_d, gamma_c, lambda_e, umin, umax):
    """
    :param P: net uncontrollable load (kW)
    :param lambda_b: battery operation penalty ($)
    :param lambda_cap: bounds violation penalty ($)
    :param T: Length of MPC horizon
    :param t_res: time resolution (fraction of hour)
    :param Qmin: Minimum charge level (kWh)
    :param Qmax: Maximum charge level (kWh)
    :param Q0: Initial charge level (kWh)
    :param cmax: Maximum charge rate (kW)
    :param dmax: Maximum discharge rate (kW)
    :param gamma_l: Leakage coefficient (% / t_res)
    :param gamma_d: Discharge efficiency (%)
    :param gamma_c: Charge efficiency (%)
    :param lambda_e: Price of electricity ($)
    :param umin: lower bound from GC (kW)
    :param umax: upper bound from GC (kW)
    :return: Charge, discharge, SOC profile, bounds violation, controller status
    """

    # TOU price arbitrage within

    # Q is battery charge level. First entry must be placed to a constant input
    Q = cp.Variable(T + 1)
    # c is charging variable
    c = cp.Variable(T)
    # d is discharging variable
    d = cp.Variable(T)

    soc_constraints = [
        c >= 0,
        c <= np.tile(cmax, T),
        d >= 0,
        d <= np.tile(dmax, T),
        Q >= Qmin,
        Q <= Qmax,
        # P + c - d + ev_c_all >= umin,
        # P + c - d + ev_c_all <= umax,
        # np.tile(u_min,T) <= P+c-d, P+c-d <= np.tile(u_max,T),  # moved to soft constraint
        Q[0] == Q0,
        Q[1:(T+1)] == gamma_l * Q[0:T] + gamma_c * t_res * c[0:T] - gamma_d * t_res * d[0:T]
    ]

    # The @ symbol is matrix multiplication
    objective = cp.Minimize(
            lambda_e.reshape((1, lambda_e.size)) @ cp.reshape(cp.pos(P + c - d), (T, 1))
            + lambda_cap * cp.sum((cp.pos(P + c - d - umax) + cp.pos(umin - (P + c - d))) ** 2)
            )
    if lambda_b > 0:
        objective += cp.Minimize(lambda_b * cp.sum_squares(c + d))

    prob = cp.Problem(objective, soc_constraints)
    prob.solve(solver=cp.ECOS)

    try:
        bounds_cost = lambda_cap * np.sum((np.maximum(P + c.value - d.value - umax, 0)
                            + np.maximum(umin - (P + c.value - d.value), 0)) ** 2)
    except:
        print('controller cost evaluation failed')
        bounds_cost = np.nan

    return c.value, d.value, Q.value, bounds_cost, prob.status


def local_controller_EV_solve(P, lambda_b, lambda_cap, T, Qmin, Qmax, Q0,
                       cmax, dmax, y_l, y_d, y_c,
                       lambda_e, umin, umax,
                       start_time, end_time, charge, charging_power):

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

    soc_constraints = [
        c >= 0,
        c <= np.tile(cmax, T),
        d >= 0,
        d <= np.tile(dmax, T),
        Q >= Qmin,
        Q <= Qmax,
        # modify constraints P + c - d <= u_max and u_min to be P + c - d + ev_c_all
        # P + c - d + ev_c_all >= umin,
        # P + c - d + ev_c_all <= umax,
        # np.tile(u_min,T) <= P+c-d, P+c-d <= np.tile(u_max,T),  # moved to soft constraint
        Q[0] == Q0,
        Q[1:(T+1)] == y_l * Q[0:T] + y_c * c[0:T] - y_d * d[0:T]
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
        soc_constraints.append(ev_q_dict[i][1:] == y_l * ev_q_dict[i][0:-1] + y_c * ev_c_dict[i])

    # end for loop

    if len(start_time) > 0:
        ev_c_all = ev_c_dict[0]
        if len(start_time) > 1:
            for i in range(1,len(start_time)):
                ev_c_all += ev_c_dict[i]
    else:
        ev_c_all = cp.Variable(1)
        soc_constraints.append(ev_c_all == 0)


    objective = cp.Minimize(
            lambda_e.reshape((1, lambda_e.size)) @ cp.reshape(cp.pos(P + c - d + ev_c_all), (T, 1))
            + lambda_cap * cp.sum((cp.pos(P + c - d + ev_c_all - umax) + cp.pos(umin - (P + c - d + ev_c_all))) ** 2)
            + lambda_b * cp.sum_squares(c + d + ev_c_all)
            )

    prob = cp.Problem(objective, soc_constraints)
    prob.solve(solver=cp.ECOS)

    # print('P', P)
    # print(np.sum(P))
    # print(P + c.value - d.value + ev_c_all.value)

    try:
        bounds_cost = lambda_cap * np.sum((np.maximum(P + c.value - d.value + ev_c_all.value - umax, 0)
                            + np.maximum(umin - (P + c.value - d.value + ev_c_all.value), 0)) ** 2)
    except:
        print('controller cost evaluation failed')
        bounds_cost = np.nan

    # print("cost e", lambda_e.reshape((1, lambda_e.size)) @ np.maximum(P + c.value - d.value + ev_c_all.value, 0).reshape((T, 1)) )
    # print('cost of bounds', bounds_cost )

    return c.value, d.value, Q.value, ev_c_all.value, ev_c_dict, ev_q_dict, bounds_cost, prob.status


if __name__ == '__main__':

    lambda_b = 0.01
    lambda_cap = 100
    T = 24 * 4  # 15 minute resolution
    t_res = 15.0/60.0
    Qmin = 0
    Qmax = 12
    cmax = Qmax/3.0
    dmax = Qmax/3.0

    P = np.array([3 * np.sin(x * 2 * np.pi / T - np.pi) + 2 for x in np.arange(T)])
    Q0 = Qmax / 2.0

    gamma_l = 0.9999
    gamma_d = 0.95
    gamma_c = 0.95
    lambda_e = np.hstack((.202 * np.ones((1, 16*4)), .463 * np.ones((1, 5*4)), .202 * np.ones((1, 3*4))))
    umin = -2*np.ones(T)
    umax = 6*np.ones(T)

    c, d, Q, bounds_cost, status = local_controller_solve(P, lambda_b, lambda_cap, T, t_res, Qmin, Qmax, Q0,
                                                        cmax, dmax, gamma_l, gamma_d, gamma_c, lambda_e, umin, umax)

    #print('charge profile', c - d)
    print('bounds violation', bounds_cost)
    print('controller status', status)

    plt.figure()
    plt.plot(P)
    plt.plot(c - d)
    plt.plot(P + c - d)
    plt.plot(lambda_e)
    plt.legend(('load', 'battery', 'net', 'price'))
    plt.show()

    P = np.array([0]*4*4*15 + [200]*2*15 + [0]*3*4*15 + [200]*2*15 + [0]*1*4*15 + [200]*2*15 + [0]*4*4*15 + [200]*2*15
                 + [0]*4*4*15 + [200]*2*15 + [0]*1*4*15 + [200]*2*15 + [0]*4*4*15)
    Q0 = Qmax / 2.0

    start_time, end_time, charge, charging_power = np.loadtxt('EV_data.csv')

    c, d, Q, ev_c_all, ev_c_dict, ev_q_dict, bounds_cost, status = local_controller_EV_solve(P, lambda_b, lambda_cap, T,
                                                                                             Qmin, Qmax, Q0,
                                                                                             cmax, dmax, gamma_l,
                                                                                             gamma_d, gamma_c, lambda_e,
                                                                                             umin,
                                                                                             umax, start_time, end_time,
                                                                                             charge, charging_power)

    # print('charge profile', c - d)
    print('bounds violation', bounds_cost)
    print('controller status', status)

    x = np.arange(4*24*15)/4./15.
    plt.figure()
    plt.plot(x, P)
    plt.legend(('Demand',), loc='upper right')
    plt.ylabel('Demand [kW]')
    plt.xlabel('Time [hour]')
    plt.ylim(-150,210)
    plt.figure()
    plt.plot(x, P + c - d)
    plt.plot(x, c - d)
    plt.legend(('Net demand','Battery'))
    plt.ylabel('Demand [kW]')
    plt.xlabel('Time [hour]')
    plt.ylim(-150, 210)
    plt.show()

