import numpy as np
import pandas as pd
import argparse
#from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import Lasso, LinearRegression
#import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.svm import SVR


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

    if FLAGS.network == 'iowa':
        dir_network = 'Iowa_feeder/'
        t_res = 24  # number of points in a day
    elif FLAGS.network == '123':
        dir_network = 'IEEE123-1ph-test/'
        t_res = 24 * 4  # number of points in a day for 15 min resolution
    else:
        print('reverting to default Iowa network')
        dir_network = 'Iowa_feeder/'
        t_res = 24  # number of points in a day

    return storagePen, solarPen, evPen, dir_network, n_days, t_res


def train_test_split(data, t_idx):
    data_train = data[:, 0:t_idx]
    data_test = data[:, t_idx:]

    return data_train, data_test


def get_sample_weights(x, y):
    weights = np.ones(y.shape[0]) * 1
    percentile = int(y.shape[0] * 0.05)
    inds = np.argsort(y)
    weights[inds[0:percentile]] = 20
    weights[inds[-percentile:]] = 60

    return weights


def error_close_0(y, y_lin):
    return np.max(2 * np.abs(y - y_lin) / (np.abs(y) + np.abs(y_lin))) * 100


def error_from_max(y, y_lin):
    return np.max(np.abs(y - y_lin) / np.max(np.abs(y))) * 100


def error_from_mean(y, y_lin):
    return np.max(np.abs(y - y_lin) / np.mean(np.abs(y))) * 100


def train_lin_models(X_train, y_train, model='v'):

    coefs_mat = np.zeros((y_train.shape[0], X_train.shape[0]))
    intercept_mat = np.zeros((y_train.shape[0], 1))
    error_max = np.zeros((y_train.shape[0], 1))

    for i in range(y_train.shape[0]):

        if model == 'v':
            y = y_train[i, :] * 100
        else:
            y = y_train[i, :]

        if model == 't':
            #model_l = Lasso(alpha=0.01, fit_intercept=True, normalize=False, positive=True)
            model_l = LinearRegression(fit_intercept=True, normalize=False)#, positive=True)

            #weights = get_sample_weights(X_train.T, y.T)
            #y_lin = model_l.fit(X_train.T, y.T, sample_weight=weights).predict(X_train.T)

            y_lin = model_l.fit(X_train.T, y.T).predict(X_train.T)

        elif model == 'v':
            model_l = LinearRegression(fit_intercept=True, normalize=False)#, positive=True)
            # model_l = SVR(kernel='linear', C=1, gamma='auto')

            #weights = get_sample_weights(X_train.T, y.T)
            #y_lin = model_l.fit(X_train.T, y.T, sample_weight=weights).predict(X_train.T)

            #X_train = -X_train

            y_lin = model_l.fit(X_train.T, y.T).predict(X_train.T)

        if np.mean(np.abs(y)) > 1e-5:
            error_max[i] = error_from_max(y, y_lin)
        else:
            print('values are too close to 0')

        coefs = model_l.coef_
        intercept = model_l.intercept_

        #if model == 'v':
            #coefs = -coefs

        coefs_mat[i, :] = coefs
        intercept_mat[i, :] = intercept

    return coefs_mat, intercept_mat, error_max


def fit_quantile_regression(data, quantile=0.95):
    ind_var = data.columns[0]
    mod = smf.quantreg('reactive ~ real', data)
    res = mod.fit(q = quantile)

    return [quantile, res.params['Intercept'], res.params[ind_var]] + \
           res.conf_int().loc[ind_var].tolist()


def train_reactive_models(real_x, imag_x):

    #coeff_mat = np.zeros((netDemandFull.shape[0], int(24 * 60 / time_resolution)), dtype=object)
    coeffs_upper = np.zeros((real_x.shape[0], 2))
    coeffs_lower = np.zeros((real_x.shape[0], 2))

    for node in range(real_x.shape[0]):
        x = real_x[node, :]
        y = imag_x[node, :]

        array = np.vstack((x, y)).T
        data = pd.DataFrame(array, columns=['real', 'reactive'])

        coeff_upper = fit_quantile_regression(data, quantile=0.9)
        coeff_lower = fit_quantile_regression(data, quantile=0.1)

        coeffs_upper[node, :] = coeff_upper[1:3]
        coeffs_lower[node, :] = coeff_lower[1:3]  # first term is intercept second is coefficient

    return coeffs_upper, coeffs_lower

def test_models(x, y, coefs, intercept):

    errors_all = np.zeros(y.shape)
    y_preds = np.zeros(y.shape)

    for i in range(y.shape[0]):
        y_test = np.dot(coefs[i, :], x) + intercept[i]
        if np.mean(np.abs(y)) > 1e-5:
            errors_all[i, :] = error_from_max(y[i, :], y_test)
            if error_from_max(y[i, :], y_test) > 25:
                print('node', i, np.max(np.abs(y[i, :])))
            y_preds[i, :] = y_test
        else:
            print('values are too close to 0')
            y_preds[i, :] = y_test

    return errors_all, y_preds


def print_correlation(v, pq):
    corrs = np.corrcoef(np.vstack((v, pq)))
    print(corrs[559, v_profile_test.shape[0]:])
    print(corrs[210, v_profile_test.shape[0]:])

    return True


def plot_errors(error_v, error_ti, error_tr, v_profile_test, y_v, t_profile_imag_test, y_ti, t_profile_real_test, y_tr):
    import matplotlib.pyplot as plt

    plt.figure()
    ind = np.unravel_index(error_v.argmax(), error_v.shape)
    print(ind)
    plt.scatter(v_profile_test[ind[0], :].T, y_v[ind[0], :].T)
    plt.scatter(v_profile_test[ind[0], :].T, v_profile_test[ind[0], :].T + 1)
    plt.scatter(v_profile_test[ind[0], :].T, v_profile_test[ind[0], :].T - 1)

    plt.figure()
    spec = 209
    plt.scatter(v_profile_test[spec, :].T, y_v[spec, :].T)
    plt.scatter(v_profile_test[spec, :].T, v_profile_test[spec, :].T + 1)
    plt.scatter(v_profile_test[spec, :].T, v_profile_test[spec, :].T - 1)

    plt.figure()
    ind = np.unravel_index(error_ti.argmax(), error_ti.shape)
    print(ind)
    plt.scatter(t_profile_imag_test[ind[0], :].T, y_ti[ind[0], :].T)
    plt.scatter(t_profile_imag_test[ind[0], :].T, 1.1 * t_profile_imag_test[ind[0], :].T)
    plt.scatter(t_profile_imag_test[ind[0], :].T, 0.9 * t_profile_imag_test[ind[0], :].T)

    plt.figure()
    ind = np.unravel_index(error_tr.argmax(), error_tr.shape)
    print(ind)
    plt.scatter(t_profile_real_test[ind[0], :].T, y_tr[ind[0], :].T)
    plt.scatter(t_profile_real_test[ind[0], :].T, 1.1 * t_profile_real_test[ind[0], :].T)
    plt.scatter(t_profile_real_test[ind[0], :].T, 0.9 * t_profile_real_test[ind[0], :].T)

    plt.show()
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train data driven models')
    parser.add_argument('--train', default=1, help='random seed')
    parser.add_argument('--storagePen', default=1, help='storage penetration percentage times 10')
    parser.add_argument('--solarPen', default=5, help='solar penetration percentage times 10')
    parser.add_argument('--evPen', default=5, help='EV penetration percentage times 10')
    parser.add_argument('--network', default='iowa', help='name of network to simulate')
    parser.add_argument('--days', default=60, help='number of days to simulate')

    FLAGS, unparsed = parser.parse_known_args()
    print('running with arguments: ({})'.format(FLAGS))

    path_networks = 'Networks/'
    path_absolute = './'

    # parse inputs
    storagePen, solarPen, evPen, dir_network, n_days, t_res = parse_inputs(FLAGS)

    # Define length of training set
    t_idx = t_res * 90  # 30 days of training data

    # load grid data
    v_profile = np.loadtxt(path_absolute + path_networks + dir_network + 'v_profile.csv')
    t_profile_real = np.loadtxt(path_absolute + path_networks + dir_network + 't_profile_real.csv')
    t_profile_imag = np.loadtxt(path_absolute + path_networks + dir_network + 't_profile_imag.csv')
    taps_profile = np.loadtxt(path_absolute + path_networks + dir_network + 'taps_profile.csv')

    """
    print(np.max(v_profile))
    print(np.min(v_profile))
    print(v_profile.shape)
    print(np.sum(v_profile[:, -30*24:] > 1.04))
    print(np.sum(v_profile[:, -30*24:] > 1.03))
    print(np.sum(v_profile[:, -30*24:] < 0.96))
    print(np.sum(v_profile[:, -30*24:] < 0.95))
    print(np.sum(v_profile[:, -30*24:] < 0.94))
    print(t_profile_real.shape)
    print(np.sum(np.sqrt(t_profile_real[:, -30*24:]**2 + t_profile_imag[:, -30*24:]**2) >
                 0.8 * np.tile(np.max(np.sqrt(t_profile_real**2 + t_profile_imag**2), axis=1).reshape((535,1)), (1, 30*24))))
    #print(0.8 * np.max(np.sqrt(t_profile_real**2 + t_profile_imag**2), axis=1))
    print(taps_profile[:, -48:])
    """

    # load demand data
    # demand = np.loadtxt(path_networks + dir_network + 'raw_demand.csv')
    # demand = np.loadtxt(path_networks + dir_network + 'demand_solar.csv')
    demand = np.loadtxt(path_networks + dir_network + 'demand_solar_ev.csv')
    demand_imag = np.loadtxt(path_networks + dir_network + 'raw_demand_imag.csv')

    # Stack real and reactive power demand into training and test set
    pq_train, pq_test = train_test_split(np.vstack((demand, demand_imag)), t_idx)

    # Split output data into train and test sets
    v_profile_train, v_profile_test = train_test_split(v_profile, t_idx)
    t_profile_real_train, t_profile_real_test = train_test_split(t_profile_real, t_idx)
    t_profile_imag_train, t_profile_imag_test = train_test_split(t_profile_imag, t_idx)
    # taps_profile_train, taps_profile_test = train_test_split(taps_profile, t_idx)

    print('demand shape', demand.shape)
    print('v shape', v_profile_train.shape)
    print('t shape', t_profile_real_train.shape)
    # print('taps shape', taps_profile_train.shape)

    # Train grid models
    coefs_mat_v, intercept_mat_v, error_v = train_lin_models(pq_train, v_profile_train, model='v')
    print('transformer models')
    coefs_mat_tr, intercept_mat_tr, error_tr = train_lin_models(pq_train, t_profile_real_train, model='t')
    coefs_mat_ti, intercept_mat_ti, error_ti = train_lin_models(pq_train, t_profile_imag_train, model='t')

    print('Training % error')
    print('mean max voltage, transformer errors')
    print(np.mean(error_v))
    print(np.mean(error_ti))
    print(np.mean(error_tr))
    print('maximum max voltage and transformer errors')
    print(np.max(error_v))
    print(np.max(error_ti))
    print(np.max(error_tr))

    v_profile_test = 100 * v_profile_test

    # test grid model errors on test set
    error_v, y_v = test_models(pq_test, v_profile_test, coefs_mat_v, intercept_mat_v)
    error_tr, y_tr = test_models(pq_test, t_profile_real_test, coefs_mat_tr, intercept_mat_tr)
    error_ti, y_ti = test_models(pq_test, t_profile_imag_test, coefs_mat_ti, intercept_mat_ti)

    error_v = np.nan_to_num(error_v, copy=False, nan=0.0)
    error_tr = np.nan_to_num(error_tr, copy=False, nan=0.0)
    error_ti = np.nan_to_num(error_ti, copy=False, nan=0.0)

    print('Testing % error')
    print('mean max voltage, transformer errors')
    print(np.mean(np.max(error_v, axis=1)))
    print(np.mean(np.max(error_ti, axis=1)))
    print(np.mean(np.max(error_tr, axis=1)))
    print('maximum max voltage and transformer errors')
    print(np.max(error_v))
    print(np.max(error_ti))
    print(np.max(error_tr))

    plot_errors(error_v, error_ti, error_tr, v_profile_test, y_v, t_profile_imag_test, y_ti, t_profile_real_test, y_tr)

    # Train reactive power models
    coefs_upper_i, coefs_lower_i = train_reactive_models(demand, demand_imag)

    np.savez(path_networks + dir_network + 'model_coefs.npz',
             coefs_mat_v=coefs_mat_v, intercept_mat_v=intercept_mat_v, coefs_mat_tr=coefs_mat_tr,
             intercept_mat_tr=intercept_mat_tr, coefs_mat_ti=coefs_mat_ti, intercept_mat_ti=intercept_mat_ti,
             coefs_upper_i=coefs_upper_i, coefs_lower_i=coefs_lower_i)
    print('Saved model coefs')






