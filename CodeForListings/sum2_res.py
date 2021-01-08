def sum2_res(params, x, y, dt, weights):

    # Calculate sum of the squared residuals.
    return np.sum(calc_res(params, x, y, dt, weights)**2)
