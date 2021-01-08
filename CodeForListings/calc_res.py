def calc_res(params, x, y, dt, weights):
    # Set parameters as input parameters
    P0, s_eta2, s_tot2, a1 = params

    # Define signal and noise function based on given params
    Noise = self.func_Noise(x, s_eta2, a1, dt)
    Signal = self.func_Signal(x, P0, s_tot2)

    # Define model as sum of noise and signal
    Pmod = Noise + Signal
    # Calculate (weighted) log residual
    res = weights*(np.log10(y) - np.log10(np.copy(Pmod)))

    return res
