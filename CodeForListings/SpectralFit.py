
def SpectralFit(self, printFitParams=True, printDiffLen=True, printParamBounds=False,**kwargs):
    '''

        Arguments:
        ----------
            printFitParams: [bool] To print fitted parameters or not.
            **kwargs:       [tuple] Contains user specified boundaries for fit parameters

        returns:
        --------
            w_PSD:          [array of floats] Spectral frequencies.
            P_PSD:          [array of floats] Power Spectral Density.
            Pnoise:         [array of floats] Estimated noise function PSD.
            Psignal:        [array of floats] Estimated signal function, PSD.
            P_fit:          [array of floats] Estimated fit to PSD data.
            opt_fit_dict:   [tuple] Dictionary containing estimated fit parameters
            params_fit:     [array of floats] Estimated fit parameters, in array, from scipy.optimize.
            fit_func_val:   [array of floats] Estimated fit, from scipy.optimize.
            fit_dict:       [tuple] Dictionary from scipy.optimize.
    '''
    f, P = self.dct_psd()

    def calc_res(params, x, y, dt, weights):
        '''
            Calculates the log residuals between data, y, and model estimated from
            x and a given set of fit parameters.


            Arguments:
            ----------
                params:     [array of floats] Parameters to compute the model estimate from.
                x:          [array of floats] Data, x values.
                y:          [array of floats] Data, y values.
                dt:         [float] Spacing of x data.
                weights:    [array of floats] Weights to fudge residuals.

            returns:
            --------
                res:        [array of floats] Residuals of data vs model.
        '''
        P0, s_eta2, s_tot2, a1 = params

        Noise = self.func_Noise(x, s_eta2, a1, dt)
        Signal = self.func_Signal(x, P0, s_tot2)

        Pmod = Noise + Signal
        res = weights*(np.log10(y) - np.log10(np.copy(Pmod)))

        return res

    def sum2_res(params, x, y, dt, weights):
        '''
            Calculates the squared sum of residuals.

            Arguments:
            ----------
                params:     [array of floats] Parameters to compute the model estimate from.
                x:          [array of floats] Data, x values.
                y:          [array of floats] Data, y values.
                dt:         [float] Spacing of x data.
                weights:    [array of floats] Weights to fudge residuals.

            returns:
            --------
                sum2_res:   [float] Value of the sum of the squarred residuals.
                                    (We seek to minimize this).
        '''
        return np.sum(calc_res(params, x, y, dt, weights)**2)

    #Define the default boundaries for the different parameters.
    boundas = {}
    boundas['P0_Min'] = 1e-5
    boundas['P0_Max'] = 10000
    boundas['s_eta2_Min'] = 1e-10
    boundas['s_eta2_Max'] = 10
    boundas['a1_Min'] = 1e-7
    boundas['a1_Max'] = 0.4
    boundas['s_tot2_Min'] = 1e-7
    boundas['s_tot2_Max'] = 1

    #If user has specified bounds for params, it is passed through here.
    if list(kwargs.keys()):
        print('Setting fit param boundaries to user specifics')
        for j in list(kwargs.keys()):
            if j in list(bounds.keys()):
                bounds[j] = kwargs[j]
                if printParamBounds:
                    print(f'setting {j} as {kwargs[j]}')
    elif not list(kwargs.keys()):
        if printParamBounds:
            print('Using default boundaries for variance and a1')

    #Initial parameter guess.
    p0 = [0.005, 0.005, 0.01, 0.1]
    #Weights
    weights = np.ones_like(f)*1.

    #Optimization routine - minimizes residuals btw. data and model.
    params_fit, fit_func_val, fit_dict = sp.optimize.fmin_l_bfgs_b(sum2_res, p0, fprime=None, args = (f, P, self.dt, weights),\
                                    approx_grad=True, bounds = [(boundas['P0_Min'], boundas['P0_Max']), (boundas['s_eta2_Min'], boundas['s_eta2_Max']), \
                                    (boundas['s_tot2_Min'], boundas['s_tot2_Max']), (boundas['a1_Min'], boundas['a1_Max'])])

    P_fit = self.func_Noise(f, params_fit[1], params_fit[3], self.dt) + self.func_Signal(f, params_fit[0], params_fit[2])

    opt_fit_dict = {"P0_fit": params_fit[0], "s_eta2_fit": params_fit[1], "s_tot2_fit": params_fit[2], "a1_fit": params_fit[3]}

    P0_fit = opt_fit_dict['P0_fit']
    s_eta2_fit = opt_fit_dict['s_eta2_fit']
    s_tot2_fit = opt_fit_dict['s_tot2_fit']
    a1_fit = opt_fit_dict['a1_fit']

    if printFitParams:
        print('Fit Parameters:\n')
        print(f'P0 = {P0_fit}')
        print(f'Var = {s_eta2_fit}')
        print(f's_eta2 = {s_eta2_fit} m')
        print(f'Diff len = {s_tot2_fit*100} cm')
        print(f'a1 = {a1_fit}')


    w_PSD, P_PSD = self.dct_psd()


    Pnoise = self.func_Noise(w_PSD, opt_fit_dict['s_eta2_fit'],opt_fit_dict['a1_fit'], self.dt)
    Psignal = self.func_Signal(w_PSD, opt_fit_dict['P0_fit'], opt_fit_dict['s_tot2_fit'])

    s_eta2_fit = opt_fit_dict['s_eta2_fit']
    s_tot2_fit = opt_fit_dict['s_tot2_fit']

    if printDiffLen:
        print(f'Diff. len., fit [cm]: {s_tot2_fit*100:.3f}')

    return w_PSD, P_PSD, Pnoise, Psignal, P_fit, opt_fit_dict, params_fit, fit_func_val, fit_dict
