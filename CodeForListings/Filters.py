def Filters(self, sigma):
    '''
        Computes spectral filters. Computes optimalfilter, OptFilter = Psignal / (Psignal + Pnoise),
        exponential transfer function, M, and restoration filter, R = OptFilter * M^(-1).


        Arguments:
        ----------
            sigma:              [float] Theoretical diffusion length to be used in transfer function.

        returns:
        --------
            w_PSD:              [array of floats] Spectral frequencies
            OptFilter:          [array of floats] Optimal filter, as a function of frequencies.
            M:                  [array of floats] Transfer function, filtering due to diffusion.
            R:                  [array of floats] Total restoration filter.

    '''
    w_PSD, P_PSD, Pnoise, Psignal, P_fit, _, _ , _, _ = self.SpectralFit(printFitParams=False, printDiffLen=False, printParamBounds=False)

    OptFilter = Psignal / (Pnoise + Psignal)
#        sigma = 0.05#s_eta2_fit
    M = np.exp(-(2 * np.pi * w_PSD)**2 * sigma**2 / 2)

    R = OptFilter * M**(-1)

    return w_PSD, OptFilter, M, R
