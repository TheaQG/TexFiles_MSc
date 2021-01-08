def deconvolve(self, sigma):
    '''
        Deconvolution of the restored spectral data, DCT(data) * R.
        Takes in to account that data and R are of different lengths, so R is discretized
        to a lower resolution to be able to be multiplied with the data.

        Arguments:
        ----------
            sigma:              [float] Theoretical diffusion length to be used in transfer function.

        returns:
        --------
            depth:              [array of floats] Original x data of time series.
            data_decon:         [array of floats] Deconvolution of data multiplied with restoration filter.
    '''
    data = copy.deepcopy(self.y)
    depth = copy.deepcopy(self.t)

    w_PSD, OptF, M, R = self.Filters(sigma)

    if data.size < self.N_min:
        idx = math.ceil(self.N_min/data.size)
        R_short = R[0::idx]
        w_PSD_short = w_PSD[0::idx]
    else:
        R_short = R
        w_PSD_short = w_PSD

    data_f = sp.fft.dct(data, 2, norm='ortho')
    decon_f = data_f * R_short

    data_decon = sp.fft.dct(decon_f, 3, norm='ortho')

    return depth, data_decon
