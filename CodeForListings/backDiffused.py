def backDiffused(self, N=2000, print_Npeaks=True):
    '''
        Method to compute the maximal diffusion length that still give ysInSec
        peaks. Computes first any value that returns ysInSec peaks, and computes
        then the maximum value that still returns that exact number of peaks.

        Arguments:
        ----------
            N:              [int] Number of points to generate spectral data with.

        returns:
        --------
            depthEst:       [arr of floats] Depth to estimated data.
            dataEst:        [arr of floats] Backdiffused d18O data.
            diffLenFin:     [float] Final diffusion length estimate.
            idxPeak:        [arr of idxs] Number of peaks in the final data set.

    '''
    sigma_rangeHL = self.diffLenEstimateHL()
    sigma_FitEst = self.spectralEstimate()

    dInt, d18OInt, Delta = self.interpCores()

    diffLen0 = min(min(sigma_rangeHL), sigma_FitEst) - 0.01
    print(f'Starting sigma: {diffLen0*100:.2f} [cm]')

    decon_inst = SpectralDecon(dInt, d18OInt, N)

    depth0, dataD0 = decon_inst.deconvolve(diffLen0)

    from scipy import signal
    N_peaks = 0

    depth = depth0
    data = dataD0
    diffLen = diffLen0


    while N_peaks != self.ysInSec:
        depth, data = decon_inst.deconvolve(diffLen)
        idxPeak = signal.find_peaks(data, distance=3)[0]
        N_peaks = len(idxPeak)
        if print_Npeaks:
            print(len(idxPeak))

        if N_peaks > self.ysInSec:
            diffLen -= 0.0005
        if N_peaks < self.ysInSec:
            diffLen += 0.0005

    while N_peaks == self.ysInSec:
        depth, data = decon_inst.deconvolve(diffLen)
        idxPeak = signal.find_peaks(data, distance=3)[0]
        N_peaks = len(idxPeak)
        if print_Npeaks:
            print(len(idxPeak))
        diffLen += 0.0001

    diffLen -= 0.0002
    depth, data = decon_inst.deconvolve(diffLen)
    idxPeak = signal.find_peaks(data, distance=3)[0]
    N_peaks = len(idxPeak)

    print(f'Final sigma: {diffLen*100:.2f} [cm]')
    print(f'Final # of peaks: {N_peaks}')
    depthEst = depth
    dataEst = data
    diffLenFin = diffLen

    return depthEst, dataEst, diffLenFin, idxPeak
