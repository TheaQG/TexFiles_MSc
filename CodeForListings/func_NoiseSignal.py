
def func_Noise(self, w, s_eta2, a1, dz):

    # Calculate the noise function based on input params
    return (s_eta2**2 * dz) / (np.abs(1 - a1 * np.exp(- 2 * np.pi * 1j * w * dz))**2)



def func_Signal(self, w, p0, s_tot2):

    # Calculate the signal function based on input params
    return p0 * np.exp(- (2 * np.pi * w * s_tot2)**2)
