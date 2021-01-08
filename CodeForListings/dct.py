def dct(self):

    # Method of SpectralDecon class. Performs Discrete Cosine Transform.

    data = copy.deepcopy(self.y)
    depth = copy.deepcopy(self.t)

    if data.size < self.N_min:
        N = math.ceil(self.N_min/data.size) * data.size
    else:
        N = data.size

    DCT = sp.fft.dct(data, 2, n = N, norm='ortho')
    freq = np.fft.fftfreq(2*N, self.dt)[:(2*N)//2]

    return freq, DCT
