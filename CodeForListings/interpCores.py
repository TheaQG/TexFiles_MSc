def interpCores(self, pad = 1):

    # Method in BackDiffuse class

    isoData = self.d18OData
    d_in = isoData['depth']
    x_in = isoData['d18O']


    if self.interpAll:
        valMin = d_in.min()
        valmax = d_in.max()
    else:
        valMin = self.depthMin - pad
        valMax = self.depthMax + pad


    d = d_in[(d_in >= valMin) & (d_in <= valMax)]
    x = x_in[(d_in >= valMin) & (d_in <= valMax)]

    diff = np.diff(d)
    Delta = round(min(diff), 3)

    d_min = Delta * np.ceil(d.values[0]/Delta)
    d_max = Delta * np.floor(d.values[-1]/Delta)

    n = int(1 + (d_max - d_min)/Delta)

    j_arr = np.linspace(0,n,n)
    dhat0 = d_min + (j_arr - 1)*Delta

    f = interpolate.CubicSpline(d,x)

    xhat0 = f(dhat0)

    dhat = dhat0[(dhat0 >= self.depthMin) & (dhat0 <= self.depthMax)]
    xhat = xhat0[(dhat0 >= self.depthMin) & (dhat0 <= self.depthMax)]

    return dhat, xhat, Delta
