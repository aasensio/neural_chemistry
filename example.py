import numpy as np
import emulator

if (__name__ == '__main__'):

    n_models = 32
    batch_size = 32

    mols = [0, 1, 2, 3, 4, 5]

    nh = 10.0**np.random.uniform(np.log10(1e4), np.log10(1e7), size=n_models)
    T = np.random.uniform(10.0, 80.0, size=n_models)
    crir = 10.0**np.random.uniform(np.log10(1e-17), np.log10(1e-15), size=n_models)
    sulfur = 10.0**np.random.uniform(np.log10(7.5e-8), np.log10(1.5e-5), size=n_models)
    uv_flux = 10.0**np.random.uniform(np.log10(0.1), np.log10(1e4), size=n_models)
    
    t = np.logspace(0, 7, 120)

    net = emulator.ChemistryEmulator(gpu=0, verbose=True)
    abundance = net.evaluate(t, T, nh, crir, sulfur, uv_flux, batch_size=batch_size, species=None)

    abundance_subset = net.evaluate(t, T, nh, crir, sulfur, uv_flux, batch_size=batch_size, species=[0,1,2,3,4,5])

    # age = np.array([1.0e6, 1.0e7])
    # Tgas = np.array([15.0, 20.0])
    # nH = np.array([1.0e6, 1.0e7])
    # CRIR = np.array([1.5e-17, 1.5e-17])
    # S = np.array([1.5e-5, 7.5e-8])
    # fUV = np.array([1.0, 1.0])
    # ab = net.evaluate(age, Tgas, nH, CRIR, S, fUV)