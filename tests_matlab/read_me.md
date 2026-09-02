This folder contains the comparison of the python code with the matlab version.

run_baseline.py does the experiment with theta and rho true taken from the results_amplitude_vector_EIGt..mat file(which was generated with Amelia's code)

Comparaison.py compares the two versions, benchmarking the angular error and eig estimation with both codes.

The simulations show that the translation of the matlab baseline to python was successful (we have the same behavior, ie at low snr, high angular uncertainty, at high snr, low angular uncertainty)