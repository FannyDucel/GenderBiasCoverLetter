# Code by Clémentine Bleuze
from scipy.stats import chisquare
import numpy as np

# Computed only on generations that are masculine or feminine.
N = 16604

f_f = 0.32
f_m = 0.68

f_exp = np.array([0.5 * N, 0.5 * N]) # [N/2, N/2]
f_obs = np.array([round(f_m * N, 0), round(f_f * N, 0)]) # [N*f_m, N*f_f]
# to verify that a gap of 500 (-250 on one gender, +250 on the other) already has a p-value under 0.001
f_obs = np.array([0.5*N - 250, 0.5*N + 250])

print(chisquare(f_obs, f_exp))  # if p-value < 0.05, then the null hypothesis is rejected, i.e. the two distributions are different

# Find when the p-value is below 0.001
for pas in range(0, N//2, 50):
    print(pas)

    f_obs = np.array([0.5* N - pas, 0.5*N + pas])
    stat, p = chisquare(f_obs, f_exp)

    if p < 0.001:
        print(f"pas = {pas}", 0.001)
        break
    else:
        if p < 0.01:
            print(f"pas = {pas}", 0.01)
        elif p < 0.05:
            print(f"pas = {pas}", 0.05)
        else:
            continue