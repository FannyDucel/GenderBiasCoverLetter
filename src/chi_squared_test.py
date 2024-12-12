"""
# Code by Clémentine Bleuze
A quick piece of code to compute a Chi Squared test (statistic significance)"""

from scipy.stats import chisquare
import numpy as np

def chi_squared(N, feminine_proportion, masculine_proportion):
    """Run a Chi Squared test with Gender Gap to find when the p-value is below 0.001.

    Args:
        N: Number of generations that are either masculine or feminine (i.e. without gender-neutral).
        feminine_proportion: The proportion of feminine generations (normalized so that feminine_proportion + masculine_proportion = 100%)
        masculine_proportion: The proportion of masculine generations (normalized so that feminine_proportion + masculine_proportion = 100%)

    Returns:
        Nothing, but prints the step at which the p-value is below 0.001.

    """

    # Expectations: 50-50% (gender equality)
    f_exp = np.array([0.5 * N, 0.5 * N]) # [N/2, N/2]
    f_obs = np.array([round(masculine_proportion * N, 0), round(feminine_proportion * N, 0)]) # [N*f_m, N*f_f]
    # to verify that a gap of 500 (-250 on one gender, +250 on the other) already has a p-value under 0.001
    f_obs = np.array([0.5*N - 250, 0.5*N + 250])

    print(chisquare(f_obs, f_exp))  # if p-value < 0.05, then the null hypothesis is rejected, i.e. the two distributions are different

    # Find when the p-value is below 0.001
    for step in range(0, N // 2, 50):
        print(step)

        f_obs = np.array([0.5 * N - step, 0.5 * N + step])
        stat, p = chisquare(f_obs, f_exp)

        if p < 0.001:
            print(f"step = {step}", 0.001)
            break
        else:
            if p < 0.01:
                print(f"step = {step}", 0.01)
            elif p < 0.05:
                print(f"step = {step}", 0.05)
            else:
                continue

# Computed only on generations that are masculine or feminine.
N = 16604

f_f = 0.32
f_m = 0.68

print(chi_squared(N, f_f, f_m))