import numpy as np

N = 100 # number of samples
P = 12 # signal power
std = np.sqrt(P)

x = std * np.random.randn(N,1) # random samples with deviation of sqrt(P)
np.mean(pow(x, 2)) # mean power of the samples

# task1: generate noises based on a theoretical SNR of 4 (i.e., N0=3)
SNR = 4 # signal-to-noise ratio in dB
N0 = P / SNR # noise power
sigma = np.sqrt(N0) # standard deviation of noise
# 1-1: generate noise samples with zero mean and N0 (power) of 3
noise = sigma * np.random.randn(N, 1)
# 1-2: calculate the SNR of each signal sample
snr = np.zeros((N, 1))
for i in range(N):
    snr[i] = np.power(x[i], 2) / np.power(noise[i], 2)
# 1-3: calculate the mean SNR
mean_snr = np.mean(snr)


# task2: sample K signal/noise samples with a given SNR pdf
# 0 <= SNR < 2: 50%
# 2 <= SNR < 3: 30%
# 3 <= SNR < 4: 20%
# Define the SNR probability distribution function
snr_probs = [0.5, 0.3, 0.2] # 50% for SNR < 2, 30% for 2 <= SNR < 3, 20% for 3 <= SNR < 4
snr_ranges = [[0, 2], [2, 3], [3, 4]] # SNR ranges for each probability

K = 50
p = np.random.rand(K,1)
# hint: put the sampling results in y
y = np.zeros((N, 1))
for i in range(K):
    snr_range_idx = np.argmax(np.random.multinomial(1, snr_probs))
    snr_range = snr_ranges[snr_range_idx]
    y[i] = np.random.uniform(snr_range[0], snr_range[1])