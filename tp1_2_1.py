import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# returns an np array of random numbers for the given distribution and parameters
# for norm dist, a is mean, b is stddev
# for unif dist, a is low interval, b is high interval
def datagen(distribution,a,b,samples):

    if distribution == 'norm':
        return np.random.normal(a, b, samples)
    elif distribution == 'unif':
        return np.random.uniform(a, b, samples)
    else:
        raise ValueError('A very specific bad thing happened')


# CONSTANTS
N = 100
# bins = int(N / 100)
bins = 10
a = 3
b = 5

# call function with constants
unif_array = datagen('unif', a, b, N)
norm_array = datagen('norm', a, b, N)

# create new figure
plt.figure()

# first subplot - uniform distribution
plt.subplot(2,2,1)
# plot data as blue histogram
plt.hist(unif_array, bins=bins, label='empirical raw data')
# give plot a title
plt.title('Uniform Dist')
plt.legend()

# second subplot - uniform distribution, normalised
plt.subplot(2,2,2)
# sort the array
unif_array.sort()

# plot normalised data as blue histogram
plt.hist(unif_array, bins=bins, normed=True, label='empirical raw data')
# hist_array, hist_bin_edges = np.histogram(unif_array, bins=bins, density=True)

# calc pdf of normalised uniform distribution?
pdf = stats.uniform.pdf(unif_array, loc=min(unif_array), scale=max(unif_array))
print(pdf)
pdf_theoretical = np.ones(np.size(unif_array)) * (1/(b-a))
# plot pdf as a red line
plt.plot(unif_array, pdf, 'r-', label='empirical pdf')
# plot theoretical pdf as a green line
plt.plot(unif_array, pdf_theoretical, 'g-', label='theoretical pdf')
# give plot a title
plt.title('Uniform Dist Normed')
plt.legend()


# third subplot - normal distribution
plt.subplot(2,2,3)
# plot data as blue histogram
plt.hist(norm_array, bins=bins, label='empirical raw data')
# give plot a title
plt.title('Normal Dist')
plt.legend()

# fourth subplot - normal distribution, normalised
plt.subplot(2,2,4)
# sort the array
norm_array.sort()
# get probability density function for the array
pdf = stats.norm.pdf(norm_array, np.mean(norm_array), np.std(norm_array))
pdf_theoretical = stats.norm.pdf(norm_array, 3, 5)

# plot pdf as a red line
plt.plot(norm_array, pdf, 'r-', label='empirical pdf')
plt.plot(norm_array, pdf_theoretical, 'g-', label='theoretical pdf')
# plot normalised data as blue histogram
plt.hist(norm_array, bins=bins, normed=True, label='empirical raw data')
# give plot a title
plt.title('Normal Dist Normed')
plt.legend()

plt.show()