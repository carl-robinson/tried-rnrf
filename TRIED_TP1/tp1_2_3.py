import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import math


# returns an np array of random numbers for the given distribution and parameters
# for norm dist, a is mean, b is stddev
# for unif dist, a is low interval, b is high interval
def datagen_reps(distribution, a, b, samples, repetitions):
    result = []
    if distribution == 'norm':
        for _ in np.arange(repetitions):
            result.append(np.random.normal(a, b, samples))
    elif distribution == 'unif':
        for _ in np.arange(repetitions):
            result.append(np.random.uniform(a, b, samples))
    else:
        raise ValueError('A very specific bad thing happened')

    return np.array(result)


# CONSTANTS
a = 3
b = 5
N_set = [10]
T_set = [5, 10, 20, 30, 40, 50, 100, 200, 500, 1000, 2000, 5000]
bins = 10
i = 1

# create new figure
fig = plt.figure()

for T in T_set:
    for N in N_set:
        # create subplot
        plt.subplot(3, 4, i)
        i += 1
        # call reps functions with constants - get T rows by N cols array of random numbers
        unif_array_reps = datagen_reps('unif', a, b, N, T)
        # print(a, b, N, T)
        # print(unif_array_reps)
        # calculate mean average for each of T arrays - get T rows by 1 col array of means
        unif_array_reps_ave = unif_array_reps.mean(axis=1)
        # sort the array of T averages
        unif_array_reps_ave.sort() # there is max 200 averages per array
        # print(unif_array_reps_ave)
        # call probability density function - get T rows array of pdf values
        pdf_emp = stats.norm.pdf(unif_array_reps_ave, np.mean(unif_array_reps_ave), np.std(unif_array_reps_ave))
        # print(pdf_emp)
        pdf = stats.norm.pdf(unif_array_reps_ave, (a + b)/2, math.pow((b-a), 2) / 24)
        # print(pdf)
        print(np.std(unif_array_reps_ave))

        # plot emp pdf as a red line
        plt.plot(unif_array_reps_ave, pdf_emp, 'r-')
        # plot pdf as a green line
        plt.plot(unif_array_reps_ave, pdf, 'g-')

        # set the y axis range manually
        plt.ylim(0.0, 2.5)
        plt.xlim(3.4, 4.6)
        # set the graph titles
        plt.title('N=' + str(N) + ', T=' + str(T))
        plt.suptitle('Convergence of empirical density on theoretical density')

        # plt.xlabel('Random number')
        # plt.ylabel('Density')

# show the graphs
plt.show()


