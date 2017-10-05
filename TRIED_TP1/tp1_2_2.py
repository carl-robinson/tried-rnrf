import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats


# returns an np array of random numbers for the given distribution and parameters
# for norm dist, a is mean, b is stddev
# for unif dist, a is low interval, b is high interval
def datagen_reps(distribution, a, b, samples, repetitions):
    result = []
    if distribution == 'norm':
        for i in np.arange(repetitions):
            result.append(np.random.normal(a, b, samples))
    elif distribution == 'unif':
        for i in np.arange(repetitions):
            result.append(np.random.uniform(a, b, samples))
    else:
        raise ValueError('A very specific bad thing happened')

    return np.array(result)


# CONSTANTS
a = 3
b = 5
# N_set = [500, 10000]
N_set = [5, 50, 500]
# T_set = [1000, 5000]
T_set = [10, 200]
bins = 10
range_low = 3.2
range_high = 4.8

list_of_outputs = []

for T in T_set:
    # create new figure
    plt.figure()
    i = 1
    for N in N_set:
        # create subplot
        plt.subplot(2, 3, i)
        i += 3
        # call reps functions with constants - get T rows by N cols array of random numbers
        unif_array_reps = datagen_reps('unif', a, b, N, T)
        # calculate mean average for each of T arrays - get T rows by 1 col array of means
        unif_array_reps_ave = unif_array_reps.mean(axis=1)
        # sort the array of T averages
        unif_array_reps_ave.sort() # there is max 200 averages per array
        # save the means for later calculating and plotting the mean of means and the variance
        list_of_outputs.append(unif_array_reps_ave.tolist())

        # plot normalised data as blue histogram
        # Remember that normed=True doesn't mean that the sum of the value at each bar will be unity,
        # but rather than the integral over the bars is unity. bins 15
        # histogram_array, histogram_bin_edges = np.histogram(unif_array_reps_ave, range=(range_low, range_high), bins=bins, density=True)
        histogram_array, histogram_bin_edges = np.histogram(unif_array_reps_ave, range=(range_low, range_high), bins=bins)

        # print('histogram_array')
        # print(histogram_array)
        # print('histogram_bin_edges')
        # print(histogram_bin_edges)

        # histogram_array = histogram_array / sum(histogram_array)
        # normalise by dividing histogram values by (width of bin * number of elements in histogram)
        histogram_array = histogram_array / (((range_high - range_low) / bins) * T)

        # print('histogram_array')
        # print(histogram_array)
        # print(sum(histogram_array)*0.16)

        plt.bar(left=histogram_bin_edges[0:bins],
                height=histogram_array,
                width=((range_high - range_low) / bins),
                orientation='vertical',
                linewidth=1,
                edgecolor='k')
        # hist_array, hist_bin_edges, hist_silent = plt.hist(unif_array_reps_ave, range=(3.2, 4.8), bins=15, normed=True)
        # set the y axis range manually
        plt.ylim(0.0, 4.5)
        # set the graph titles
        plt.title('N=' + str(N) + ', T=' + str(T))
        plt.suptitle('Bar width = ' + str(round(((range_high - range_low) / bins),2)))

        # plot pdf graph
        plt.subplot(2, 3, i)
        i -= 2

        # call probability density function - get T rows array of pdf values
        pdf = stats.norm.pdf(unif_array_reps_ave, np.mean(unif_array_reps_ave), np.std(unif_array_reps_ave))

        # normalise pdf by dividing by largest pdf value
        # pdf = pdf / pdf.max()
        # pdf = pdf / pdf.sum()

        # DEBUG
        # print('pdf')
        # print(pdf)
        # print('unif_array_reps_ave')
        # print(unif_array_reps_ave)

        # plot pdf as a red line
        plt.plot(unif_array_reps_ave, pdf, 'r-')

        # set the x axis range manually (no need to set y axis as it's normalised already)
        plt.xlim(2.8, 5.2)
        # set the graph titles
        plt.title('N=' + str(N) + ', T=' + str(T))


        # print out stats
        # print('##################')
        # print('N = ' + str(N) + ', T= ' + str(T))
        # print('X_bar empirical = ' + str(np.mean(unif_array_reps_ave)))
        # print('X_bar theoretical = ' + str((a + b) / 2))
        # print('X_bar difference = ' + str(np.mean(unif_array_reps_ave) - ((a + b) / 2)))
        # print('var(X) empirical = ' + str(np.var(unif_array_reps_ave)))
        # # this should be a function of N (or T?) based on the pdf
        # print('var(X) theoretical = ' + str(np.var(pdf)))
        # print('var(X) difference = ' + str(np.var(unif_array_reps_ave) - np.var(pdf)))

# show the graphs
plt.show()

# DEBUG
# print('hist_array')
# print(hist_array)
# print(sum(hist_array))
# print('hist_bin_edges')
# print(hist_bin_edges)

varlist = []
# print(len(list_of_outputs))
for i in list_of_outputs:
    j = np.array([i])
    varlist.append(j.var())

plt.figure()
plt.plot(N_set, varlist[0:3], 'b-', label='T=10')
plt.plot(N_set, varlist[3:6], 'g-', label='T=200')

# set the graph titles
plt.title('Variance as a function of N')
plt.xlabel('N')
plt.ylabel('Variance')
plt.legend()
plt.show()