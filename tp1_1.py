import numpy as np
import matplotlib.pyplot as plt

# CONSTANTS
DATA_SIZES = [10, 20, 50, 100, 500, 2000]

# create list
list_of_arrays = []

plt.figure()

# generate array of random integers between two intervals
for x, n in enumerate(DATA_SIZES):
    list_of_arrays.append(np.random.randint(1,7,size=n))
    
    # print counts of each unique element
    uniques = np.unique(list_of_arrays[x], return_counts=True)
    print(uniques)

    # calculate percentages
    percentage_of_each_element = uniques[1] / n
    print("percentage_of_each_element = " + str(percentage_of_each_element))

    # minimum %
    min_percentage = min(percentage_of_each_element)
    print("min_percentage = " + str(min_percentage))

    # maximum %
    max_percentage = max(percentage_of_each_element)
    print("max_percentage = " + str(max_percentage))

    # range
    percentage_range = max_percentage - min_percentage
    print("percentage_range = " + str(percentage_range))

    # mean
    mean = np.mean(list_of_arrays[x])
    print("mean = " + str(mean))

    # standard deviation
    standard_deviation = np.std(list_of_arrays[x])
    print("standard_deviation = " + str(standard_deviation))

    # draw pie charts with subplot
    plt.subplot(2,3,x+1)
    plt.pie(uniques[1], labels=uniques[0], autopct='%1.1f%%', startangle=140)
    plt.title("n = " + str(n))

plt.show()

plt.figure()
# draw boxplot chart
plt.subplot(1,1,1)
plt.boxplot(list_of_arrays, labels=DATA_SIZES, notch=True)
plt.show()




