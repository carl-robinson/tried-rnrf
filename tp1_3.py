import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import math

# open the data file
input_file = open('pluie.txt', 'r')

# read in lines into a list of lists, skipping every other line as these are blank
input_data = []
for line in input_file:
    # (year, zone1, zone2, zone3, zone4) = line.strip().split('\t')
    input_data.append(line.split())
    # print(line)
    input_file.readline()

# make table, column for each zone, with la moyenne, l’écart type, le minimum, le maximum et l’étendue
input_data_np = np.array(input_data)

# get years data as list of ints
zone_np = input_data_np[:, 0].astype(np.int)
years = zone_np.tolist()
print(years)

# get column data for each of the four zones (cols 2 to 5 in txt file)
means = []
standard_devs = []
mins = []
maxs = []
ranges = []
for col in np.arange(1, 5):
    zone_np = input_data_np[:, col].astype(np.float)

    means.append(np.mean(zone_np))

    standard_devs.append(np.std(zone_np))

    zone_min = np.min(zone_np)
    mins.append(zone_min)

    zone_max = np.max(zone_np)
    maxs.append(zone_max)

    ranges.append(zone_max - zone_min)

print(means)
print(standard_devs)
print(mins)
print(maxs)
print(ranges)

# create new figure
plt.figure()
# plot the lines
plt.plot(years, input_data_np[:, 1].astype(np.float), 'r-', label='Zone 1')
plt.plot(years, input_data_np[:, 2].astype(np.float), 'g-', label='Zone 2')
plt.plot(years, input_data_np[:, 3].astype(np.float), 'b-', label='Zone 3')
plt.plot(years, input_data_np[:, 4].astype(np.float), 'm-', label='Zone 4')
# set the axis ranges manually
# plt.ylim(0.0, 2.5)
# plt.xlim(3.4, 4.6)
# set the graph titles
plt.title('Rainfall in Senegal, from 1950 to 2000')
plt.xlabel('Year')
plt.ylabel('Rainfall in mm')
plt.legend()
# plt.suptitle('Convergence of empirical density on theoretical density')
plt.show()

# another 4 graphs, one for each one - plot raw rainfall, average rainfall horiz, average+1std, average-1std
for col in np.arange(1, 5):
    # create new figure
    plt.figure()
    # plot the lines
    plt.plot(years, input_data_np[:, col].astype(np.float), 'b-', label='Rainfall')
    plt.plot(years, np.ones(np.size(years)) * means[col - 1], 'g-', label='Mean Rainfall')
    plt.plot(years, np.ones(np.size(years)) * (means[col - 1] + standard_devs[col - 1]), 'r^', label='One std dev above mean')
    plt.plot(years, np.ones(np.size(years)) * (means[col - 1] - standard_devs[col - 1]), 'rv', label='One std dev below mean')
    # set the axis ranges manually
    # plt.ylim(0.0, 2.5)
    # plt.xlim(3.4, 4.6)
    # set the graph titles
    plt.title('Rainfall in Zone ' + str(col) + ', from 1950 to 2000')
    plt.xlabel('Year')
    plt.ylabel('Rainfall in mm')
    plt.legend()
    plt.show()

# get % of points inside 1 std dev of mean
# for each zone
for col in np.arange(1, 5):
    # get rainfall for the zone
    data = input_data_np[:, col].astype(np.float)
    # set counter
    counter = 0
    # set max
    maxlimit = means[col - 1] + standard_devs[col - 1]
    # set min
    minlimit = means[col - 1] - standard_devs[col - 1]
    # if data is less than max but more than min, increment counter
    for i in data:
        if i < maxlimit:
            if i > minlimit:
                counter += 1
    print('zone = ' + str(col))
    print('counter = ' + str(counter))
    percentage = (counter / 51) * 100
    print('percentage = ' + str(percentage))

# get quartiles and interquartile range
# for each zone
for col in np.arange(1, 5):
    # get rainfall for the zone
    data = input_data_np[:, col].astype(np.float)
    pcent25 = np.percentile(data, 25)
    pcent50 = np.percentile(data, 50)
    pcent75 = np.percentile(data, 75)
    pcent_range = pcent75 - pcent25

    print('zone = ' + str(col))
    print('25th percentile = ' + str(pcent25))
    print('50th percentile = ' + str(pcent50))
    print('75th percentile = ' + str(pcent75))
    print('range = ' + str(pcent_range))

# create new figure
plt.figure()
# for each zone
# for col in np.arange(1, 5):
    # get rainfall for the zone
    # data = input_data_np[:, col].astype(np.float)
data = input_data_np[:, 1:5].astype(np.float)
zone_lbls = ['Zone 1', 'Zone 2', 'Zone 3', 'Zone 4']
bp = plt.boxplot(data, labels=zone_lbls, notch=True)
plt.title('Rainfall in Senegal from 1950 to 2000')
plt.ylabel('Rainfall in mm')
# plt.legend()
plt.show()