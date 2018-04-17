import os

import matplotlib.pyplot as plt
from pandas import *
import numpy as np

if not os.path.exists("../results"):
    os.makedirs("../results")
if not os.path.exists("../results/correlation"):
    os.makedirs("../results/correlation")

manual = False
every_for_in_automated = ['airly-pm25']

if manual:
    correlations_to_check = [['airly-pm25', 'airly-tmp'], ['airly-pm25', 'here-traffic-speed'], ['airly-pm25', 'airly-pm25'], ['airly-pm25', 'airly-hum']]
else:
    correlations_to_check = []

if __name__ == "__main__":
    directory = '../data'
    for filename in os.listdir(directory):
        # print(filename)
        if not filename in ['raw-210.csv', 'raw-218.csv']:
            continue
        if not manual:
            correlations_to_check = []

        csv = os.path.join(directory, filename)
        df = read_csv(csv, header=0, index_col=0)
        df.index = to_datetime(df.index)
        df = df.apply(lambda ts: ts.interpolate(method='nearest'))
        df = df.apply(lambda ts: ts.resample('1H').nearest())

        months = {n: g
                  for n, g in df.groupby(TimeGrouper('M'))}

        correlations = {}

        sorted_months = sorted(months.keys())

        for k in sorted_months:
            df_per_month = months[k]

            if not manual:
                correlations_to_check = []
                for main_column in every_for_in_automated:
                    for column in df_per_month.columns.values.tolist():
                        correlations_to_check.append([main_column, column])

            for pair_of_columns in correlations_to_check:
                correlation = df_per_month[pair_of_columns[0]].corr(df_per_month[pair_of_columns[1]])
                if np.isnan(correlation):
                    correlation = 0
                if (pair_of_columns[0], pair_of_columns[1]) not in correlations:
                    correlations[(pair_of_columns[0], pair_of_columns[1])] = [correlation]
                else:
                    correlations[(pair_of_columns[0], pair_of_columns[1])].append(correlation)

        months_labels = []
        for month_sample in sorted_months:
            months_labels.append(str(month_sample.year) + "-" + str(month_sample.month))

        for sources_for_correlation, correlations_list in correlations.items():
            fix, ax = plt.subplots()
            plt.bar(range(len(correlations_list)), correlations_list)
            plt.xticks(range(len(correlations_list)), months_labels)
            title = 'Correlation of ' + sources_for_correlation[0] + ' and ' + sources_for_correlation[1] + ' distribution\n' + filename
            ax.set_title(title)
            plt.savefig("../results/correlation/" + filename[:-4] + "-" + sources_for_correlation[0] + "-" + sources_for_correlation[1] + ".png")
            plt.close()