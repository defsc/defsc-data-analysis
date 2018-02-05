import matplotlib.pyplot as plt

def sign(value):
    if value > 0:
        return 1
    elif value < 0:
        return -1
    else:
        return 0

class DaylyMeasurements:
    def __init__(self, ini_line_temperature, ini_line_pollution):
        splited_temperature = ini_line_temperature[:-1].split(';')
        self.day = splited_temperature[0]
        self.time = splited_temperature[1]
        self.hour = splited_temperature[1].split(":")[0]
        self.temperature = float(splited_temperature[2])

        splited_pollution = ini_line_pollution[:-1].split(';')
        self.pollution = float(splited_pollution[2])

    def __str__(self):
        return str(self.day) + " " + str(self.time) + " " + str(self.hour) + \
            " " + str(self.temperature) + " " + str(self.pollution)

    def __repr__(self):
        return self.__str__()

    def get_change_sign(self, previous_measurement):
        previous_temperature = previous_measurement.temperature
        previous_pollution = previous_measurement.pollution

        self.pollution_change = sign(self.pollution - previous_pollution)
        self.temperature_change = sign(self.temperature - previous_temperature)

    def get_agreement(self):
        if self.pollution_change == self.temperature_change and self.pollution_change != 0:
            self.agreement = 1
        elif (self.pollution_change == 1 and self.temperature_change == -1) or \
            (self.pollution_change == -1 and self.temperature_change == 1):
            self.agreement = -1
        else:
            self.agreement = 0

class MeasurementsManager:
    def __init__(self, temperature_path, pollution_path):
        with open(temperature_path, "r") as f:
            self.temperatures = f.readlines()
        with open(pollution_path, "r") as f:
            self.pollutions = f.readlines()

        self.hourly_measurements = [DaylyMeasurements(temperature_line,pollution_line) \
            for pollution_line, temperature_line in zip(self.pollutions, self.temperatures)]

    def get_agreement_for_measurements(self):
        for idx in range(1, len(self.hourly_measurements)):
            self.hourly_measurements[idx].get_change_sign(self.hourly_measurements[idx-1])
            self.hourly_measurements[idx].get_agreement()

    def reduce_agreements_for_daily(self):
        self.agreements_in_day = {}
        for measurement in self.hourly_measurements[1:]:
            if measurement.hour in self.agreements_in_day:
                plus,minus,zero = self.agreements_in_day[measurement.hour]
            else:
                plus,minus,zero = (0,0,0)

            if measurement.agreement == 1:
                plus += 1
            elif measurement.agreement == -1:
                minus += 1
            else:
                zero += 1

            self.agreements_in_day[measurement.hour] = (plus,minus,zero)

        for hour in self.agreements_in_day:
            plus,minus,zero = self.agreements_in_day[hour]
            total = int(plus) + int(minus) + int(zero)
            self.agreements_in_day[hour] = (float(plus)/total,float(minus)/total,float(zero)/total)

    def plot_daily_measurements(self, name):
        x = sorted(self.agreements_in_day)
        pluses_list = [self.agreements_in_day[hour][0] for hour in x]
        minuses_list = [self.agreements_in_day[hour][1] for hour in x]
        zeros_list = [self.agreements_in_day[hour][2] for hour in x]

        fig = plt.figure()
        ax1 = fig.add_subplot(111)

        ax1.scatter(x, pluses_list,c='g',label="Rise->Rise | Down->Down")
        ax1.scatter(x, minuses_list,c='r',label="Rise->Down | Down->Rise")
        ax1.scatter(x, zeros_list,c='b',label="Others")
        plt.legend(loc='upper center', bbox_to_anchor=(0.5,-0.05),fancybox=True, shadow=True,ncol=3)
        plt.title("Rozklad kierunku asocjacji w zaleznosci od pory dnia powyzej " + str(name))
        plt.ylabel("Rozklad")
        plt.ylim(0.0, 1.0)
        plt.savefig("./plots/" + str(name) + ".png")

    # def filter_days_not_fitting(self, from, to):
    #     pass

    def filter_days_too_low(self, low_limit):
        days_to_remove = {measurement.day for measurement in self.hourly_measurements}

        for measurement in self.hourly_measurements:
            if measurement.pollution >= low_limit and measurement.day in days_to_remove:
                days_to_remove.remove(measurement.day)

        self.hourly_measurements = [measurement for measurement in self.hourly_measurements \
            if not measurement.day in days_to_remove]


if __name__ == "__main__":
    for i in range(0, 115):
        manager = MeasurementsManager("temperature.csv", "zanieczyszczenie.csv")
        manager.get_agreement_for_measurements()
        manager.filter_days_too_low(i)
        manager.reduce_agreements_for_daily()
        manager.plot_daily_measurements(i)
