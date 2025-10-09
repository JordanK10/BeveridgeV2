import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

SENSITIVITY_COEFFICIENT = [.5] #C
FIRM = [1000] #sigma
PRODUCTIVITY_DENSITY = [100] #pi
MATCHING_RATE_CONSTANT = [5E-6] #K
INIT_EMPLOYMENT = [100] #E0
INIT_VACANCIES = [100] #V0
INIT_UNEMPLOYMENT = 10000
SEPARATION_RATE = .05 #s

POPULATION = np.sum(INIT_EMPLOYMENT)+INIT_UNEMPLOYMENT

STEPS = 500
dt = 1

TIME = range(int(STEPS/dt))


#read in GDP data from dummy_gdp.pkl
GDP = pd.read_pickle('dummy_demand.pkl')

# firm class definition that has data on demand, vacancies, and employment
class Firm:

    def plot_demand(self):
        plt.plot(self.signal,label = "signal")
        plt.plot(self.employment_demand,label = "demand density")
        plt.legend()
        plt.savefig("demand.png")
        plt.close()

    def plot_employment(self, unemployment):
        plt.plot(self.time,np.log(self.employment),label = "employment")
        plt.plot(self.time,np.log(unemployment),label = "unemployment")
        plt.plot(self.time,np.log(self.vacancies),label = "vacancies")
        plt.legend()
        # plt.ylim(top = np.log(POPULATION),bottom = 0)
        plt.savefig("employment.png")
        plt.close()

    def update_vacancies(self,t,unemployment):
        self.matching_function[t] = self.matching_rate_constant*unemployment
        print(" VACANCY UPDATE")
        print("Personnel demand:     ",self.employment_demand[t])
        print("Employment:           ",self.employment[t])
        print("Productivity Diff:    ",self.employment_demand[t]-self.employment[t])
        print("Sep Rate:             ", self.separation_rate[t])
        print("Matching Function:    ", self.matching_function[t])
        print("Exp Separation:       ", self.employment[t]*self.separation_rate[t]/self.matching_function[t])
        v_t = self.employment_demand[t]-self.employment[t]+self.employment[t]*self.separation_rate[t]/self.matching_function[t]

        if v_t < 0:
            self.vacancies[t] = 0
        else:
            self.vacancies[t] = v_t

        print("VACANCIES: ",self.vacancies[t])
    
    def update_employment(self,t):
        demand_update = self.employment_demand[t]-self.employment[t]

        if demand_update > 0:
            demand_update = 0
        print("EMPLOYMENT UPDATE")
        print("Hires:             ",self.vacancies[t]*self.matching_function[t])
        print("Separations:       ", self.separation_rate[t]*self.employment[t])
        print("Demand Update:      ", demand_update)
        print("\n\nOld Employment:    ", self.employment[t])
        update = self.vacancies[t]*self.matching_function[t]-self.employment[t]*self.separation_rate[t]-demand_update

        print("Net Change: ",update)
        #FIRE EVERYONE!
        if self.employment[t] + update < 0:
            self.employment[t+1] = 0
        else:
            self.employment[t+1] = self.employment[t]+update

        print("New Employment:    ", self.employment[t+1])



    def computeDemand(self, signal):
        self.signal = signal
        return self.firm_size*(1+self.sensitivity_coefficient*self.signal)
    
    def set_target(self):
        return self.employment_demand
    
    def __init__(self, signal,init_size,init_productivity_density,init_employment,init_vacancies,matching_rate_constant,sensitivity_coefficient):
        self.sensitivity_coefficient = sensitivity_coefficient
        self.firm_size = init_size
        self.employment = [init_employment for _ in TIME]
        self.vacancies = [init_vacancies for _ in TIME]
        self.matching_function = [0 for _ in TIME]
        self.separation_rate = [SEPARATION_RATE for _ in TIME]
        self.time = None

        self.productivity_density = init_productivity_density
        self.matching_rate_constant = matching_rate_constant
        self.demand = self.computeDemand(signal)
        self.employment_demand = self.demand/self.productivity_density
        self.target = self.set_target()

        #Initialize matching function
        

        self.employment_rate=[]
        


    def set_time(self, time):
        self.time = time

    def plot_handler(self, plot_list, unemployment):

        for plot in plot_list:
            if plot == "demand":
                self.plot_demand()

            if plot == "employment":
                self.plot_employment(unemployment)

def run_market(firms):
    unemployment = [INIT_UNEMPLOYMENT for _ in TIME]
    for t in TIME[:-1]:
        for firm in firms:
            firm.update_vacancies(t,unemployment[t])
            firm.update_employment(t)
        
        unemployment[t+1] = POPULATION - np.sum([firm.employment[t] for firm in firms])
    
    return unemployment,firms
        

def main():

    #initialize firm
    # firm = Firm(GDP['gdp_pos_step'],FIRM[0],PRODUCTIVITY_DENSITY[0])
    firm = Firm(GDP['gdp_neg_step'],FIRM[0],PRODUCTIVITY_DENSITY[0],INIT_EMPLOYMENT[0],INIT_VACANCIES[0],MATCHING_RATE_CONSTANT[0],SENSITIVITY_COEFFICIENT[0])
    # firm = Firm(GDP['gdp_sine'],FIRM[0],PRODUCTIVITY_DENSITY[0])

    # firm.set_time(GDP['time'])
    firm.set_time(TIME)

    unemployment,firms = run_market([firm])

    #plot demand
    plot_list = ["demand","employment"]
    for firm in firms:
        firm.plot_handler(plot_list, unemployment)

if __name__ == "__main__":
    main()