import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.cm as cm
import matplotlib.colors as mcolors

SENSITIVITY_COEFFICIENT = [.25] #C
BASE_SIGMA = [100] #sigma
POPULATION = 1
PRODUCTIVITY_DENSITY = [100] #pi
MATCHING_RATE_CONSTANT = [5E-4] #K
INIT_VACANCIES = [100] #V0
INIT_UNEMPLOYMENT = 0
SEPARATION_RATE = .05 #s
SURPLUS_XI = 0.05 #xi

STEPS = 1000
dt = 1

TIME = range(int(STEPS/dt))


sigmas = np.linspace(250, 1900, 20)
c_values = [0.25, 0.5,.75, 1]
k_values_sweep = np.logspace(np.log10(5e-5), np.log10(5e-4), 4)



#read in GDP data from dummy_gdp.pkl
GDP = pd.read_pickle('dummy_demand.pkl')

# firm class definition that has data on demand, vacancies, and employment
class Firm:

    def plot_demand(self):
        plt.plot(self.signal,label = "signal")
        plt.plot(self.employment_demand,label = "target employment")
        plt.legend()
        plt.savefig("demand.png")
        plt.close()

    def update_vacancies(self,t,unemployment):
        self.matching_function[t] = self.matching_rate_constant*unemployment
        # print(" VACANCY UPDATE")
        # print("Personnel demand:     ",self.employment_demand[t])
        # print("Employment:           ",self.employment[t])
        # print("Productivity Diff:    ",self.employment_demand[t]-self.employment[t])
        # print("Sep Rate:             ", self.separation_rate[t])
        # print("Matching Function:    ", self.matching_function[t])
        # print("Exp Separation:       ", self.employment[t]*self.separation_rate[t]/self.matching_function[t])
        v_t = self.employment_demand[t]-self.employment[t]+self.employment[t]*self.separation_rate[t]/self.matching_function[t]

        if v_t < 0:
            self.vacancies[t] = 0
        else:
            self.vacancies[t] = v_t

        # print("VACANCIES: ",self.vacancies[t])
    
    def update_employment(self,t):
        demand_update = self.employment_demand[t]-self.employment[t]

        if demand_update > 0:
            demand_update = 0
        # print("EMPLOYMENT UPDATE")
        # print("Hires:             ",self.vacancies[t]*self.matching_function[t])
        # print("Separations:       ", self.separation_rate[t]*self.employment[t])
        # print("Demand Update:     ", demand_update)
        # print("\n\nOld Employment:    ", self.employment[t])
        update = self.vacancies[t]*self.matching_function[t]-self.employment[t]*self.separation_rate[t]+demand_update

        # print("Net Change: ",update)
        #FIRE EVERYONE!
        if self.employment[t] + update < 0:
            self.employment[t+1] = 0
        else:
            self.employment[t+1] = self.employment[t]+update

        # print("New Employment:    ", self.employment[t+1])

        # Calculate efficiency for the current step
        target = self.employment_demand[t]
        actual = self.employment[t+1]

        if target <= 0:
            self.efficiency[t+1] = 1.0

        else:  # target > actual, inefficiency due to hiring friction
            self.efficiency[t+1] = (actual-np.abs(target-actual)) / actual



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
        self.efficiency = [0 for _ in TIME]
        self.time = None

        self.productivity_density = init_productivity_density
        self.employment_demand = self.computeDemand(signal)
        self.matching_rate_constant = matching_rate_constant
        # self.matching_rate_constant = (POPULATION/500)*(1-SURPLUS_XI)*SEPARATION_RATE
        self.target = self.set_target()
        
    def set_time(self, time):
        self.time = time

def compute_loop_area(x, y):
    """
    Calculates the area of a polygon using the Shoelace formula.
    Assumes the x and y arrays represent the vertices of a closed loop.
    """
    import numpy as np
    x = np.asarray(x)
    y = np.asarray(y)
    return 0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

def run_market(firms, population, init_unemployment):
    unemployment = [init_unemployment for _ in TIME]
    for t in TIME[:-1]:
        for firm in firms:
            firm.update_vacancies(t,unemployment[t])
            firm.update_employment(t)
        
        unemployment[t+1] = population - np.sum([firm.employment[t+1] for firm in firms])
    
    # One last vacancy update for the final timestep
    final_t = TIME[-1]
    for firm in firms:
        firm.update_vacancies(final_t, unemployment[final_t])

    return unemployment,firms
        
def compute_rates(firms, economy_name, population=None, plot=True):
    vacancies = [np.sum([firm.vacancies[t] for firm in firms]) for t in TIME]
    employment = [np.sum([firm.employment[t] for firm in firms]) for t in TIME]

    # Use provided population or fall back to global if not provided
    pop = population if population is not None else POPULATION
    
    vacancy_rate = [vacancies[t]/(vacancies[t]+employment[t]) if (vacancies[t]+employment[t]) > 0 else 0 for t in TIME]
    unemployment_rate = [1-employment[t]/pop for t in TIME]

    if plot:
        plt.plot(vacancy_rate,unemployment_rate,label = "signal")
        plt.xlabel("vacancy rate")
        plt.ylabel("unemployment rate")
        plt.axhline(y=SURPLUS_XI, color='red', linestyle='--')
        plt.legend()
        plt.savefig("beveridge"+economy_name+".png")
        plt.close()

    return vacancy_rate,unemployment_rate


def plot_multi_employment(firms, unemployment, time, population, economy_name):
    """
    Plots the employment of multiple individual firms and the aggregate unemployment.
    """
    plt.figure(figsize=(10, 6))
    
    # Plot employment for each firm
    colors = cm.get_cmap('tab10', len(firms))
    for i, firm in enumerate(firms):
        plt.plot(time, firm.employment, label=f"Employment Firm {i+1} (C={firm.sensitivity_coefficient})", color=colors(i), linestyle='-')
        plt.plot(time, firm.vacancies, label=f"Vacancies Firm {i+1}", color=colors(i), linestyle='--')

    # Plot aggregate unemployment
    plt.plot(time, unemployment, label="Aggregate Unemployment", color='black', linestyle=':')

    # --- Population Conservation Check ---
    # Plot theoretical constant population
    plt.axhline(y=population, color='gray', linestyle='--', label='Total Population (Theoretical)')

    # Calculate and plot the actual total people in the system
    total_employment = np.sum([np.array(f.employment) for f in firms], axis=0)
    total_people_actual = total_employment + np.array(unemployment)
    plt.plot(time, total_people_actual, color='blue', linestyle='-.', label='Total Population (Actual)')

    # Add theoretical minimum unemployment line
    plt.axhline(y=SURPLUS_XI * population, color='red', linestyle='--', label=f'$\\xi L$ = {SURPLUS_XI*population:.0f}')


    plt.xlabel("Time")
    plt.ylabel("Number of People")
    plt.title(f"Employment Dynamics ({economy_name})")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"employment_{economy_name}.png")
    plt.close()
    print(f"Saved multi-firm employment plot to employment_{economy_name}.png")


def initialize_economy(demand_signal, firm_weights_k, base_sigma, surplus_xi, productivity_density, sensitivity_coefficients):
    """
    Initializes the economy's structural parameters using a non-circular method.

    This function calculates firm sizes (sigma_i) and the total labor force (L)
    based on a base firm size, firm weights, and the peak of the demand signal.

    Args:
        demand_signal (pd.Series): The time series of the economic signal (G(t)).
        firm_weights_k (list or np.array): A vector of weights k_i for each firm.
        base_sigma (float): The base firm size (sigma), an anchor for the economy's scale.
        surplus_xi (float): The desired labor surplus fraction (e.g., 0.05 for 5%).
        productivity_density (float): The productivity per worker (pi).
        sensitivity_coefficients (list or np.array): The firm sensitivities to the economic signal (C_i).
                                       (Assumed to be the same for all firms for now).

    Returns:
        tuple: A tuple containing:
            - sigmas (np.array): An array of sigma values for each firm.
            - population (float): The total labor force L.
            - num_firms (int): The number of firms N.
    """
    num_firms_N = len(firm_weights_k)
    k = np.asarray(firm_weights_k)
    C = np.asarray(sensitivity_coefficients)

    # --- Step 1: Calculate firm sizes (sigma_i) from the base sigma ---
    sigmas = k * base_sigma

    # --- Step 2: Calculate peak target employment ---
    # Find the peak of the economic signal G(t)
    g_max = demand_signal.max()

    # Calculate the target employment for each firm at the peak signal
    # target_employment_i = sigma_i * (1 + C_i * g_max)
    peak_target_employment_per_firm = sigmas * (1 + C * g_max)
    total_peak_target_employment = np.sum(peak_target_employment_per_firm)

    # --- Step 3: Calculate the total labor force L ---
    # At peak demand, employment = E_max and we want ξ people unemployed
    # So: (L - E_max) / L = ξ, which implies L = E_max / (1 - ξ)
    population_L = total_peak_target_employment / (1 - surplus_xi)
    print("Population: ", population_L)
    print("Sigmas: ", sigmas)
    print("Num Firms: ", num_firms_N)
    return sigmas, population_L, num_firms_N

def run_k_sweep_for_c(c_value):
    """
    Runs a parameter sweep for K for a given C value and returns the results.
    """
    k_values_sweep = np.logspace(np.log10(5e-5), np.log10(5e-4), 20)
    areas = []
    beveridge_curves = []
    
    print(f"Running K sweep for C = {c_value}...")
    sigma_value = BASE_SIGMA[0]

    # --- Initialize economy for this sweep (doesn't depend on K) ---
    firm_weights_k = [1]
    current_sigmas, current_population, _ = initialize_economy(
        GDP['gdp_sine'], firm_weights_k, sigma_value, SURPLUS_XI, 
        PRODUCTIVITY_DENSITY[0], c_value
    )
    # Calculate initial employment based on the demand at t=0
    initial_signal = GDP['gdp_sine'].iloc[0] if 'gdp_sine' in GDP.columns else 0
    init_employment = current_sigmas[0] * (1 + c_value * initial_signal)
    init_unemployment = current_population - init_employment

    for k in k_values_sweep:
        firm = Firm(GDP['gdp_sine'], current_sigmas[0], PRODUCTIVITY_DENSITY[0], init_employment, 0, k, c_value)
        firm.set_time(TIME)

        unemployment, firms = run_market([firm], current_population, init_unemployment)
        vacancy_rate, unemployment_rate = compute_rates(firms, "single", current_population, plot=False)

        loop_vacancies = vacancy_rate[400:800]
        loop_unemployment = unemployment_rate[400:800]
        area = compute_loop_area(loop_vacancies, loop_unemployment)
        areas.append(area)

        if area <= 0.5:
            beveridge_curves.append({'k': k, 'vr': vacancy_rate, 'ur': unemployment_rate, 'efficiency': firms[0].efficiency})
            
    return k_values_sweep, areas, beveridge_curves

def run_nested_sweep_k_major():
    """
    Runs the major C sweep for several minor K sweeps and plots the results.
    """
    all_results = []
    for c in c_values:
        k_vals, areas, curves = run_k_sweep_for_c(c)
        all_results.append({'c': c, 'k_values': k_vals, 'areas': areas, 'curves': curves})

    # Plot Area vs. K for all C values
    plt.figure(figsize=(10, 6))
    for result in all_results:
        plt.plot(result['k_values'], result['areas'], marker='o', linestyle='-', label=f"C = {result['c']}")
    plt.title("Area of Beveridge Curve vs. K for different C")
    plt.xlabel("K (Matching Rate Constant)")
    plt.ylabel("Area of Loop")
    plt.xscale('log')
    plt.legend()
    plt.grid(True)
    plt.savefig("beveridge_area_vs_k_for_c.png")
    plt.close()
    print("Saved beveridge_area_vs_k_for_c.png")

    # Plot Beveridge curves in a 2x2 subplot grid
    fig, axs = plt.subplots(2, 2, figsize=(15, 12), constrained_layout=True)
    fig.suptitle('Beveridge Curves: K sweeps for different C values', fontsize=16)

    norm = mcolors.Normalize(vmin=5e-5, vmax=5e-4)
    cmap = plt.get_cmap('inferno')

    for i, (ax, result) in enumerate(zip(axs.flat, all_results)):
        ax.set_title(f"C = {result['c']}")
        for curve in result['curves'][::2]:
            ax.plot(curve['ur'][400:800], curve['vr'][400:800], color=cmap(norm(curve['k'])))
        ax.set_ylabel("Vacancy Rate")
        ax.set_xlabel("Unemployment Rate")
        ax.grid(True)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axs, shrink=0.6, location='bottom')
    cbar.set_label('K (Matching Rate Constant)')

    plt.savefig("beveridge_curves_k_major_sweep.png")
    plt.close(fig)
    print("Saved beveridge_curves_k_major_sweep.png")

    # Plot Efficiency vs. Time in a 2x2 subplot grid
    fig, axs = plt.subplots(2, 2, figsize=(15, 12), constrained_layout=True)
    fig.suptitle('Firm Efficiency: K sweeps for different C values', fontsize=16)

    for i, (ax, result) in enumerate(zip(axs.flat, all_results)):
        ax.set_title(f"C = {result['c']}")
        for curve in result['curves'][::2]:
            ax.plot(TIME[5:],curve['efficiency'][5:], color=cmap(norm(curve['k'])))
        ax.set_ylabel("Efficiency")
        ax.set_xlabel("Time")
        ax.grid(True)
        ax.set_ylim(.85, 1.1)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axs, shrink=0.6, location='bottom')
    cbar.set_label('K (Matching Rate Constant)')

    plt.savefig("efficiency_k_major.png")
    plt.close(fig)
    print("Saved efficiency_k_major.png")

def run_single_timeseries():
    """
    Runs a single simulation with the default parameters and plots the primary time series.
    """
    print("Running single simulation to generate time series plots...")

    # --- Initialize economy using constraints ---
    firm_weights_k = [1]
    base_sigma = BASE_SIGMA[0]
    sigmas, population, num_firms = initialize_economy(
        GDP['gdp_sine'], 
        firm_weights_k, 
        base_sigma, 
        SURPLUS_XI, 
        PRODUCTIVITY_DENSITY[0], 
        SENSITIVITY_COEFFICIENT[0]
    )

    # Calculate initial employment based on the demand at t=0
    initial_signal = GDP['gdp_sine'].iloc[0] if 'gdp_sine' in GDP.columns else 0
    init_employment = sigmas[0] * (1 + SENSITIVITY_COEFFICIENT[0] * initial_signal) 
    init_unemployment = population - init_employment
    
    # Update global INIT_UNEMPLOYMENT to be consistent with the new population
    global INIT_UNEMPLOYMENT; global POPULATION
    INIT_UNEMPLOYMENT = init_unemployment
    POPULATION = population
    
    # Initialize a single firm with the new constrained parameters
    firm = Firm(GDP['gdp_sine'], sigmas[0], PRODUCTIVITY_DENSITY[0], init_employment, 0, MATCHING_RATE_CONSTANT[0], SENSITIVITY_COEFFICIENT[0])
    firm.set_time(TIME)

    # Run the market simulation
    unemployment, firms = run_market([firm], population, init_unemployment)

    # Generate and save the plots
    plot_multi_employment(firms, unemployment, TIME, population, "single_firm")
    vacancy_rate, unemployment_rate = compute_rates(firms, "single", population, plot=True)
    
    print("\nGenerated single-run plots:")
    print("- demand.png")
    print("- employment.png")
    print("- beveridge.png")

    return vacancy_rate, unemployment_rate

def run_double_timeseries():
    """
    Runs a simulation with two firms competing in the same labor market.
    """
    print("Running dual-firm simulation...")

    # --- Define parameters for two distinct firms ---
    # Firm 1: Larger, less sensitive to market changes
    # Firm 2: Smaller, more sensitive (volatile)
    firm_weights_k = [0.7, 0.3] # Firm 1 is 70% of base size, Firm 2 is 30%
    sensitivity_coefficients = [0.25, 0.75] # C values for each firm

    # --- Initialize economy using constraints for the two-firm system ---
    base_sigma = BASE_SIGMA[0]
    sigmas, population, num_firms = initialize_economy(
        GDP['gdp_sine'], 
        firm_weights_k, 
        base_sigma, 
        SURPLUS_XI, 
        PRODUCTIVITY_DENSITY[0], 
        sensitivity_coefficients
    )
    
    # --- Calculate initial state based on t=0 demand for both firms ---
    initial_signal = GDP['gdp_sine'].iloc[0] if 'gdp_sine' in GDP.columns else 0
    init_employment_firm1 = sigmas[0] * (1 + sensitivity_coefficients[0] * initial_signal)
    init_employment_firm2 = sigmas[1] * (1 + sensitivity_coefficients[1] * initial_signal)
    init_unemployment = population - (init_employment_firm1 + init_employment_firm2)

    # --- Create Firm instances ---
    firm1 = Firm(GDP['gdp_sine'], sigmas[0], PRODUCTIVITY_DENSITY[0], init_employment_firm1, 0, MATCHING_RATE_CONSTANT[0], sensitivity_coefficients[0])
    firm2 = Firm(GDP['gdp_sine'], sigmas[1], PRODUCTIVITY_DENSITY[0], init_employment_firm2, 0, MATCHING_RATE_CONSTANT[0], sensitivity_coefficients[1])
    
    firm1.set_time(TIME)
    firm2.set_time(TIME)
    
    # --- Run the market simulation ---
    firms = [firm1, firm2]
    unemployment, firms = run_market(firms, population, init_unemployment)

    # --- Generate and save the plots ---
    # Aggregate Beveridge Curve
    vacancy_rate, unemployment_rate = compute_rates(firms, "double", population, plot=True) 
    # Individual employment dynamics
    plot_multi_employment(firms, unemployment, TIME, population, "two_firm")

    return vacancy_rate, unemployment_rate

def run_four_firm_timeseries():
    """
    Runs a simulation with four equal-sized firms competing in the same labor market.
    """
    print("Running four-firm simulation...")

    # --- Define parameters for four distinct firms ---
    # All firms are equal size (k=0.25)
    # Two are stable (low C), two are volatile (high C)
    firm_weights_k = [0.25, 0.25, 0.25, 0.25]
    sensitivity_coefficients = [0.25, 0.25, 0.75, 0.75] 

    # --- Initialize economy ---
    base_sigma = BASE_SIGMA[0]
    sigmas, population, num_firms = initialize_economy(
        GDP['gdp_sine'], firm_weights_k, base_sigma, SURPLUS_XI, 
        PRODUCTIVITY_DENSITY[0], sensitivity_coefficients
    )
    
    # --- Calculate initial state for all firms ---
    initial_signal = GDP['gdp_sine'].iloc[0] if 'gdp_sine' in GDP.columns else 0
    
    firms = []
    total_initial_employment = 0
    for i in range(num_firms):
        init_emp = sigmas[i] * (1 + sensitivity_coefficients[i] * initial_signal)
        total_initial_employment += init_emp
        firm = Firm(GDP['gdp_sine'], sigmas[i], PRODUCTIVITY_DENSITY[0], init_emp, 0, MATCHING_RATE_CONSTANT[0], sensitivity_coefficients[i])
        firm.set_time(TIME)
        firms.append(firm)

    init_unemployment = population - total_initial_employment
    
    # --- Run the market simulation ---
    unemployment, firms = run_market(firms, population, init_unemployment)

    # --- Generate and save the plots ---
    vacancy_rate, unemployment_rate = compute_rates(firms, "four_firm", population, plot=True) 
    plot_multi_employment(firms, unemployment, TIME, population, "four_firm")

    return vacancy_rate, unemployment_rate


def plot_beveridge_comparison(curves_data):
    """
    Plots multiple Beveridge curves on the same axes for comparison.

    Args:
        curves_data (dict): A dictionary where keys are labels for the curves
                            and values are tuples of (vacancy_rate, unemployment_rate).
    """
    plt.figure(figsize=(8, 6))
    
    for label, (vr, ur) in curves_data.items():
        plt.plot(ur,vr, label=label)

    plt.xlabel("Vacancy Rate")
    plt.ylabel("Unemployment Rate")
    plt.title("Beveridge Curve Comparison")
    plt.axvline(x=SURPLUS_XI, color='red', linestyle='--', label=f'Min Unemployment Rate ($\\xi={SURPLUS_XI}$)')
    plt.legend()
    plt.grid(True)
    plt.savefig("beveridge_comparison.png")
    plt.close()
    print("Saved Beveridge curve comparison plot to beveridge_comparison.png")


def main():
    # Run simulations and get the curve data
    vr_single, ur_single = run_single_timeseries()
    vr_double, ur_double = run_double_timeseries()
    vr_four, ur_four = run_four_firm_timeseries()

    # Plot the comparison
    comparison_data = {
        "Single Firm": (vr_single, ur_single),
        "Two Firms (Aggregate)": (vr_double, ur_double),
        "Four Firms (Aggregate)": (vr_four, ur_four)
    }
    plot_beveridge_comparison(comparison_data)

    # run_nested_sweep_k_major()

if __name__ == "__main__":
    main()