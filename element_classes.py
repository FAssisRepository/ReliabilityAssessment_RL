# =================================================================================== #
# ELEMENT CLASSES
# =================================================================================== #
import pandas as pd
import numpy as np
import math

# =================================================================================== #
# Class for System
class system:

    # -----------------------------------------
    # Constructor
    def __init__(self):
        # -----------------------------------------
        # Attributes
        self.dbus = []                 # Bus data - list
        self.dcir = []                 # Circuit data - list
        self.dgstat = []               # Generating station data - list
        self.n_bus = 0                 # Number of buses of the system
        self.n_cir = 0                 # Number of circuits of the system
        self.bus_sw = []               # Swing bus (original)
        self.bus_sw_current = []       # Current swing bus (in case of unavailability of generation at the original swing bus)
        self.areas = []                # List with areas of the system
        self.peak_load = 0.0           # Peak load of the system [MW]
        # Auxiliar
        self.dbus_load = []            # Buses with load - list
        self.n_busload = 0             # Number of load buses
        self.dbus_gstat = []           # Buses with generating station - list
        self.n_gstat = 0               # Number of buses with generating stations
        self.n_branches = 0            # Number of branches (less than or equal to n_cir)
        self.n_total_generation_units = 0
        # OPF
        self.x0 = None                 # Initial solution to OPF (reoptmization)
        # Reliability
        self.dbus_reliab = []          # Region of interest buses for reliability assessment
        self.total_load = 0.00         # Total load of the system [MW]
        self.current_total_load = 0.00 # Current total load of the system [MW] - current operational state
        self.region_load = 0.00        # Total load of the interest region [mW]
        # PTDF-based
        self.D = None                  # Diagonal matrix with reatances (diag(xkm)) - dimension n_cir x n_cir
        self.A = None                  # Incidence matrix - dimension n_cir x n_bus
        self.B_bus = None              # Susceptance matrix - dimension n_bus x n_bus
        self.PTDF = None               # Power Transfer Distribution Factor (PTDF) - PTDF = -(D^-1)A'(B'^-1)
        self.PD = None                 # Bus load vector - dimension n_bus x 1
        self.Pkmmax = None             # Circuit maximum power flow vector - dimension n_cir x 1
        self.PGdispmin = None          # Bus minimum generation dispatch vector - dimension n_bus x 1
        self.PGdispmax = None          # Bus maximum generation dispatch vector - dimension n_bus x 1
        self.C = None                  # Lower bound - dimension n_cir x 1
        self.N = None                  # Upper bound - dimension n_cir x 1
        # DRL
        self.max_cir_capacity = 0.0    # Maximum capacity of the circuit with the highest capacity [MW or MVA]
        self.glmax = 0.0               # Highest longitudinal conductance value among all the circuits
        self.blmax = 0.0               # Highest longitudinal susceptance value among all the circuits
        self.max_Pmax = 0.0            # Maximum capacity of the unit with the highest capacity [MW]

    # -----------------------------------------
    # Cumput B_bus
    def compute_B_bus(self):
        self.B_bus = np.zeros((self.n_bus, self.n_bus))
        for cir in self.dcir:
            if (cir.available):
                i = cir.bF.id - 1          # Bus from 
                j = cir.bT.id - 1          # Bus to 
                b = -1/(cir.x)
                self.B_bus[i,i] += b
                self.B_bus[j,j] += b
                self.B_bus[i,j] += -b
                self.B_bus[j,i] += -b

    # -----------------------------------------
    # Cumput PD
    def compute_PD(self, _data):
        S_base = _data.simulation_set.s_base
        Pd = np.zeros((self.n_bus, 1))  
        self.PD = np.zeros((self.n_bus, 1))      
        for bus in self.dbus:
            self.PD[bus.id-1,0] = bus.PL_current / S_base

    # -----------------------------------------
    # Cumput Pkmmax
    def compute_Pkmmax(self, _data):
        cir_del = []
        S_base = _data.simulation_set.s_base
        self.Pkmmax = np.zeros((self.n_cir, 1))
        for cir in self.dcir:
            if cir.available:
                self.Pkmmax[cir.id-1, 0] = cir.cap_n / S_base
            else:
                cir_del.append(cir.id-1)
        self.Pkmmax = np.delete(self.Pkmmax, cir_del, axis = 0)

    # -----------------------------------------
    # Cumput PGdispmin and PGdispmax
    def compute_PGdispmin_PGdispmax(self):
        # Deleting rows and columns related to the sw bar
        PDl = self.PD
        PDl = np.delete(PDl, self.bus_sw.id - 1, axis = 0)
    
    # -----------------------------------------
    # String representation while debugging Python
    def __repr__(self) -> str:
        return f"Person(nbus: {self.n_bus}, ncir: {self.n_cir}, total load: {self.total_load}, region_load: {self.region_load})"
    
# =================================================================================== #
# Class for Bus
class bus:  
    
    # -----------------------------------------
    # Constructor
    def __init__(self):
        # -----------------------------------------
        # Attributes
        # Input
        self.id = 0                    # Bus id number (ascending order)
        self.number = 0                # Bus number (name)
        self.type = ''                 # Bus type (PV, PQ or SW)
        self.type_current = ''         # Bus type (PV, PQ or SW) - in case of unavailability of generation at the original swing bus
        self.V = 0.00                  # Specified voltage magnitude (pu)
        self.V_orig = 0.00             # Original specified voltage magnitude (pu)
        self.Vmin = 0.00               # Vmin - Voltage magnitude minimum limit (pu)
        self.Vmax = 0.00               # Vmax - Voltage magnitude maximum limit (pu)
        self.bshunt = 0.0              # b shunt (pu)
        self.PL = 0.00                 # Load - Active power (MW)
        self.QL = 0.00                 # Load - Reactive power (MVAr)
        self.PL_orig = 0.00            # Original load - Active power (MW)
        self.QL_orig = 0.00            # Original load - Reactive power (MVAr)
        self.PG_desp = 0.00            # Base dispatch (MW)
        self.PG_desp_orig = 0.00       # Original base dispatch (MW)
        self.area = 0                  # Area
        self.int_cost = 0.00           # Interruption cost ($/MW)
        self.region_interest = 0       # Does it participate in the region of ​​interest for reliability assessment? (1: yes; 0: no)
        self.gstat = []                # Conected generating station
        self.pos = 0                   # Input position
        self.cir_conec = []            # Connected circuits list
        # Power flow output
        self.V_FAC = 0.00 
        self.V_OPFAC = 0.00
        self.Theta_FDC = 0.00
        self.Theta_OPF = 0.00
        self.Theta_FAC   = 0.00
        self.Theta_OPFAC = 0.00
        self.Pi_FDC = 0.00
        self.Pi_OPF = 0.00
        self.Pi_FAC   = 0.00
        self.Qi_FAC   = 0.00
        self.Pi_OPFAC = 0.00        
        self.Qi_OPFAC = 0.00
        self.PG_FDC = 0.00
        self.PG_OPF = 0.00
        self.Pr_OPF = 0.00
        self.PG_FAC   = 0.00
        self.QG_FAC   = 0.00
        self.PG_OPFAC = 0.00
        self.QG_OPFAC = 0.00
        self.Pr_OPFAC = 0.00
        self.Pr_PFAC = 0.00  # Calculated considering GNN solution
        # Auxiliar
        self.Ploss_ficload = 0.00
        self.cir_available = 0
        self.isolated = False
        # Reliability
        self.PL_current = 0.00
        self.QL_current = 0.00
        self.PG_desp_current = 0.00
        self.AF_LOLP = 0.0                       # LOLP cumulative test function        
        self.AF2_LOLP = 0.0                      # LOLP cumulative squared test function
        self.AF_EPNS = 0.0                       # EPNS cumulative test function - general
        self.AF2_EPNS = 0.0                      # EPNS cumulative squared test function
        self.LOLP = 0.0
        self.LOLE = 0.0
        self.EPNS = 0.0
        self.EENS = 0.0
        self.beta_LOLP = 0.0
        self.beta_EPNS = 0.0
        # DRL
        self.P_LS = 0.00  
        # Gstat info (even when from different generating station)
        self.PG_max = 0.0
        self.PG_min = 0.0
        self.QG_max = 0.0
        self.QG_min = 0.0
        self.PG_max_current = 0.0
        self.PG_min_current = 0.0
        self.QG_max_current = 0.0
        self.QG_min_current = 0.0
        self.nu_total = 0.0
        self.nu_available_current = 0.0
        self.PG_max_orig = 0.0
        self.cap_slack = 0.0

    # -----------------------------------------
    # Reset values - FDC
    def reset_values_FDC(self):
        self.Theta_FDC = 0.00
        self.Pi_FDC = 0.00
        self.PG_FDC = 0.00
        self.Ploss_ficload = 0.00

    # -----------------------------------------
    # Reset values - OPF
    def reset_values_OPF(self):
        self.Theta_OPF = 0.00
        self.Pi_OPF = 0.00
        self.PG_OPF = 0.00
        self.Pr_OPF = 0.00
        self.Ploss_ficload = 0.00
    
    # -----------------------------------------
    # Reset values - FAC
    def reset_values_FAC(self):
        self.Theta_FAC = 0.00
        self.Pi_FAC = 0.00
        self.Qi_FAC = 0.00
        self.PG_FAC = 0.00
        self.QG_FAC = 0.00
        self.P_LS = 0.00
        #self.Pr_PFAC = 0.00
    
    # -----------------------------------------
    # Reset values - OPFAC
    def reset_values_OPFAC(self):
        self.Theta_OPFAC = 0.00
        self.Pi_OPFAC = 0.00
        self.Qi_OPFAC = 0.00
        self.PG_OPFAC = 0.00
        self.QG_OPFAC = 0.00
        self.Pr_OPFAC = 0.00

    # -----------------------------------------
    # Reset values - Relibility Indices
    def reset_values_reliability_indices(self):
        self.AF_LOLP = 0.0                       # LOLP cumulative test function        
        self.AF2_LOLP = 0.0                      # LOLP cumulative squared test function
        self.AF_EPNS = 0.0                       # EPNS cumulative test function - general
        self.AF2_EPNS = 0.0                      # EPNS cumulative squared test function
        self.LOLP = 0.0
        self.LOLE = 0.0
        self.EPNS = 0.0
        self.EENS = 0.0
        self.beta_LOLP = 0.0
        self.beta_EPNS = 0.0

    # -----------------------------------------
    # String representation while debugging Python
    def __repr__(self) -> str:
        return f"Person(id: {self.id}, number: {self.number}, {self.type}, Pr_PFAC: {self.Pr_PFAC:6.4f}, V_FAC: {self.V_FAC:6.4f}, PL: {self.PL:6.2f}, PL_curr: {self.PL_current:6.2f}, PG_desp: {self.PG_desp:6.2f}, PG_desp_curr: {self.PG_desp_current:6.2f})"
    
# =================================================================================== #
# Class for Circuit
class cir:
    
    # -----------------------------------------
    # Constructor
    def __init__(self):
        # -----------------------------------------
        # Attributes
        # Input
        self.bF = bus()                # Bus FROM
        self.bT = bus()                # Bus TO
        self.id = 0                    # Cir id number
        self.name = ''                 # Cir name (bus from - bus to)
        self.r = 0.0                   # Resistence (pu)
        self.x = 0.0                   # Reactance (pu)
        self.b = 0.0                   # Susceptance (pu)
        self.cap_n = 0.0               # Normal capacity (MVA)
        self.cap_e = 0.0               # Emergence capacity (MVA)
        self.cap_n_orig = 0.0          # Original normal capacity (MVA)
        self.cap_e_orig = 0.0          # Original emergence capacity (MVA)
        self.tap = 0.0                 # tap transform angle (pu)
        self.shift_def = 0.0           # Shift transform angle (°)
        self.area = 0                  # Area
        self.failure_rate = 0.0        # Failure rate (occ./year)
        self.MTTR = 0.0                # Mean time to failure (H)
        self.FOR = 0.0                 # Forced outage rate
        self.pos = 0                   # Input position
        self.visited = False
        # Power flow output
        self.Pij_FDC = 0.00
        self.Pji_FDC = 0.00
        self.Pij_OPF = 0.00
        self.Pji_OPF = 0.00
        self.Ploss_FDC = 0.00
        self.Ploss_OPF = 0.00
        self.Pij_FAC = 0.00
        self.Pji_FAC = 0.00
        self.Qij_FAC = 0.00
        self.Qji_FAC = 0.00
        self.Sij_FAC = 0.00
        self.Sji_FAC = 0.00
        self.Pij_OPFAC = 0.00
        self.Pji_OPFAC = 0.00
        self.Qij_OPFAC = 0.00
        self.Qji_OPFAC = 0.00
        self.Sij_OPFAC = 0.00
        self.Sji_OPFAC = 0.00
        self.Ploss_FAC = 0.00
        self.Qloss_FAC = 0.00
        self.Ploss_OPFAC = 0.00
        self.Qloss_OPFAC = 0.00
        self.flowRead = False
        # Reliability
        self.available = True          # Boolean variable indicating whether the circuit is available in the current evaluation state (True or False)
        # ML
        self.cluster = 0               # Cluster to which it belongs - clustering process using (PsiF, PsiS)
        self.phiList = {'state_train_ID': [], 'n_u': [], 'phi': []}  # Dictionary with: state_ID, n_u (number of unavailable circuits), and phi indicators - for each state
        self.PsiF = 0.0                # Score that expresses how much the component’s unavailability is associated with system failure
        self.PsiS = 0.0                # Score that translates how much the component’s unavailability is associated with system success
        # DRL
        self.S_charge_rate = 0.0       # Circuit charge rate
        self.g_l = 0.0                   # Longitudinal conductance (pu)
        self.b_l = 0.0                   # Longitudinal susceptance (pu)

    # -----------------------------------------
    # Reset values - FDC
    def reset_values_FDC(self):
        self.Pij_FDC = 0.00
        self.Pji_FDC = 0.00
        self.Ploss_FDC = 0.00
        self.flowRead = False

    # -----------------------------------------
    # Reset values - OPF
    def reset_values_OPF(self):
        self.Pij_OPF = 0.00
        self.Pji_OPF = 0.00
        self.Ploss_OPF = 0.00
        self.flowRead = False

    # -----------------------------------------
    # Reset values - FAC
    def reset_values_FAC(self):
        self.Pij_FAC = 0.00
        self.Pji_FAC = 0.00
        self.Qij_FAC = 0.00
        self.Qji_FAC = 0.00
        self.Sij_FAC = 0.00
        self.Sji_FAC = 0.00
        self.Ploss_FAC = 0.00
        self.Qloss_FAC = 0.00
        self.flowRead = False

    # -----------------------------------------
    # Reset values - OPFAC
    def reset_values_OPFAC(self):
        self.Pij_OPFAC = 0.00
        self.Pji_OPFAC = 0.00
        self.Qij_OPFAC = 0.00
        self.Qji_OPFAC = 0.00
        self.Sij_OPFAC = 0.00
        self.Sji_OPFAC = 0.00
        self.Ploss_OPFAC = 0.00
        self.Qloss_OPFAC = 0.00
        self.flowRead = False

    # -----------------------------------------
    # Cumput phi for ML features
    def compute_phi(self, _state_train_ID, _n_unavailableC, _n_uc):
        self.phiList['state_train_ID'].append(_state_train_ID)
        self.phiList['n_u'].append(_n_uc)
        #self.phiList['phi'].append(_n_uc/_n_unavailableC)
        self.phiList['phi'].append(_n_uc)
        
    # -----------------------------------------
    # String representation while debugging Python
    def __repr__(self) -> str:
        return f"Person(id: {self.id}, {self.bF.number}-{self.bT.number}, available: {self.available}, f_rate: {self.failure_rate}, S_charge: {self.S_charge_rate})"
    
# =================================================================================== #
# Class for Generating station
class gstat:
    
    # -----------------------------------------
    # Constructor
    def __init__(self):
        # -----------------------------------------
        # Attributes
        self.id = 0                    # Generating station id number
        self.bus = bus()               # Connection bus
        self.nu = 0                    # Number of generating units
        self.stat_class = 0            # Generating station class
        self.P_min = 0.0               # Minimum capacity (MW)
        self.P_max = 0.0               # Maximum capacity (MW)
        self.P_min_orig = 0.0          # Original minimum capacity (MW)
        self.P_max_orig = 0.0          # Original maximum capacity (MW)
        self.Q_min = 0.0               # Minimum capacity (MVAr)
        self.Q_max = 0.0               # Maximum capacity (MVAr)
        self.Q_min_orig = 0.0          # Original minimum capacity (MVAr)
        self.Q_max_orig = 0.0          # Original maximum capacity (MVAr)
        self.cost = 0.0                # Cost ($/MWh) - linear coefficient
        self.cost2 = 0.0               # Cost ($/MWh^2) - quadratic coefficient
        self.cost_const = 0.0          # Cost ($) - constant coefficient
        self.failure_rate = 0.0        # Failure rate (occ./year)
        self.MTTR = 0.0                # Mean time to failure (H)
        self.FOR = 0.0                 # Forced outage rate
        self.pos = 0                   # Input position
        self.generationCapacity = 0.0  # Generation capacity (MW)
        # Reliability
        self.nu_available = 0          # Variable indicating the number of generation units available in the current evaluation state
        self.state_space = []          # State space for nu generating units
        # Auxiliar
        self.cap_slack = 0.0           # Difference between current dispatch in relation to capacity (capacity slack)
        self.factor_disp = 0.0
        self.PG_ACPF = 0.0
        self.QG_ACPF = 0.0
        self.PG_ACOPF = 0.0
        self.QG_ACPPF = 0.0
        # ML
        self.cluster = 0               # Cluster to which it belongs - clustering process using (PsiF, PsiS)
        self.phiList = {'state_train_ID': [], 'n_u': [], 'phi': []}  # Dictionary with: state_ID, n_u (number of unavailable generating units), and phi indicators - for each state
        self.PsiF = 0.0                # Score that expresses how much the component’s unavailability is associated with system failure
        self.PsiS = 0.0                # Score that translates how much the component’s unavailability is associated with system success
    
    # -----------------------------------------
    # Method to construct the state space
    def constructing_state_space(self):
        self.state_space = []
        N = self.nu
        Prob_prev = 0
        for m in range(N + 1):
            C = math.factorial(N) / (math.factorial(m) * math.factorial(N - m))
            Prob = C * math.pow((1 - self.FOR), m) * math.pow(self.FOR, N - m)
            self.state_space.append(Prob + Prob_prev)
            Prob_prev += Prob

    # -----------------------------------------
    # Cumput phi for ML features
    def compute_phi(self, _state_train_ID, _n_unavailableG, _n_ug):
        self.phiList['state_train_ID'].append(_state_train_ID)
        self.phiList['n_u'].append(_n_ug)
        #self.phiList['phi'].append(_n_ug/_n_unavailableG)
        self.phiList['phi'].append(_n_ug)
    
    # -----------------------------------------
    # String representation while debugging Python
    def __repr__(self) -> str:
        return f"Person(id: {self.id}, bus: {self.bus.number}, nu: {self.nu}, nu_a: {self.nu_available}, PG_FAC: {self.bus.PG_FAC:.2f}, P_max: {self.P_max}, QG_FAC: {self.bus.QG_FAC:.2f}, QG_max: {self.Q_max})"

        