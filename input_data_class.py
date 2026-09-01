# =================================================================================== #
# INPUT DATA CLASS
# =================================================================================== #
import os
import element_classes
import numpy as np

# =================================================================================== #
# Class for loading input data - System
class inputData_system:
    
    # -----------------------------------------
    # Attributes
    system = element_classes.system()            # System data object
    loadcurve = []                               # Load curve list
    REScurve = []                                # RES curve list
    simulation_set = []                          # Simulation data object

    # -----------------------------------------
    # Constructor
    def __init__(self, _mainDir, _inputDir, _output):
               
        # Changing the directory 
        os.chdir(_inputDir)

        self.simulation_set = simulation_settings(_mainDir, _inputDir, _output)
        if(self.simulation_set.system_assessment[0][-3:] == "_AC"):
            self.load_input_data_AC(_inputDir, _output)
        else:
            self.load_input_data_DC(_inputDir, _output)
        self.load_load_curve()
                                                           
    # -----------------------------------------
    # Method to load input data - Non-Linear Power Flow   
    def load_input_data_AC(self, _inputDir, _output):
        
        # Opening file
        name_file = self.simulation_set.system_assessment[0][:-3] +'_'+self.simulation_set.region_assessment +'_AC'+".dat"
        self._systemdata =  open(name_file,'r',encoding='utf-8')
        linesfile = self._systemdata.readlines()

        os.chdir(_output)
        copy_file = open(name_file,'w',encoding='utf-8')
        for line in linesfile:
            copy_file.write(line)
        copy_file.close()
        os.chdir(_inputDir)
              
        # -----------------------------------------
        # Bus data
        cont_line = 0
        while(True):            
            if(len(linesfile[cont_line]) >= 9):
                if(linesfile[cont_line][0:8] == 'BUS DATA'):
                    cont_line = cont_line + 7                  # Start line of the bus data
                    break                
            cont_line += 1                   
        
        peak_load = 0.0
        cont_bus = 0
        vector_number_buses = []
        buses = []
        while(True):
            
            if(linesfile[cont_line][0:5] == '09999'):
                break
            
            # Creating a new bus
            bus = element_classes.bus()
            bus.id = cont_bus + 1                              # Bus id number (ascending order)
            bus.number = int(linesfile[cont_line][0:5])        # Bus number (name)
            bus.type = linesfile[cont_line][19:21]             # Bus type (PV, PQ or SW)
            bus.type_current = bus.type                        # Bus type (PV, PQ or SW)
            bus.V  = float(linesfile[cont_line][22:29])        # V - Voltage magnitude (pu)
            bus.V_orig  = float(linesfile[cont_line][22:29])   # V_orig - Voltage magnitude (pu) - Original
            bus.Vmin  = float(linesfile[cont_line][30:37])     # Vmin - Voltage magnitude minimum limit (pu)
            bus.Vmax  = float(linesfile[cont_line][38:45])     # Vmax - Voltage magnitude maximum limit (pu)
            bus.PL = float(linesfile[cont_line][46:53])        # Load - Active power (MW)
            bus.QL = float(linesfile[cont_line][54:61])        # Load - Reactive power (MVAr)
            bus.QL_orig = bus.QL
            bus.PL_orig = bus.PL
            bus.QL_current = bus.QL
            bus.PL_current = bus.PL
            bus.PG_desp = float(linesfile[cont_line][62:69])   # Base dispatch (MW)
            bus.PG_desp_orig = bus.PG_desp
            bus.PG_desp_current = bus.PG_desp
            peak_load += bus.PL
            bus.bshunt  = float(linesfile[cont_line][70:77])   # b shunt (pu)
            bus.area = int(linesfile[cont_line][78:82])        # Area
            bus.int_cost = float(linesfile[cont_line][83:90])  # Interruption cost ($/MW)
            bus.region_interest = int(linesfile[cont_line][91:97])    # Does it participate in the area of ​​interest for reliability assessment? (1: yes; 0: no)
            bus.pos = cont_bus
            if(bus.type == "SW"):
                bus_sw = bus
            cont_bus += 1
            
            buses.append(bus)               # Adding bus object to list "dbus"
            self.system.n_bus += 1          # Increasing number of system bars
            vector_number_buses.append(bus.id)
            
            cont_line += 1

        aux = np.argsort(vector_number_buses)
        for pos in range(len(aux)):
            self.system.dbus.append(buses[aux[pos]])
        self.system.peak_load = peak_load

        areas = []                          # Areas List
        for bus in self.system.dbus:
            areas.append(bus.area)
            if(bus.PL > 0.00):
                self.system.dbus_load.append(bus)
                self.system.total_load += bus.PL
                self.system.current_total_load += bus.PL
                self.system.n_busload += 1
            if(bus.region_interest == 1):
                self.system.dbus_reliab.append(bus)
                if(bus.PL > 0.00):
                    self.system.region_load += bus.PL

        areas = list(set(areas))
        self.system.areas = areas
        self.system.bus_sw = bus_sw
        self.system.bus_sw_current = bus_sw
        
        # -----------------------------------------
        # Deterministic circuit data    
        max_cap = 0.0
        while(True):            
            if(len(linesfile[cont_line]) >= 27):
                if(linesfile[cont_line][0:26] == 'DETERMINISTIC CIRCUIT DATA'):
                    cont_line = cont_line + 7                  # Start line of the deterministic cir data
                    break                
            cont_line += 1                   
        
        cont_cir = 0
        while(True):
            
            if(linesfile[cont_line][0:5] == '09999'):
                break
            
            # Creating a new cir
            cir = element_classes.cir()
            for b in self.system.dbus:
                if(b.number == int(linesfile[cont_line][0:5])):
                    cir.bF = b                                 # Bus FROM
                    break
            for b in self.system.dbus:
                if(b.number == int(linesfile[cont_line][6:11])):
                    cir.bT = b                                 # Bus TO
                    break    
            # cir.id = int(linesfile[cont_line][12:16])          # Cir id number
            cir.id = cont_cir + 1                              # Cir id number
            cir.name = linesfile[cont_line][58:73].split()     # Cir name
            cir.r = float(linesfile[cont_line][17:23])/100     # Resistence (pu)
            cir.x = float(linesfile[cont_line][24:30])/100     # Reactance (pu)
            cir.b = float(linesfile[cont_line][31:37])/100     # Susceptance (pu)
            cir.g_l =  cir.r / (pow(cir.r, 2) + pow(cir.x, 2))
            cir.b_l = -cir.x / (pow(cir.r, 2) + pow(cir.x, 2))
            if(cir.g_l > self.system.glmax):
                self.system.glmax = cir.g_l
            if(cir.b_l < self.system.blmax):
                self.system.blmax = cir.b_l
            cir.cap_n = float(linesfile[cont_line][38:45])     # Normal capacity (MW)
            cir.cap_e = float(linesfile[cont_line][46:53])     # Emergence capacity (MW)
            cir.cap_n_orig = cir.cap_n
            cir.cap_e_orig = cir.cap_e
            if(cir.cap_n > max_cap):
                max_cap = cir.cap_n
            cir.tap = float(linesfile[cont_line][54:59])       # tap transform angle (pu)
            cir.shift_def = float(linesfile[cont_line][60:65]) # Shift transform angle (°)            
            cir.area = int(linesfile[cont_line][66:70])        # Area
            cir.pos = cont_cir
            cont_cir += 1

            flag = True                     # Counting the number of branches
            for cir_exist in self.system.dcir:
                if(cir.bF.number == cir_exist.bF.number and cir.bT.number == cir_exist.bT.number):
                    flag = False
                    break
            if(flag):
                self.system.n_branches += 1
            
            self.system.dcir.append(cir)    # Adding bus object to list "dcir"
            self.system.n_cir += 1          # Increasing number of system circuits
            
            cir.bF.cir_conec.append(cir)
            cir.bT.cir_conec.append(cir)

            cont_line += 1

        self.system.max_cir_capacity = max_cap    
        
        # -----------------------------------------
        # Stochastic circuit data 
        while(True):            
            if(len(linesfile[cont_line]) >= 24):
                if(linesfile[cont_line][0:23] == 'STOCHASTIC CIRCUIT DATA'):
                    cont_line = cont_line + 7                  # Start line of the stochastic circuit data
                    break                
            cont_line += 1                   
        while(True):
            
            if(linesfile[cont_line][0:5] == '09999'):
                break
            
            for c in self.system.dcir:
                if(c.bF.number == int(linesfile[cont_line][0:5])):
                    if(c.bT.number == int(linesfile[cont_line][6:11])):
                        if(c.visited == False):
                            cir_aux = element_classes.cir()
                            cir_aux = c
                            c.visited = True
                            break
                if(c.bT.number == int(linesfile[cont_line][0:5])):
                    if(c.bF.number == int(linesfile[cont_line][6:11])):
                        if(c.visited == False):
                            cir_aux = element_classes.cir()
                            cir_aux = c
                            c.visited = True
                            break
            cir_aux.failure_rate = float(linesfile[cont_line][17:27])
            cir_aux.MTTR = float(linesfile[cont_line][28:38])
            if(cir_aux.MTTR > 0.0):
                cir_aux.FOR = cir_aux.failure_rate / (cir_aux.failure_rate + (8760/cir_aux.MTTR))
                        
            cont_line += 1
        
        # -----------------------------------------
        # Generation station data
        while(True):            
            if(len(linesfile[cont_line]) >= 24):
                if(linesfile[cont_line][0:23] == 'GENERATING STATION DATA'):
                    cont_line = cont_line + 7                  # Start line of the generation station data
                    break                
            cont_line += 1

        cont_gstat = 0 
        n_total_generation_units = 0          
        while(True):
            
            if(linesfile[cont_line][0:4] == '9999'):
                break
            
            # Creating a new gstat
            gstat = element_classes.gstat()
            # gstat.id = int(linesfile[cont_line][0:4])          # Generating station id number
            gstat.id = cont_gstat + 1                            # Generating station id number
            for b in self.system.dbus:
                if(b.number == int(linesfile[cont_line][18:23])):
                    gstat.bus = b                              # Connection bus
                    bus_aux = element_classes.bus()
                    bus_aux = b
                    break
            gstat.nu = int(linesfile[cont_line][24:28])        # Number of generating units
            gstat.nu_available = gstat.nu
            n_total_generation_units += gstat.nu
            gstat.stat_class = int(linesfile[cont_line][29:33])# Generating station class
            gstat.P_min = float(linesfile[cont_line][34:42])   # Minimum capacity (MW)
            gstat.P_max = float(linesfile[cont_line][43:51])   # Maximum capacity (MW)
            gstat.P_min_orig = gstat.P_min
            gstat.P_max_orig = gstat.P_max
            if(gstat.nu * gstat.P_max_orig > self.system.max_Pmax):
                self.system.max_Pmax = gstat.nu * gstat.P_max_orig
            gstat.Q_min = float(linesfile[cont_line][52:60])   # Minimum capacity (MVAr)
            gstat.Q_max = float(linesfile[cont_line][61:69])   # Maximum capacity (MVAr)
            gstat.Q_min_orig = gstat.Q_min
            gstat.Q_max_orig = gstat.Q_max
            gstat.cost2 =      float(linesfile[cont_line][70:78])    # Cost ($/MWh) - quadratic coefficient
            gstat.cost =       float(linesfile[cont_line][79:87])    # Cost ($/MWh) - linear coefficient
            gstat.cost_const = float(linesfile[cont_line][88:96])    # Cost ($/MWh) - constant coefficient
            gstat.pos = cont_gstat
            # Conecting generating station into bus
            bus_aux.gstat.append(gstat)                              
            bus_aux.PG_max += gstat.P_max * gstat.nu
            bus_aux.PG_min += gstat.P_min * gstat.nu
            bus_aux.QG_max += gstat.Q_max * gstat.nu
            bus_aux.QG_min += gstat.Q_min * gstat.nu
            bus_aux.nu_total += gstat.nu
            bus_aux.PG_max_current += gstat.P_max * gstat.nu
            bus_aux.PG_min_current += gstat.P_min * gstat.nu
            bus_aux.QG_max_current += gstat.Q_max * gstat.nu
            bus_aux.QG_min_current += gstat.Q_min * gstat.nu
            bus_aux.nu_available_current += gstat.nu
            
            self.system.dgstat.append(gstat)    # Adding bus object to list "dcir"
            self.system.n_gstat += 1
            cont_gstat += 1 
            
            cont_line += 1
            
        self.system.n_total_generation_units = n_total_generation_units    
        
        # -----------------------------------------
        # Classes of generating station data
        cont_line = 0
        while(True):            
            if(len(linesfile[cont_line]) >= 35):
                if(linesfile[cont_line][0:34] == 'CLASSES OF GENERATING STATION DATA'):
                    cont_line = cont_line + 7                  # Start line of the classes of generation station data
                    break                
            cont_line += 1                   
        while(True):
            
            if(linesfile[cont_line][0:4] == '9999'):
                break
            
            class_id = int(linesfile[cont_line][0:4])
            for gs in self.system.dgstat:
                if(gs.stat_class == class_id):
                    gstat_aux = element_classes.gstat()
                    gstat_aux = gs
                    gstat_aux.failure_rate = float(linesfile[cont_line][29:39])
                    gstat_aux.MTTR = float(linesfile[cont_line][40:50])
                    gstat_aux.FOR = gstat_aux.failure_rate / (gstat_aux.failure_rate + (8760/gstat_aux.MTTR))
                   
            cont_line += 1 
        
        del linesfile

        # Constructing state space
        for gstat in self.system.dgstat:
            gstat.constructing_state_space()

        for bus in self.system.dbus:
            if (bus.gstat != []):
                self.system.dbus_gstat.append(bus)

        for bus in self.system.dbus_gstat:
            bus.PG_max_current = 0.0
            for gstat in bus.gstat:
                bus.PG_max_current += gstat.nu * gstat.P_max
                bus.PG_max_orig += gstat.nu * gstat.P_max
        for gstat in self.system.dgstat:
            if gstat.bus.PG_max_current > 0.0:
                gstat.factor_disp = gstat.nu * gstat.P_max / gstat.bus.PG_max_current
    
    # -----------------------------------------
    # Method to load the load curve data    
    def load_load_curve(self):

        # -----------------------------------------
        # Opening file and getting the values
        with open('IEEERTS_LOAD.load','r',encoding='utf-8') as file:
            cont_line = 0
            for line in file:
                if(cont_line > 0):
                    self.loadcurve.append(float(line))
                cont_line += 1

    # -----------------------------------------
    # Method to apply load factor
    def apply_load_factor(self):
        load_factor = self.simulation_set.load_factor[self.simulation_set.current_test]
        peak_load = 0.0
        for bus in self.system.dbus_load:
            # Increases the load by load_factor            
            bus.PL = bus.PL_orig * load_factor 
            bus.QL = bus.QL_orig * load_factor 
            bus.PL_current = bus.PL
            bus.QL_current = bus.QL
            bus.PG_desp = bus.PG_desp_orig * load_factor 
            bus.PG_desp_current = bus.PG_desp
            peak_load += bus.PL
        self.system.peak_load = peak_load

    # -----------------------------------------
    # Method to apply gen factor
    def apply_gen_factor(self):
        gen_factor = self.simulation_set.gen_factor[self.simulation_set.current_test]
        max_Pmax = 0.0
        for gstat in self.system.dgstat:
            gstat.P_min = gen_factor * gstat.P_min_orig    # Minimum capacity (MW)
            gstat.P_max = gen_factor * gstat.P_max_orig    # Maximum capacity (MW)
            gstat.P_min_orig = gstat.P_min
            gstat.P_max_orig = gstat.P_max
            gstat.Q_min = gen_factor * gstat.Q_min_orig    # Minimum capacity (MVAr)
            gstat.Q_max = gen_factor * gstat.Q_max_orig    # Maximum capacity (MVAr)
            gstat.Q_min_orig = gstat.Q_min
            gstat.Q_max_orig = gstat.Q_max
            gstat.bus.PG_max = gstat.P_max * gstat.nu
            gstat.bus.PG_min = gstat.P_min * gstat.nu
            gstat.bus.QG_max = gstat.Q_max * gstat.nu
            gstat.bus.QG_min = gstat.Q_min * gstat.nu
            if(gstat.bus.PG_max > max_Pmax):
                max_Pmax = gstat.bus.PG_max
        self.system.max_Pmax = max_Pmax
        for bus in self.system.dbus_gstat:
            bus.PG_max_current = 0.0
            bus.PG_min_current = 0.0
            bus.QG_max_current = 0.0
            bus.QG_min_current = 0.0
            for gstat in bus.gstat:
                bus.PG_max_current += gstat.P_max * gstat.nu
                bus.PG_min_current += gstat.P_min * gstat.nu
                bus.QG_max_current += gstat.Q_max * gstat.nu
                bus.QG_min_current += gstat.Q_min * gstat.nu        
        
    # -----------------------------------------
    # Method to apply circuit factor
    def apply_circuit_factor(self):
        cap_factor = self.simulation_set.circuit_factor[self.simulation_set.current_test]
        max_cir_capacity = 0.0
        for cir in self.system.dcir:
            # Increases the circuit capacity by cap_factor
            cir.cap_n = cir.cap_n_orig * cap_factor
            cir.cap_e = cir.cap_e_orig * cap_factor
            if(cir.cap_n > max_cir_capacity):
                max_cir_capacity = cir.cap_n
        self.system.max_cir_capacity = max_cir_capacity
        
        # Special cases:
        # Cables: IEEE-24B - RTS
        if(self.simulation_set.system_assessment[0][:-3] == 'IEEERTS79'):
            for cir in self.system.dcir:
                if(cir.bF.number == 1 and cir.bT.number == 2):
                    cir.cap_n = cir.cap_n_orig 
                    cir.cap_e = cir.cap_e_orig 
                if(cir.bF.number == 6 and cir.bT.number == 10):
                    cir.cap_n = cir.cap_n_orig 
                    cir.cap_e = cir.cap_e_orig
        # TRAFOS 13.8kV: SIST_200B
        if(self.simulation_set.system_assessment[0][:-3] == 'SIST_200B'):
            for cir in self.system.dcir:
                if(cir.failure_rate == 0.0 and cir.MTTR == 0.0):
                    cir.cap_n = cir.cap_n_orig 
                    cir.cap_e = cir.cap_e_orig 

class simulation_settings:

    # -----------------------------------------
    # Constructor
    def __init__(self, _mainDir, _inputDir, _outputDir):    

        # -----------------------------------------
        # Attributes
        # Case study
        self.system_assessment = ''    # System for reliability assessment
        self.s_base = 0.00             # Base power value (MVA)
        self.region_assessment = ''    # Region of assessment: "GLOBAL", "AREA", "BUS"
        self.loss_consid = False       # Will ohmic losses be considered? (True or False)
        self.RES_consid = False        # Will uncertainties related to renewable energy sources (RES) be considered? (True or False)
        self.iter = 0                  # Number of iterations to calculate losses
        # Monte Carlo simulation
        self.seed = 0                  # Seed for generating pseudorandom numbers
        self.NS_min = 0                # Minimum number of simulations (samples)
        self.NS_max = 0                # Maximum number of simulations (samples)
        self.tol = 0.00                # Tolerance for estimator variance (beta)
        self.reoptimization = False    # Will reoptimization be used for OPFs? (True or False)
        self.RA_crude_eval = []        # List - In each test, will a crude MCS (OPF without optimizaion) be performed? (True or False)
        self.RA_GNN_PT_eval = []       # List - In each test, will a GNN-MCS (trained only via SL - PT) be performed? (True or False)
        self.RA_GNN_RL_eval = []       # List - In each test, will a GNN-MCS (trained via RL) be performed? (True or False)
        # Test settings
        self.testeID_RL = []           # List with test IDs - RL 
        self.seed_RL = []              # List with seeds for generating pseudorandom numbers
        self.load_factor = []          # List with multiplicative factors for load at all buses
        self.gen_factor = []           # List with multiplicative factors for generation capacity at all generating stations
        self.circuit_factor = []       # List with multiplicative factors for the capacity of all circuits, TLs and TRAFOS
        self.samples_performance = []  # List with total of samples to performance analysis 
        # SL settings - Pretraining
        self.N_samples_PT = []         # List with numbers of samples 
        self.n_epochs_PT = []          # List with numbers of epochs 
        self.batch_size_PT = []        # List with batch sizes
        self.lr_actor_PT = []          # List with learning rates - actor
        self.lr_critic_PT = []         # List with learning rates - critic (baseline)
        # RL settings        
        self.n_episodes = []           # List with numbers of episodes during the training using RL
        self.batch_size_RL = []        # List with batch sizes
        self.lr_actor_RL = []          # List with learning rates - actor
        self.lr_critic_RL = []         # List with learning rates - critic (baseline)  
        self.entropy_coef_RL = []      # List with entropy coefficients
        self.ppo_epochs = []           # List with numbers of epochs - PPO
        self.ppo_clip = []             # List with clip values - PPO
        # Reward function weights
        self.c_LS = []                 # List with load shedding penalty weights
        self.c_cir = []                # List with line flow violation penalty weights 
        self.c_SW = []                 # List with SW generator limit violation penalty weights
        self.c_V = []                  # List with voltage magnitude violation penalty weights
        self.c_Q = []                  # List with reactive power generation violation penalty weights
        # GNNs settings
        self.hidden_dim_actor = []     # List with numbers of hidden dimensions for actor GNN
        self.n_heads_actor = []        # List with numbers of heads for actor GNN
        self.n_layers_actor = []       # List with numbers of layers for actor GNN
        self.hidden_dim_critic = []    # List with numbers of hidden dimensions for critic GNN
        self.n_heads_critic = []       # List with numbers of heads for critic GNN
        self.n_layers_critic = []      # List with numbers of layers for critic GNN
        self.std_min = []              # List with minimal values for standart deviation (exploration parameter)
        self.std_max = []              # List with maximal values for standart deviation (exploration parameter)
        self.dropout = []              # List with dropout rates
        # Simulation control
        self.current_test = 0          # Flag that indicates which test is current
        self.mainDir = _mainDir
        self.inputDir = _inputDir
        self.outputDir = _outputDir

        self.load_simulation_settings()

    # -----------------------------------------
    # Method to load simulation settings    
    def load_simulation_settings(self):
        # Opening file
        self._simulationsettings =  open('00_SIMULATION_SETTINGS.set','r',encoding='utf-8')
        linesfile = self._simulationsettings.readlines()

        os.chdir(self.outputDir)
        copy_file = open('00_SIMULATION_SETTINGS.set','w',encoding='utf-8')
        for line in linesfile:
            copy_file.write(line)
        copy_file.close()
        os.chdir(self.inputDir)

        # -----------------------------------------
        # Case study
        cont_line = 0
        while(True):            
            if(len(linesfile[cont_line]) >= 11):
                if(linesfile[cont_line][0:10] == 'CASE STUDY'):
                    cont_line = cont_line + 6                        # Start line of the case study data
                    break                
            cont_line += 1                   
            
        self.system_assessment = linesfile[cont_line][0:12].split()  # System for reliability assessment
        self.s_base = float(linesfile[cont_line][12:19])             # Base power value (MVA)
        self.region_assessment = "GLOBAL"
        self.loss_consid = False       
        self.RES_consid = False        

        # -----------------------------------------
        # Monte Carlo simulation
        cont_line = 0
        while(True):            
            if(len(linesfile[cont_line]) >= 23):
                if(linesfile[cont_line][0:22] == 'MONTE CARLO SIMULATION'):
                    cont_line = cont_line + 6                        # Start line of the case study data
                    break                
            cont_line += 1                   
            
        self.seed = int(linesfile[cont_line][0:6])                   # Seed for generating pseudorandom numbers
        self.NS_min = int(linesfile[cont_line][7:18])                # Minimum number of simulations (samples)
        self.NS_max = int(linesfile[cont_line][19:30])               # Maximum number of simulations (samples)
        self.tol = float(linesfile[cont_line][31:37])                # Tolerance for estimator variance (beta)
        self.reoptimization = False

        # -----------------------------------------
        # Test settings  
        cont_line = 0             
        while(True):            
            if(len(linesfile[cont_line]) >= 14):
                if(linesfile[cont_line][0:13] == 'TEST SETTINGS'):
                    cont_line = cont_line + 6                        # Start line of the case study data
                    break                
            cont_line += 1
        
        while(linesfile[cont_line][0:4] != '####'):
            self.testeID_RL.append(int(linesfile[cont_line][0:3]))                 # Teste ID
            self.seed_RL.append(int(linesfile[cont_line][4:9]))                    # Seed for generating pseudorandom numbers - RL
            self.load_factor.append(float(linesfile[cont_line][10:16]))            # Multiplicative factor for load at all buses
            self.gen_factor.append(float(linesfile[cont_line][17:23]))             # Multiplicative factor for generation capacity at all generating stations
            self.circuit_factor.append(float(linesfile[cont_line][24:30]))         # Multiplicative factor for the capacity of all circuits, TLs and TRAFOS
            # SL settings - Pretraining
            self.N_samples_PT.append(int(linesfile[cont_line][31:37]))             # List with numbers of samples 
            self.n_epochs_PT.append(int(linesfile[cont_line][38:44]))              # List with numbers of epochs 
            self.batch_size_PT.append(int(linesfile[cont_line][45:51]))            # List with batch sizes
            self.lr_actor_PT.append(float(linesfile[cont_line][52:58]))            # List with learning rates - actor
            self.lr_critic_PT.append(float(linesfile[cont_line][59:65]))           # List with learning rates - critic (baseline)
            # RL settings        
            self.n_episodes.append(int(linesfile[cont_line][66:74]))               # List with numbers of episodes during the training using RL
            self.batch_size_RL.append(int(linesfile[cont_line][75:81]))            # List with batch sizes
            self.lr_actor_RL.append(float(linesfile[cont_line][82:88]))            # List with learning rates - actor
            self.lr_critic_RL.append(float(linesfile[cont_line][89:95]))           # List with learning rates - critic (baseline)  
            self.entropy_coef_RL.append(float(linesfile[cont_line][96:102]))       # List with entropy coefficients  
            self.ppo_epochs.append(int(linesfile[cont_line][103:107]))             # List with numbers of epochs - PPO
            self.ppo_clip.append(float(linesfile[cont_line][108:112]))             # List with clip values - PPO
            # GNNs settings
            self.hidden_dim_actor.append(int(linesfile[cont_line][113:118]))       # List with numbers of hidden dimensions for actor GNN
            self.n_heads_actor.append(int(linesfile[cont_line][119:123]))          # List with numbers of heads for actor GNN
            self.n_layers_actor.append(int(linesfile[cont_line][124:128]))         # List with numbers of layers for actor GNN
            self.hidden_dim_critic.append(int(linesfile[cont_line][129:134]))      # List with numbers of hidden dimensions for critic GNN
            self.n_heads_critic.append(int(linesfile[cont_line][135:139]))         # List with numbers of heads for critic GNN
            self.n_layers_critic.append(int(linesfile[cont_line][140:144]))        # List with numbers of layers for critic GNN
            self.std_min.append(float(linesfile[cont_line][145:151]))              # Minimal value for standart deviation (exploration parameter)
            self.std_max.append(float(linesfile[cont_line][152:158]))              # Maximal value for standart deviation (exploration parameter)
            self.dropout.append(float(linesfile[cont_line][159:165]))              # Dropout rate
            self.RA_crude_eval.append((linesfile[cont_line][166:172].split())[0])  # Will a crude MCS (OPF without optimizaion) be performed? (True or False)
            self.RA_GNN_PT_eval.append((linesfile[cont_line][173:179].split())[0]) # Will a GNN-MCS (trained only via SL - PT) be performed? (True or False)
            self.RA_GNN_RL_eval.append((linesfile[cont_line][180:186].split())[0]) # Will a GNN-MCS (trained via RL) be performed? (True or False)
            self.samples_performance.append(int(linesfile[cont_line][187:195]))    # Total of samples to performance analysis 
            cont_line += 1

        del linesfile