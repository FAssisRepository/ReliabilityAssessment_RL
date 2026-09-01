# =================================================================================== #
# OPTIMAL POWER FLOW (OPF) AC
# =================================================================================== #
import os
import pandas as pd
import numpy as np
import math
from pypower.api import runopf, ppoption
from pypower.idx_bus import BUS_I, PD, QD, VM, VA, LAM_P, LAM_Q
from pypower.idx_gen import GEN_BUS, PG, QG, PMAX, PMIN, GEN_STATUS
from pypower.idx_brch import F_BUS, T_BUS, PF, QF, PT, QT, RATE_A

# =================================================================================== #
# Class for OPF AC
class OPF_AC_class:

    # -----------------------------------------
    # Constructor
    def __init__(self):
        # -----------------------------------------
        # Attributes
        self.loadshedding_cost = 1000    # Cost of load shedding [$/pu]
        self.loadshedding_total = 0.0    # Total load shedding observed [MW]
        self.x0 = None                   # Initial solution for OPF (reoptmization)
        self.results = ''                # Results from AC power flow
        self.gen_dict = {'bus_id': [], 'gen_id':[], 'gen_type':[]}

    # -----------------------------------------
    # Method to run OPF AC
    def run_OPF_AC(self, _data):

        # -----------------------------------------
        system = _data.system
        S_base = _data.simulation_set.s_base
        n_bus = system.n_bus
        n_cir = system.n_cir
        #self.x0 = system.x0                      # Initial solution to reoptimization

        self.loadshedding_total = 0.0

        # -----------------------------------------
        # Resetting values to "zero", identifying sw bar, and creating the system in PYPOWER
        buses = []               # Bus data
        gstagions_LS = []        # Load shedding “generator” data
        gstagions_LS_cost = []   # Load shedding “generator” cost data
        type_gen = 2             # Polynomial cost (cg*Pg)
        index_gen = []
        cont_gen = 0
        index_LS  = []
        self.gen_dict = {'bus_id': [], 'gen_id':[], 'gen_type':[]}
        gstagions = []           # Generator data
        gstagions_cost = []      # Generator cost data
        for bus in system.dbus:
            bus.reset_values_OPFAC()
            if(bus.isolated == False):
                Vmax = bus.Vmax
                Vmin = bus.Vmin
                if(bus.type_current == "PQ"): bus_type = 1
                elif(bus.type_current == "PV"): bus_type = 2
                else: bus_type = 3
                # Columns: bus_i type Pd Qd Gs Bs area Vm Va baseKV zone Vmax Vmin
                # new_bus = np.array([bus.id, bus_type, bus.PL_current, bus.QL, 0, S_base*bus.bshunt, bus.area, bus.V, 0, 500, 1, bus.Vmax, bus.Vmin])
                # PL_current = 0 even para load buses (To represent as 'Dispatchable load' in mpc.gen)
                new_bus = np.array([bus.id, bus_type, 0, bus.QL_current, 0, S_base*bus.bshunt, bus.area, bus.V_FAC, bus.Theta_FAC, 500, 1, Vmax, Vmin])
                buses.append(new_bus)
                # Creating the load shedding generator
                if bus.PL_current > 0:
                    index_LS.append(cont_gen)
                    # Columns: bus Pg Qg Qmax Qmin Vg mBase status Pmax Pmin ...
                    g_LS = np.array([bus.id, -bus.PL_current, 0, 0, 0, 1.0, S_base, 1, 0, -bus.PL_current])
                    gstagions.append(g_LS)
                    # Columns: 1 startup shutdown n c(n-1) ... c0
                    new_g_LS_cost = np.array([type_gen, 0, 0, 3, 1, bus.int_cost, 0])
                    gstagions_cost.append(new_g_LS_cost)
                    self.gen_dict['bus_id'].append(bus.id)
                    self.gen_dict['gen_id'].append(cont_gen)
                    self.gen_dict['gen_type'].append('LS')
                    cont_gen += 1
            else:
                bus.Pr_OPFAC = bus.PL_current
        # Generator data
        for gstat in system.dgstat:
            if(gstat.nu_available > 0 and gstat.bus.isolated == False):
                index_gen.append(cont_gen)
                # Columns: bus Pg Qg Qmax Qmin Vg mBase status Pmax Pmin ...
                # new_gstat = np.array([gstat.bus.id, gstat.bus.PG_desp_current, 0, gstat.nu_available*gstat.Q_max, gstat.nu_available*gstat.Q_min, gstat.bus.V, S_base, 1, gstat.nu_available*gstat.P_max, gstat.nu_available*gstat.P_min])
                new_gstat = np.array([gstat.bus.id, gstat.bus.PG_FAC * gstat.factor_disp, gstat.bus.QG_FAC * gstat.factor_disp, gstat.nu_available*gstat.Q_max, gstat.nu_available*gstat.Q_min, gstat.bus.V, S_base, 1, gstat.nu_available*gstat.P_max, gstat.nu_available*gstat.P_min])
                gstagions.append(new_gstat)
                # Columns: 1 startup shutdown n c(n-1) ... c0
                new_gstat_cost = np.array([type_gen, 0, 0, 3, gstat.cost2, gstat.cost, gstat.cost_const])
                gstagions_cost.append(new_gstat_cost)
                self.gen_dict['bus_id'].append(gstat.bus.id)
                self.gen_dict['gen_id'].append(cont_gen)
                self.gen_dict['gen_type'].append('GEN')
                cont_gen += 1
        # Branch data
        circuits = []
        for cir in system.dcir:
            cir.reset_values_OPFAC()
            # Columns: fbus tbus r x b rateA rateB rateC ratio angle status
            new_cir = np.array([cir.bF.id, cir.bT.id, cir.r, cir.x, cir.b, cir.cap_n, cir.cap_e, cir.cap_e, cir.tap, cir.shift_def, cir.available])
            circuits.append(new_cir)          
        
        ppc = {
            "version": '2',
            "baseMVA": 100.0,           
            "bus": np.array(buses, dtype=float),
            "gen": np.array(gstagions, dtype=float),
            "branch": np.array(circuits, dtype=float),
            "gencost": np.array(gstagions_cost, dtype=float)
        }

        # VERBOSE (0: No printed output (silent); 1: Errors and warnings only; 2: Detailed output; 3: Detailed solver output)  
        # OUT_ALL (0: No result tables printed; 1: Results for buses, generators, and branches; -1: Only solved values)
        
        ppopt = ppoption(VERBOSE=0, OUT_ALL=0)
        self.results = runopf(ppc, ppopt)

        # -----------------------------------------
        # Colecting power flow results       
        # Gen results
        gen = self.results["gen"]
        bus_idx = gen[:, GEN_BUS]      # Generator bus        
        Pg = gen[:, PG]                # Active power generation (MW)
        for id_gen in range(gen.shape[0]):
            if(self.gen_dict['gen_type'][id_gen] == 'LS'):
                for bus in _data.system.dbus:
                    if(bus.id == self.gen_dict['bus_id'][id_gen]):
                        ls = float(Pg[self.gen_dict['gen_id'][id_gen]]) + bus.PL_current
                        if(ls > 0.00001):                                       
                            bus.Pr_OPFAC = ls
                            bus.Pi_OPFAC += bus.Pr_OPFAC
                            self.loadshedding_total += ls
                        break

        # -----------------------------------------
        # Colecting power flow results
        # self.colect_flow_OPFAC_results(_data)
        
    # -----------------------------------------
    # Method to colect power flow results
    def colect_flow_OPFAC_results(self, _data):

        # -----------------------------------------
        # Colecting power flow results
        # Bus results
        buses = self.results["bus"]
        bus_numbers = buses[:, BUS_I]
        Pd = buses[:, PD]              # Active power load (MW)
        Qd = buses[:, QD]              # Reactive power load (MVAr)
        Vm = buses[:, VM]              # Voltage magnitude (p.u.)
        Va = buses[:, VA]              # Voltage angle (deg)        
        # Gen results
        gen = self.results["gen"]
        bus_idx = gen[:, GEN_BUS]      # Generator bus        
        Pg = gen[:, PG]                # Active power generation (MW)
        Qg = gen[:, QG]                # Reactive power (MVAr)
        Pmax = gen[:, PMAX]
        Pmin = gen[:, PMIN]

        Pg_gstat = []
        Qg_gstat = []
        for id_gen in range(gen.shape[0]):
            if(self.gen_dict['gen_type'][id_gen] == 'GEN'):
                Pg_gstat.append(Pg[id_gen])
                Qg_gstat.append(Qg[id_gen])
        
        cont_gen = 0
        for gstat in _data.system.dgstat:
            gstat.PG_ACOPF = 0.0
            gstat.QG_ACOPF = 0.0
            if(gstat.nu_available > 0 and gstat.bus.isolated == False):
                gstat.PG_ACOPF = float(Pg_gstat[cont_gen])
                gstat.QG_ACOPF = float(Qg_gstat[cont_gen])
                cont_gen += 1
        
        cont_bus = 0
        for bus in _data.system.dbus:
            if(bus.isolated == False):
                bus.V_OPFAC = float(Vm[cont_bus])
                bus.Theta_OPFAC = float(Va[cont_bus])
                cont_bus += 1
        
        for bus in _data.system.dbus_gstat:
            bus.PG_OPFAC = 0.0
            bus.QG_OPFAC = 0.0
            for gstat in bus.gstat:    
                bus.PG_OPFAC += gstat.PG_ACOPF
                bus.QG_OPFAC += gstat.QG_ACOPF     
            bus.Pi_OPFAC = bus.PG_OPFAC - bus.PL_current
            bus.Qi_OPFAC = bus.QG_OPFAC - bus.QL_current       

        # Circuit results
        branch = self.results["branch"]
        from_bus = branch[:, F_BUS]
        to_bus = branch[:, T_BUS]
        pf = branch[:, PF]             # Active power flow from sending end (MW)
        qf = branch[:, QF]             # Reactive power flow from sending end (MVAr)
        pt = branch[:, PT]             # Active power flow from receiving end (MW)
        qt = branch[:, QT]             # Reactive power flow from receiving end (MVAr)

        for cont_cir, cir in enumerate(_data.system.dcir):
            cir.Pij_OPFAC = float(pf[cont_cir])
            cir.Qij_OPFAC = float(qf[cont_cir])
            cir.Pji_OPFAC = float(pt[cont_cir])
            cir.Qji_OPFAC = float(qt[cont_cir])
            cir.Sij_OPFAC = float(np.sqrt(cir.Pij_OPFAC**2 + cir.Qij_OPFAC**2))
            cir.Sji_OPFAC = float(np.sqrt(cir.Pji_OPFAC**2 + cir.Qji_OPFAC**2))
            cir.Ploss_OPFAC = cir.Pij_OPFAC + cir.Pji_OPFAC
            cir.Qloss_OPFAC = cir.Qij_OPFAC + cir.Qji_OPFAC           
    
    # -----------------------------------------
    # Method to print results  
    def print_OPF_AC(self, _data, _file_name=None):
        
        _mainDir = _data.simulation_set.mainDir
        _outputDir = _data.simulation_set.outputDir+'\Test-'+str(_data.simulation_set.current_test + 1)
        if os.path.isdir(_outputDir):
            _outputDir = _data.simulation_set.outputDir+'\Test-'+str(_data.simulation_set.current_test + 1)
        else:
            _outputDir = _data.simulation_set.outputDir
        #_outputDir = _mainDir
        
        # Changing the directory 
        os.chdir(_outputDir)

        # -----------------------------------------
        # Colecting power flow results
        self.colect_flow_OPFAC_results(_data)

        # Creating file
        if(_file_name==None):
            file_name = 'AC-OPF.txt'
        else:
            file_name = 'AC-OPF'+_file_name+'.txt'
        file =  open(file_name, 'w')

        file.write('-------------------------\n')
        file.write('OPF - AC\n')
        file.write('-------------------------\n\n')

        file.write(' System: {}\n\n'.format(_data.simulation_set.system_assessment))

        file.write('-------------------------------------------------------------------------------------------------------------------------------\n')
        file.write('BUS REPORT:\n')
        file.write('-------------------------------------------------------------------------------------------------------------------------------\n')
        file.write('{:8s} {:9s} {:8s}   {:12s} {:12s} {:12s} {:10s} {:14s} {:11s} {:9s}   {:11s}\n'.format('  Bus', 'V (pu)', 'Theta (°)','Pi (MW)','Qi (MW)','PG (MW)','QG (MW)','CapG (MW)','PL (MW)','QL (MW)','Pr (MW)'))
        PG_total = 0.00
        QG_total = 0.00
        PL_total = 0.00
        QL_total = 0.00
        Pr_total = 0.00
        for bus in _data.system.dbus:
            if(bus.gstat == []):
                if(bus.Pr_OPFAC > 0.0):
                    Pr_total += bus.Pr_OPFAC
                    file.write('{:5d}  {:8.4f}  {:8.2f}  {:11.4f}  {:11.4f}  {:11.4f}  {:11.4f}  {:11.4f}  {:11.4f} {:11.4f} {:11.4f}\n'.format(bus.number, bus.V_OPFAC, bus.Theta_OPFAC, bus.Pi_OPFAC, bus.Qi_OPFAC, bus.PG_OPFAC, bus.QG_OPFAC, 0.00, bus.PL_current - bus.Pr_OPFAC, bus.QL_current, bus.Pr_OPFAC))
                else:
                    file.write('{:5d}  {:8.4f}  {:8.2f}  {:11.4f}  {:11.4f}  {:11.4f}  {:11.4f}  {:11.4f}  {:11.4f} {:11.4f} {:11s}\n'.format(bus.number, bus.V_OPFAC, bus.Theta_OPFAC, bus.Pi_OPFAC, bus.Qi_OPFAC, bus.PG_OPFAC, bus.QG_OPFAC, 0.00, bus.PL_current - bus.Pr_OPFAC, bus.QL_current, ' '))
            else:
                if(bus.Pr_OPFAC > 0.0):
                    Pr_total += bus.Pr_OPFAC
                    file.write('{:5d}  {:8.4f}  {:8.2f}  {:11.4f}  {:11.4f}  {:11.4f}  {:11.4f}  {:11.4f}  {:11.4f} {:11.4f} {:11.4f}\n'.format(bus.number, bus.V_OPFAC, bus.Theta_OPFAC, bus.Pi_OPFAC, bus.Qi_OPFAC, bus.PG_OPFAC, bus.QG_OPFAC, bus.PG_max_current, bus.PL_current - bus.Pr_OPFAC, bus.QL_current, bus.Pr_OPFAC))
                else:
                    file.write('{:5d}  {:8.4f}  {:8.2f}  {:11.4f}  {:11.4f}  {:11.4f}  {:11.4f}  {:11.4f}  {:11.4f} {:11.4f} {:20s}\n'.format(bus.number, bus.V_OPFAC, bus.Theta_OPFAC, bus.Pi_OPFAC, bus.Qi_OPFAC, bus.PG_OPFAC, bus.QG_OPFAC, bus.PG_max_current, bus.PL_current - bus.Pr_OPFAC, bus.QL_current,' '))
            PG_total += bus.PG_OPFAC
            QG_total += bus.QG_OPFAC
            PL_total += bus.PL_current - bus.Pr_OPFAC
            QL_total += bus.QL_current
        file.write('-------------------------------------------------------------------------------------------------------------------------------\n')
        file.write('Total: {:57.4f} {:12.4f} {:25.4f} {:11.4f} {:11.4f}\n'.format(PG_total, QG_total, PL_total, QL_total,Pr_total))
        file.write('-------------------------------------------------------------------------------------------------------------------------------\n\n')
        
        file.write('----------------------------------------------------------------------------------------------------------------------------------------------------------\n')
        file.write('CIRCUIT REPORT:\n')
        file.write('----------------------------------------------------------------------------------------------------------------------------------------------------------\n')
        file.write('{:10s} {:9s} {:13s} {:13s} {:13s} {:12s} {:12s}  {:12s} {:12s} {:12s} {:12s} {:8s}\n'.format('BusFrom','BusTo','Pij (MW)','Qij (MW)','Sij (MW)','Pji (MW)','Qji (MW)','Sji (MW)','Ploss(MW)','Qloss(MW)','Cap (MVA)','Activated'))
        Ploss_total = 0.00
        Qloss_total = 0.00
        for cir in _data.system.dcir:
            if(abs(max(cir.Sij_OPFAC,cir.Sji_OPFAC) - cir.available*cir.cap_n) < 0.0001):
                file.write('{:7d}  {:7d}  {:11.4f}  {:12.4f}  {:12.4f}  {:12.4f}  {:11.4f}  {:12.4f}  {:12.4f}  {:11.4f}  {:11.4f}     {:6s}\n'.format(cir.bF.number, cir.bT.number, cir.Pij_OPFAC, cir.Qij_OPFAC, cir.Sij_OPFAC, cir.Pji_OPFAC, cir.Qji_OPFAC, cir.Sji_OPFAC, cir.Ploss_OPFAC, cir.Qloss_OPFAC, cir.available*cir.cap_n,'   ***   '))
            else:
                file.write('{:7d}  {:7d}  {:11.4f}  {:12.4f}  {:12.4f}  {:12.4f}  {:11.4f}  {:12.4f}  {:12.4f}  {:11.4f}  {:11.4f}     {:6s}\n'.format(cir.bF.number, cir.bT.number, cir.Pij_OPFAC, cir.Qij_OPFAC, cir.Sij_OPFAC, cir.Pji_OPFAC, cir.Qji_OPFAC, cir.Sji_OPFAC, cir.Ploss_OPFAC, cir.Qloss_OPFAC, cir.available*cir.cap_n,'         '))
            Ploss_total += cir.Ploss_OPFAC
            Qloss_total += cir.Qloss_OPFAC
        file.write('----------------------------------------------------------------------------------------------------------------------------------------------------------\n')
        file.write('Total: {:105.4f} {:12.4f}\n'.format(Ploss_total, Qloss_total))
        file.write('----------------------------------------------------------------------------------------------------------------------------------------------------------\n\n')

        # Closing file
        file.close()
        os.chdir(_mainDir)
        