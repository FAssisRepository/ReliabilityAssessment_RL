# =================================================================================== #
# FLOW AC
# =================================================================================== #
import os
import pandas as pd
import numpy as np
import math
from pypower.api import runpf, ppoption
#from pypower.api import makeYbus
from pypower.idx_bus import BUS_I, PD, QD, VM, VA
from pypower.idx_gen import GEN_BUS, PG, QG, PMAX, PMIN, GEN_STATUS
from pypower.idx_brch import F_BUS, T_BUS, PF, QF, PT, QT

# =================================================================================== #
# Class for flow AC
class flow_AC_class:

    # -----------------------------------------
    # Constructor
    def __init__(self):

        self.overload_total = 0.0            # Total circuit overload observed [MW]
        self.circuit_violation = 0.0         # Sum of power flow violations (absolute values of aparent power flow deviations) [pu]
        self.voltage_violation = 0.0         # Sum of voltage magnitude violations (absolute values of both under- and over-voltage deviations) [pu]
        self.reactive_violation = 0.0        # Sum of reactive power generation violations (absolute values of both under- and over-reative capacity deviations) [pu]
        self.activeSW_violation = 0.0        # Active power violation at slack bus [MW]
        self.flowAC_solution_found = True    # Boolean variable to check the identification of the flow AC solution (True or False)
        self.island_load_shedding = 0.0      # Load shedding due to bus island
        self.results = ''                    # Results from AC power flow
        self.Ybus = 0
        self.max_V_viol = 0.0
        self.max_PGSW_viol = 0.0
        self.max_QG_viol = 0.0
        self.max_Sij_viol = 0.0
        self.max_Pr = 0.0
        self.total_load_shedding = 0.0

    # -----------------------------------------
    # Method to run flow AC    
    def run_flow_AC(self, _data):
        
        # -----------------------------------------
        system = _data.system
        S_base = _data.simulation_set.s_base
        n_bus = system.n_bus
        n_cir = system.n_cir

        # -----------------------------------------
        # Resetting values to "zero", identifying sw bar, and creating the system in PYPOWER
        # Bus data
        buses = []
        self.max_Pr = 0.0
        self.island_load_shedding = 0.0
        self.total_load_shedding = 0.0
        for bus in system.dbus:
            bus.reset_values_FAC()
            if(bus.isolated == False):
                if(bus.type_current == "PQ"): bus_type = 1
                elif(bus.type_current == "PV"): 
                    if(bus.nu_available_current > 0):
                        bus_type = 2
                    else:
                        bus_type = 1
                else: bus_type = 3
                # Columns: bus_i type Pd Qd Gs Bs area Vm Va baseKV zone Vmax Vmin
                new_bus = np.array([bus.id, bus_type, bus.PL_current, bus.QL_current, 0, S_base*bus.bshunt, bus.area, bus.V, 0, 500, 1, bus.Vmax, bus.Vmin])
                buses.append(new_bus)
            else:
                self.island_load_shedding += bus.PL_current
            self.total_load_shedding += bus.Pr_PFAC
            self.max_Pr = max(self.max_Pr, bus.Pr_PFAC)
        # Generator data
        gstagions = []
        for gstat in system.dgstat:
            if(gstat.nu_available > 0 and gstat.bus.isolated == False):
                # Columns: bus Pg Qg Qmax Qmin Vg mBase status Pmax Pmin ...
                new_gstat = np.array([gstat.bus.id, gstat.bus.PG_desp_current * gstat.factor_disp, 0, gstat.nu_available*gstat.Q_max, gstat.nu_available*gstat.Q_min, gstat.bus.V, S_base, 1, gstat.nu_available*gstat.P_max, gstat.nu_available*gstat.P_min])
                gstagions.append(new_gstat)
        # Branch data
        circuits = []
        for cir in system.dcir:
            cir.reset_values_FAC()
            # Columns: fbus tbus r x b rateA rateB rateC ratio angle status
            new_cir = np.array([cir.bF.id, cir.bT.id, cir.r, cir.x, cir.b, cir.cap_n, cir.cap_e, cir.cap_e, cir.tap, cir.shift_def, cir.available])
            circuits.append(new_cir)
        
        ppc = {
            "version": '2',
            "baseMVA": 100.0,           
            "bus": np.array(buses, dtype=float),
            "gen": np.array(gstagions, dtype=float),
            "branch": np.array(circuits, dtype=float)
        }

        # VERBOSE (0: No printed output (silent); 1: Errors and warnings only; 2: Detailed output; 3: Detailed solver output)  
        # OUT_ALL (0: No result tables printed; 1: Results for buses, generators, and branches; -1: Only solved values)
        ppopt = ppoption(VERBOSE=0, OUT_ALL=0)
        self.results, success = runpf(ppc, ppopt)

        # -----------------------------------------
        # Colecting power flow results
        # self.colect_flow_AC_results(_data)

        return success

    # -----------------------------------------
    # Method to colect power flow results
    def colect_flow_AC_results(self, _data):

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
        self.max_V_viol = 0.0
        self.max_PGSW_viol = 0.0
        self.max_QG_viol = 0.0
        self.max_Sij_viol = 0.0
        self.max_Pr = 0.0
        
        cont_bus = 0
        self.voltage_violation = 0.0
        for i, bus in enumerate(_data.system.dbus):
            if(bus.isolated == False):
                bus.V_FAC = float(Vm[cont_bus])
                volt_viol = max(0, bus.V_FAC - bus.Vmax) + max(0, bus.Vmin - bus.V_FAC)
                self.voltage_violation +=  volt_viol / (bus.Vmax - bus.Vmin)
                self.max_V_viol = max(self.max_V_viol, volt_viol)
                bus.Theta_FAC = float(Va[cont_bus])
                cont_bus += 1     
        
        cont_gen = 0
        for gstat in _data.system.dgstat:
            gstat.PG_ACPF = 0.0
            gstat.QG_ACPF = 0.0
            if(gstat.nu_available > 0 and gstat.bus.isolated == False):
                gstat.PG_ACPF = float(Pg[cont_gen])
                gstat.QG_ACPF = float(Qg[cont_gen])
                cont_gen += 1
        self.reactive_violation = 0.0
        for bus in _data.system.dbus_gstat:
            bus.PG_FAC = 0.0
            bus.QG_FAC = 0.0
            for gstat in bus.gstat:    
                bus.PG_FAC += gstat.PG_ACPF
                bus.QG_FAC += gstat.QG_ACPF     
            bus.Pi_FAC = bus.PG_FAC - bus.PL_current
            bus.Qi_FAC = bus.QG_FAC - bus.QL_current
            QG_viol = max(0, bus.QG_FAC - bus.QG_max_current) + max(0, bus.QG_min_current - bus.QG_FAC)
            if(abs(bus.QG_max_current) > 0.0 or abs(bus.QG_min_current) > 0.0):
                self.reactive_violation += QG_viol / (bus.QG_max_current - bus.QG_min_current)
            else:
                self.reactive_violation += QG_viol
            self.max_QG_viol = max(self.max_QG_viol, QG_viol)

        bus_sw = _data.system.bus_sw_current
        self.activeSW_violation = max(0.0, bus_sw.PG_min_current - bus_sw.PG_FAC, bus_sw.PG_FAC - bus_sw.PG_max_current)
        self.max_PGSW_viol = self.activeSW_violation
        
        self.total_load_shedding = 0.0
        for bus in _data.system.dbus_load:
            self.total_load_shedding += bus.Pr_PFAC
            self.max_Pr = max(self.max_Pr, bus.Pr_PFAC)
        
        # Circuit results
        branch = self.results["branch"]
        from_bus = branch[:, F_BUS]
        to_bus = branch[:, T_BUS]
        pf = branch[:, PF]             # Active power flow from sending end (MW)
        qf = branch[:, QF]             # Reactive power flow from sending end (MVAr)
        pt = branch[:, PT]             # Active power flow from receiving end (MW)
        qt = branch[:, QT]             # Reactive power flow from receiving end (MVAr)

        self.overload_total = 0.0
        self.circuit_violation = 0.0
        for cont_cir, cir in enumerate(_data.system.dcir):
            cir.Pij_FAC = float(pf[cont_cir])
            cir.Qij_FAC = float(qf[cont_cir])
            cir.Pji_FAC = float(pt[cont_cir])
            cir.Qji_FAC = float(qt[cont_cir])
            cir.Sij_FAC = float(np.sqrt(cir.Pij_FAC**2 + cir.Qij_FAC**2))
            cir.Sji_FAC = float(np.sqrt(cir.Pji_FAC**2 + cir.Qji_FAC**2))
            Sij_max_load = max(cir.Sij_FAC, cir.Sji_FAC) 
            Sij_charge_rate = Sij_max_load / cir.cap_n
            cir.S_charge_rate = Sij_charge_rate 
            if(Sij_charge_rate > 1.0):
                self.overload_total += Sij_max_load - cir.cap_n
                self.circuit_violation += Sij_charge_rate - 1.0
                self.max_Sij_viol = max(self.max_Sij_viol, Sij_charge_rate - 1.0)
            cir.Ploss_FAC = cir.Pij_FAC + cir.Pji_FAC
            cir.Qloss_FAC = cir.Qij_FAC + cir.Qji_FAC

    # -----------------------------------------
    # Method to print results  
    def print_flow_AC(self, _data, _file_name=None):
        
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
        self.colect_flow_AC_results(_data)

        # Creating file
        if(_file_name==None):
            file_name = 'AC-PF.txt'
        else:
            file_name = 'AC-PF'+_file_name+'.txt'
        file =  open(file_name, 'w')

        file.write('-------------------------\n')
        file.write('POWER FLOW - AC\n')
        file.write('-------------------------\n\n')

        file.write(' System: {}\n\n'.format(_data.simulation_set.system_assessment))

        file.write('----------------------------------------------------------------------------------------------------------------------------------------------------------\n')
        file.write('BUS REPORT:                                                                                                           |      VIOLATIONS      |\n')
        file.write('----------------------------------------------------------------------------------------------------------------------------------------------------------\n')
        file.write('{:8s} {:9s} {:8s}   {:12s} {:12s} {:12s} {:10s} {:14s} {:11s} {:9s} {:7s} {:7s} {:7s} {:7s}\n'.format('  Bus', 'V (pu)', 'Theta (°)','Pi (MW)','Qi (MW)','PG (MW)','QG (MW)','CapG (MW)','PL (MW)','QL (MW)','   V', '  PG', '  QG', '    Pr (MW)'))
        PG_total = 0.00
        QG_total = 0.00
        PL_total = 0.00
        QL_total = 0.00
        Pr_total = 0.00
        load_level = (_data.system.current_total_load / _data.system.peak_load)
        for bus in _data.system.dbus:
            voltage_violate = '     '
            PG_violate = '     '
            QG_violate = '     '
            Applied_LS = '     '
            tol = 0.0001
            if(bus.Vmin - bus.V_FAC > tol or bus.V_FAC - bus.Vmax > tol): voltage_violate = ' *** '
            if(bus.type_current in ["PV", "SW"] and (bus.PG_min_current - bus.PG_FAC > tol or bus.PG_FAC - bus.PG_max_current > tol)): PG_violate = ' *** '
            if(bus.type_current in ["PV", "SW"] and (bus.QG_min_current - bus.QG_FAC > tol or bus.QG_FAC - bus.QG_max_current > tol)): QG_violate = ' *** '
            if(bus.Pr_PFAC > tol or bus.isolated == True): 
                Applied_LS = bus.Pr_PFAC
                if(bus.isolated == True): Applied_LS = bus.Pr_PFAC
                Pr_total += Applied_LS
                if(bus.gstat == []):
                    file.write('{:5d}  {:8.4f}  {:8.2f}  {:11.4f}  {:11.4f}  {:11.4f}  {:11.4f}  {:11.4f}  {:11.4f} {:11.4f}    {}  {}   {} {:13.4f}\n'.format(bus.number, bus.V_FAC, bus.Theta_FAC, bus.Pi_FAC, bus.Qi_FAC, bus.PG_FAC, bus.QG_FAC, 0.00, bus.PL_current, bus.QL_current, voltage_violate, PG_violate, QG_violate, Applied_LS))
                else:
                    file.write('{:5d}  {:8.4f}  {:8.2f}  {:11.4f}  {:11.4f}  {:11.4f}  {:11.4f}  {:11.4f}  {:11.4f} {:11.4f}    {}  {}   {} {:13.4f}\n'.format(bus.number, bus.V_FAC, bus.Theta_FAC, bus.Pi_FAC, bus.Qi_FAC, bus.PG_FAC, bus.QG_FAC, bus.PG_max_current, bus.PL_current, bus.QL_current, voltage_violate, PG_violate, QG_violate, Applied_LS))
            else:
                if(bus.gstat == []):
                    file.write('{:5d}  {:8.4f}  {:8.2f}  {:11.4f}  {:11.4f}  {:11.4f}  {:11.4f}  {:11.4f}  {:11.4f} {:11.4f}    {}  {}   {}        {}\n'.format(bus.number, bus.V_FAC, bus.Theta_FAC, bus.Pi_FAC, bus.Qi_FAC, bus.PG_FAC, bus.QG_FAC, 0.00, bus.PL_current, bus.QL_current, voltage_violate, PG_violate, QG_violate, Applied_LS))
                else:
                    file.write('{:5d}  {:8.4f}  {:8.2f}  {:11.4f}  {:11.4f}  {:11.4f}  {:11.4f}  {:11.4f}  {:11.4f} {:11.4f}    {}  {}   {}        {}\n'.format(bus.number, bus.V_FAC, bus.Theta_FAC, bus.Pi_FAC, bus.Qi_FAC, bus.PG_FAC, bus.QG_FAC, bus.PG_max_current, bus.PL_current, bus.QL_current, voltage_violate, PG_violate, QG_violate, Applied_LS))

            PG_total += bus.PG_FAC
            QG_total += bus.QG_FAC
            PL_total += bus.PL_current
            QL_total += bus.QL_current
            #Pr_total += (_data.system.current_total_load / _data.system.peak_load) * bus.PL - bus.PL_current
        file.write('----------------------------------------------------------------------------------------------------------------------------------------------------------\n')
        file.write('Total: {:57.4f} {:12.4f} {:25.4f} {:11.4f} {:37.4f} \n'.format(PG_total, QG_total, PL_total, QL_total, Pr_total))
        file.write('----------------------------------------------------------------------------------------------------------------------------------------------------------\n\n')
        
        file.write('----------------------------------------------------------------------------------------------------------------------------------------------------------\n')
        file.write('CIRCUIT REPORT:\n')
        file.write('----------------------------------------------------------------------------------------------------------------------------------------------------------\n')
        file.write('{:10s} {:9s} {:13s} {:13s} {:13s} {:12s} {:12s}  {:12s} {:12s} {:12s} {:12s} {:8s}\n'.format('BusFrom','BusTo','Pij (MW)','Qij (MW)','Sij (MW)','Pji (MW)','Qji (MW)','Sji (MW)','Ploss(MW)','Qloss(MW)','Cap (MVA)','Overload (%)'))
        Ploss_total = 0.00
        Qloss_total = 0.00
        overload_total = 0.00
        for cir in _data.system.dcir:
            if(max(cir.Sij_FAC,cir.Sji_FAC) > cir.available*cir.cap_n):
                if(cir.available*cir.cap_n > 0):
                    overload = 100*(max(cir.Sij_FAC,cir.Sji_FAC) - cir.cap_n) / (cir.available*cir.cap_n)
                    overload_total += overload
                file.write('{:7d}  {:7d}  {:11.4f}  {:12.4f}  {:12.4f}  {:12.4f}  {:11.4f}  {:12.4f}  {:12.4f}  {:11.4f}  {:11.4f}      {:6.2f}\n'.format(cir.bF.number, cir.bT.number, cir.Pij_FAC, cir.Qij_FAC, cir.Sij_FAC, cir.Pji_FAC, cir.Qji_FAC, cir.Sji_FAC, cir.Ploss_FAC, cir.Qloss_FAC, cir.available*cir.cap_n, overload))
            else:
                file.write('{:7d}  {:7d}  {:11.4f}  {:12.4f}  {:12.4f}  {:12.4f}  {:11.4f}  {:12.4f}  {:12.4f}  {:11.4f}  {:11.4f}  {:6s}\n'.format(cir.bF.number, cir.bT.number, cir.Pij_FAC, cir.Qij_FAC, cir.Sij_FAC, cir.Pji_FAC, cir.Qji_FAC, cir.Sji_FAC, cir.Ploss_FAC, cir.Qloss_FAC, cir.available*cir.cap_n,'     ---'))
            Ploss_total += cir.Ploss_FAC
            Qloss_total += cir.Qloss_FAC
        file.write('----------------------------------------------------------------------------------------------------------------------------------------------------------\n')
        file.write('Total: {:105.4f} {:12.4f} {:24.4f}\n'.format(Ploss_total, Qloss_total, overload_total))
        file.write('----------------------------------------------------------------------------------------------------------------------------------------------------------\n\n')

        # Closing file
        file.close()
        os.chdir(_mainDir)