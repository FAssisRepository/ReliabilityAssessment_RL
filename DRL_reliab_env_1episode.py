# =================================================================================== #
# DRL- Composite reliability assessment environment - One MCS sample = one RL episode
# =================================================================================== #
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import flow_AC
import torch
import math 
from torch_geometric.data import Data

# =================================================================================== #
# Class for DRL-based composite reliability assessment
class CompositeReliabilityEnv_1episode(gym.Env):
    ## GYMNASIUM REQUIREMENT
    metadata = {"render_modes": []}    # This environment does not support rendering

    # Attributes
    system = []
    data = []
    n_loads = 0        # Number of load buses
    n_gens = 0         # Number of generating buses (PV and SW)
    n_circuits = 0     # Number of circuits
    
    # -----------------------------------------
    # Constructor ## GYMNASIUM REQUIREMENT
    def __init__(self, _data):
        super().__init__()

        # -----------------------------------------
        # System information
        self.data = _data
        self.system = _data.system
        self.n_bus = len(self.system.dbus)
        self.n_line = len(self.system.dcir)

        # -----------------------------------------
        # Penalty weights for the reward function (fixed reward case)
        cont_Test = self.data.simulation_set.current_test

        # -----------------------------------------
        # Node features
        #    o Physical state:
        #        - Pd                                      -- active power load 
        #        - Qd                                      -- reactive power load 
        #        - available generation                    -- normalized by the maximum capacity of the generating station 
        #        - voltage magnitude                       -- V_FAC / Vmax (pu)
        #        - sin voltage angle                       -- sin(theta)
        #    o Diagnostics (pre-action): 
        #        - voltage violation                       -- max(0, V - Vmax) + max(0, Vmin - V)
        #        - reactive power violation                -- max(0, Q - Qmax) + max(0, Qmin - Q)
        self.node_feat_dim = 10
        obs_dim = self.n_bus * self.node_feat_dim

        # -----------------------------------------
        # Edge features
        #    o Physical state:
        #        - g: condutance (if circuit is available)
        #        - b: susceptance (if circuit is available)
        #        - cap: maximum capacity
        #    o Diagnostics (pre-action): 
        #        - circuit overloads ij (one per available circuit)        -- max(0, abs(Sij) - Sijmax) / Sijmax
        #        - circuit overloads ji (one per available circuit)        -- max(0, abs(Sji) - Sijmax) / Sijmax
        self.edge_feat_dim = 3

        ## GYMNASIUM REQUERIMENT
        self.observation_space = spaces.Graph(
            node_space=spaces.Box(low=-np.inf, high=np.inf, shape=(self.node_feat_dim,), dtype=np.float32),
            edge_space=spaces.Box(low=-np.inf, high=np.inf, shape=(self.edge_feat_dim,), dtype=np.float32)
        )
        ##

        # -----------------------------------------
        # Action dimension per bus
        # The DRL agent controls V of PV and SW buses, Pg of PV buses, and Pr at load buses. 
        # As controls: 
        #    o ΔV  -> voltage magnitude control            -- xxxxx (PV and SW buses) - [x,x]
        #    o ΔPg -> active power generation variation    -- xxxxx (PV buses) - [x,x]
        #    o ΔPr -> load shedding                        -- xxxxx (load buses) - [x,x]
        self.action_dim = 3
        act_dim = self.n_bus * self.action_dim
        ## GYMNASIUM REQUIREMENT
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(act_dim,), dtype=np.float32)
        ##

        # -----------------------------------------
        # Object for AC power flow
        self.flow_AC_obj = flow_AC.flow_AC_class() 

        # -----------------------------------------
        # Episode information
        self.n_steps = 0

        # -----------------------------------------
        # Individual reward information (to check during RL training)
        self.last_reward_info = {
            "load_shed": 0.0, "voltage": 0.0,
            "circuit": 0.0, "sw_active": 0.0, "reactive": 0.0
        }

        # -----------------------------------------
        # Pre-allocate observation arrays (reused every call)
        self._node_feat_buf  = np.zeros((self.n_bus,  self.node_feat_dim), dtype=np.float32)
        self._mask_buf       = np.zeros((self.n_bus,  self.action_dim),    dtype=np.float32)        
        # Cache static edge data per contingency (g, b don't change mid-episode)
        self._edge_index_cache = None
        self._edge_static_cache = None      # g, b columns
        self._n_edges_cache = 0
        self._edge_overload_buf = None      # overload column, allocated on first use
        
        self._refresh_edge_cache()
    
    # -----------------------------------------
    # Reset method
    ## GYMNASIUM REQUIREMENT
    def reset(self, seed=None, options=None):
        ## GYMNASIUM REQUIREMENT
        super().reset(seed=seed)

        # Build observation from Monte Carlo operating state
        self._refresh_edge_cache()
        state = self._build_observation()

        ## GYMNASIUM REQUIREMENT
        return state, {}       
  
    # -----------------------------------------
    # Step method
    ## GYMNASIUM REQUIREMENT
    def step(self, action):
        
        # -----------------------------------------
        # Action projection
        action = action.reshape(self.n_bus, self.action_dim)
        self._apply_action(action)

        # -----------------------------------------
        # Run power flow (without optimization)
        success = self.flow_AC_obj.run_flow_AC(self.data)
        self.flow_AC_obj.colect_flow_AC_results(self.data)
        self.last_pf_success = success

        # -----------------------------------------
        # Compute reward
        full_reward = self._compute_reward_pf(success)             # legacy combined

        # -----------------------------------------
        # Building the new observation
        
        obs  = self._build_observation()
        mask = obs.mask

        return obs, full_reward, True, False, {
            "mask":           mask,
            "reward_info":    self.last_reward_info,
        }

    # ------------------------------------------------------------------
    # Method to Apply agent action
    def _apply_action(self, action):

        for i, bus in enumerate(self.system.dbus):

            # V_adj:  Voltage magnitude variation [pu of the bus V_max]
            # Pg_adj: Active power generation variation [pu of total bus capacity]
            # Pr_adj: Active load shedding variation [pu of total bus load]
            V_adj, Pg_adj, Pr_adj = action[i]

            if bus.type in ["PV", "SW"]:
                bus.V = V_adj.item() * (bus.Vmax - bus.Vmin) + bus.Vmin

            if (bus.type in ['PV', 'SW'] and bus.PG_max_current > 0.0):
                if(bus.PG_max_current != bus.PG_min_current):
                    bus.PG_desp_current = Pg_adj.item() * ((bus.PG_max_current - bus.PG_min_current)) + bus.PG_min_current
                else:
                    bus.PG_desp_current = Pg_adj.item() * (bus.PG_max_current)
            else:
                bus.PG_desp_current = 0.0

            if (bus.PL > 0):
                Pr = Pr_adj.item() * bus.PL_current
                bus.Pr_PFAC = Pr
                bus.PL_current = bus.PL_current - Pr
    
    # ------------------------------------------------------------------
    # Method to compute costs related to reward - AC power flow (AC-PF) - Without optimization
    def compute_costs_pf(self, success=None):
        
        if success is None:
            success = self.last_pf_success
 
        n_costs = 4
 
        if not success:
            return -10.0, np.zeros(n_costs, dtype=np.float32)
 
        # ------------------------------------------------------------------
        # OBJECTIVE: load shedding (identical shaping to _compute_reward_pf)
        # ------------------------------------------------------------------
        load_shed = 0.0
        for bus in self.system.dbus:
            if bus.PL > 0.0:
                load_shed += bus.Pr_PFAC / bus.PL
        obj_reward = math.exp(-load_shed) - 1.0

        # ------------------------------------------------------------------
        # COST 0 - Voltage (all buses), normalised, RAW magnitude
        # ------------------------------------------------------------------
        cost_V = 0.0
        cont_bus = 0
        for bus in self.system.dbus:
            v_v = (max(0.0, bus.V_FAC - bus.Vmax) + max(0.0, bus.Vmin - bus.V_FAC)) / (bus.Vmax - bus.Vmin)
            if(v_v > 0.0):
                cost_V += v_v
                cont_bus += 1
        if(cont_bus > 0):
            cost_V /= cont_bus
        
        # ------------------------------------------------------------------
        # COST 1 - Reactive power QG (PV/SW buses), normalised, RAW magnitude
        # ------------------------------------------------------------------
        cost_Q = 0.0
        cont_bus = 0
        for bus in self.system.dbus_gstat:
            if bus.nu_available_current > 0:
                dQ = bus.QG_max_current - bus.QG_min_current
                denom = dQ if dQ > 0.0 else (bus.QG_max - bus.QG_min)
            else:
                denom = (bus.QG_max - bus.QG_min)
            if denom > 0.0:
                v_q = (max(0.0, bus.QG_FAC - bus.QG_max_current)  + max(0.0, bus.QG_min_current - bus.QG_FAC)) / denom
                if(v_q > 0.0):
                    cost_Q += v_q
                    cont_bus += 1
        if(cont_bus > 0):
            cost_Q /= cont_bus
        
        # ------------------------------------------------------------------
        # COST 2 - Circuit overload, normalised, RAW magnitude
        # ------------------------------------------------------------------
        cost_cir = 0.0
        cont_cir = 0
        for cir in self.system.dcir:
            v_c = max(0.0, max(abs(cir.Sij_FAC), abs(cir.Sji_FAC)) - cir.cap_n) / cir.cap_n
            if(v_c > 0.0):
                cost_cir += v_c
                cont_cir += 1
        if(cont_cir > 0):
            cost_cir /= cont_cir
        
        # ------------------------------------------------------------------
        # COST 3 - Slack-bus active power, normalised, RAW magnitude
        # ------------------------------------------------------------------
        bus_sw = self.system.bus_sw_current
        pg_viol = max(0.0,
                      bus_sw.PG_FAC - bus_sw.PG_max_current,
                      bus_sw.PG_min_current - bus_sw.PG_FAC)
        if bus_sw.nu_available_current > 0 and bus_sw.PG_max != bus_sw.PG_min:
            pg_viol /= (bus_sw.PG_max - bus_sw.PG_min) * bus_sw.nu_available_current
        cost_SW = pg_viol
        
        costs = np.array([cost_V, cost_Q, cost_cir, cost_SW], dtype=np.float32)
        return obj_reward, costs

    # ------------------------------------------------------------------
    # Method to compute reward - AC power flow (AC-PF) - Without optimization
    def _compute_reward_pf(self, success):

        if not success:
            return -10.0

        PG_SW_violation = 0.0
        QG_SW_PV_violation = 0.0
        voltage_violation = 0.0
        circuit_overload = 0.0
        load_shed = 0.0

        # -----------------------------------------
        # Bus information
        load_level = self.system.current_total_load / self.system.peak_load
        for bus in self.system.dbus:
            voltage_violation += ((max(0, bus.V_FAC - bus.Vmax) + max(0, bus.Vmin - bus.V_FAC))) / (bus.Vmax - bus.Vmin)
            if bus.PL > 0.0:
                load_shed += bus.Pr_PFAC / bus.PL
        voltage_violation /= len(self.system.dbus)
        load_shed = math.exp(-load_shed) - 1
        voltage_violation = math.exp(-voltage_violation) - 1         
        
        # -----------------------------------------
        # SW and PV buses
        for bus in self.system.dbus_gstat:
            if(bus.nu_available_current > 0):
                QG_SW_PV_violation += ((max(0, bus.QG_FAC - bus.QG_max_current) + max(0, bus.QG_min_current - bus.QG_FAC))) / (bus.QG_max_current - bus.QG_min_current)
            else:
                QG_SW_PV_violation += (max(0, bus.QG_FAC - bus.QG_max_current) + max(0, bus.QG_min_current - bus.QG_FAC)) / (bus.QG_max - bus.QG_min)
        QG_SW_PV_violation /= len(self.system.dgstat)        
        QG_SW_PV_violation = math.exp(-QG_SW_PV_violation) - 1
            
        # -----------------------------------------
        # SW bus
        PG_SW_violation = max(0, self.system.bus_sw_current.PG_FAC - self.system.bus_sw_current.PG_max_current, self.system.bus_sw_current.PG_min_current - self.system.bus_sw_current.PG_FAC)
        if(self.system.bus_sw_current.nu_available_current > 0 and self.system.bus_sw_current.PG_max != self.system.bus_sw_current.PG_min):
            PG_SW_violation /= (self.system.bus_sw_current.PG_max - self.system.bus_sw_current.PG_min) * self.system.bus_sw_current.nu_available_current        
        PG_SW_violation = math.exp(-PG_SW_violation) - 1

        # -----------------------------------------
        # Circuit information
        for cir in self.system.dcir:
            circuit_overload += max(0.0, max(abs(cir.Sij_FAC), abs(cir.Sji_FAC)) - cir.cap_n) / cir.cap_n
        circuit_overload /= len(self.system.dcir)
        
        circuit_overload = math.exp(-circuit_overload) - 1

        
        reward = 1.0 + (    load_shed
                          + voltage_violation 
                          + circuit_overload
                          + PG_SW_violation 
                          + QG_SW_PV_violation) / 1.0

        self.last_reward_info = {
                "load_shed":        load_shed,
                "voltage":          voltage_violation,
                "circuit":          circuit_overload,
                "sw_active":        PG_SW_violation,
                "reactive":         QG_SW_PV_violation,
            }

        return reward

    # ------------------------------------------------------------------
    # Method to compute reward - OPF - For pretraining (only LS in this case)
    def _compute_reward_opf(self, success):

        if not success:
            return -10.0

        load_shed = 0.0

        # -----------------------------------------
        # Bus information
        for bus in self.system.dbus:
            if bus.PL > 0.0:
                load_shed += bus.Pr_OPFAC / bus.PL

        load_shed = math.exp(-load_shed) - 1

        reward = (load_shed) / 1.0

        return reward    
    
    # ------------------------------------------------------------------
    # Method to build observation (state)
    def _build_observation(self):
        
        # Node features 
        f = self._node_feat_buf
        m = self._mask_buf
        f[:] = 0.0
        m[:] = 0.0
        
        for i, bus in enumerate(self.system.dbus):
            
            # Load 
            if bus.PL > 0.0:
                f[i, 0] = bus.PL_current / bus.PL
                f[i, 1] = bus.QL_current / bus.QL

            # Operating point
            f[i, 2] = bus.V_FAC / bus.Vmax
            f[i, 3] = math.sin(bus.Theta_FAC * math.pi / 180.0)

            # Reactive-power margins (signed) 
            if bus.type in ("PV", "SW") and bus.nu_available_current > 0:
                dQ = bus.QG_max_current - bus.QG_min_current
                if dQ <= 0.0:
                    dQ = bus.QG_max - bus.QG_min          # fallback to nominal range
                if dQ > 0.0:
                    f[i, 4] = (bus.QG_max_current - bus.QG_FAC) / dQ   # upper Q margin
                    f[i, 5] = (bus.QG_FAC - bus.QG_min_current) / dQ   # lower Q margin

            # Active-power margins (signed)
            if bus.type in ("PV", "SW") and bus.nu_available_current > 0:
                dP = bus.PG_max_current - bus.PG_min_current
                if dP > 0.0:
                    f[i, 6] = (bus.PG_max_current - bus.PG_FAC) / dP   # upper P margin
                    f[i, 7] = (bus.PG_FAC - bus.PG_min_current) / dP   # lower P margin

            # Voltage margins (signed)
            dV = bus.Vmax - bus.Vmin
            if dV > 0.0:
                f[i, 8] = (bus.Vmax - bus.V_FAC) / dV    # upper V margin
                f[i, 9] = (bus.V_FAC - bus.Vmin) / dV    # lower V margin

            # Mask (note PG_max_current, consistent with _apply_action)
            m[i, 0] = 1.0 if bus.type in ("PV", "SW") else 0.0
            m[i, 1] = 1.0 if bus.type == "PV" and bus.PG_max_current > bus.PG_min_current else 0.0
            m[i, 2] = 1.0 if bus.PL > 0 else 0.0

        # Edge features
        self._refresh_edge_cache()

        eidx = 0
        for cir in self.system.dcir:
            if cir.available:
                self._edge_overload_buf[eidx, 0] = max(0.0, (abs(cir.Sij_FAC) - cir.cap_n) / cir.cap_n)
                eidx += 1
                self._edge_overload_buf[eidx, 0] = max(0.0, (abs(cir.Sji_FAC) - cir.cap_n) / cir.cap_n)
                eidx += 1

        overload_t = torch.from_numpy(self._edge_overload_buf)                    # zero-copy
        edge_attr  = torch.cat([overload_t, self._edge_static_cache], dim=1)      # [n_edges, 3]

        x     = torch.from_numpy(f.copy())
        mask  = torch.from_numpy(m.copy())

        return Data(
            x          = x,
            edge_index = self._edge_index_cache,   # reuse, no allocation
            edge_attr  = edge_attr,
            mask       = mask,
        )

    # ------------------------------------------------------------------
    # Method to refresh edge cache
    def _refresh_edge_cache(self):
        
        edges, statics = [], []
        for cir in self.system.dcir:
            if cir.available:
                i, j = cir.bF.id - 1, cir.bT.id - 1
                g = cir.g_l / self.system.glmax
                b = cir.b_l / self.system.blmax
                edges  += [[i, j], [j, i]]
                statics += [[g, b], [g, b]]
        
        n = len(edges)
        self._edge_index_cache  = torch.tensor(edges,   dtype=torch.long).t().contiguous()
        self._edge_static_cache = torch.tensor(statics, dtype=torch.float32)   # shape [n, 2]
        self._edge_overload_buf = np.zeros((n, 1), dtype=np.float32)
        self._n_edges_cache = n