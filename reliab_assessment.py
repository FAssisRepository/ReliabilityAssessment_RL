# =================================================================================== #
# RELIABILITY ASSESSMENT 
# =================================================================================== #
import os
import math
import random
import flow_AC
import OPF_AC
import DRL_reliab_env_1episode
from time import perf_counter
import numpy as np
from pathlib import Path
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
from torch.utils.data import DataLoader
from torch_geometric.nn import global_add_pool
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATv2Conv, global_mean_pool

# =================================================================================== #
# Class for flow DC with loss
class Reliab_assessment:

    # -----------------------------------------
    # Constructor
    def __init__(self, _data):
        # -----------------------------------------
        # Attributes
        self.NS = 0              # Sampled state counter (simulations)
        self.NS_discarded = 0    # Number of discarded states (optimal solution not found in OPF)
        self.N_flowDC = 0        # Number of flow DC runs
        self.N_flowAC = 0        # Number of flow AC runs
        self.N_LPDC = 0          # Number of LP DC runs
        self.N_OPF = 0           # Number of OPF runs
        self.N_GNN = 0           # Number of GNN predictions
        self.N_classifML = 0     # Number of ML classifications
        self.LOLP_G = 0.0        # LOLP index (generation insufficiency)
        self.LOLP_T = 0.0        # LOLP index (transmission insufficiency)
        self.LOLP = 0.0          # LOLP index
        self.beta_LOLP = 1.0     # Coefficient of variation - LOLP
        self.LOLE_G = 0.0        # LOLE index (generation insufficiency)
        self.LOLE_T = 0.0        # LOLE index (transmission insufficiency)
        self.LOLE = 0.0          # LOLE index
        self.EPNS_G = 0.0        # EPNS index (generation insufficiency)
        self.EPNS_T = 0.0        # EPNS index (transmission insufficiency)
        self.EPNS = 0.0          # EPNS index
        self.beta_EPNS = 1.0     # Coefficient of variation - EPNS
        self.EENS_G = 0.0        # EENS index (generation insufficiency)
        self.EENS_T = 0.0        # EENS index (transmission insufficiency)
        self.EENS = 0.0          # EENS index
        self.spent_time = 0.0    # Time - MCS crude
        self.spent_time_OPF = 0.0        # Mean time - DC-OPF
        self.spent_time_OPFAC = 0.0      # Mean time - AC-OPF
        self.spent_time_LPDC = 0.0       # Mean time - DC-LP
        self.spent_time_flowDC = 0.0     # Mean time - DC-flow
        self.spent_time_flowAC = 0.0     # Mean time - AC-flow
        self.spent_time_GNN = 0.0        # Mean time - GNN prediction
        self.spent_time_classifML = 0.0  # Mean time - ML classification
        self.spent_time_train = 0.0      # Time to train the ML
        # Auxiliar
        self.state_generationcapacity = 0.0                # Generation capacity of the current state
        self.state_load = 0.0                              # Total load of the current state
        self.states = []                                   # List of generated states
        self.store_states = False                          # Store states?
        random.seed(_data.simulation_set.seed)             # Defining seed for pseudorandom number generation
        self.current_state_seed_MCS = random.getstate()    # Current state seed for MCS - Random method
        # ML
        self.train_states = []   # List with states for ML training
        self.threshold = 0.5     # Threshold for state classification
        self.N_eval_ML = 0       # Number of ML model evaluation
        self.N_S_beforeML = 0    # Number of states sampled before using the ML model
        self.N_OPF_beforeML = 0  # Number of OPF runs before using the ML model
        self.N_OPF_withML = 0    # Number of OPF runs with using the ML model
        self.TP = 0              # True positive
        self.FP = 0              # False positive
        self.TN = 0              # True negative
        self.FN = 0              # False negative
        self.Accuracy = 0.0      # Accuracy
        self.Precision = 0.0     # Precision
        self.Recall = 0.0        # Recall
        self.F1Score = 0.0       # F1-Score
        self.Specificity= 0.0    # Specificity
        self.FPR = 0.0           # False Positive Rate
        self.n_elements_clusters_g = 0 # Number of generating station classes - new features
        self.n_elements_clusters_c = 0 # Number of circuit classes - new features
        # RL
        self.NS_RLtraining = 0   # Number of states used to train the RL agent
    
    # -----------------------------------------
    # Method to run Monte Carlo simulation - DRL - PPO-Lagrangian - GNN
    def run_MCS_DRL_PPO_GNN_Lagrangian_AC(self, _data, _contTest, _mainDir, _outputDir, _modelDir):
 
        def print_model_norm(model, tag):
            total = 0.0
            for p in model.parameters():
                total += p.data.norm().item()
            print(f"{tag} model norm: {total:.6f}")
 
        _mainDir = _data.simulation_set.mainDir
        _outputDir = _data.simulation_set.outputDir
        os.makedirs(os.path.join(_outputDir, f'Test-{_contTest + 1}'), exist_ok=True)
        file_name = f'training_dynamics_performance_{_contTest + 1}.txt'     # distinct log
        os.chdir(os.path.join(_outputDir, f'Test-{_contTest + 1}'))
        self.file = open(file_name, 'w')
        os.chdir(_mainDir)
 
        def log(msg):
            print(msg)
            self.file.write(msg + '\n')
 
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Device] Using: {device}")
        if device.type == "cuda":
            torch.backends.cudnn.benchmark = True
 
        # ==============================================================
        # Initial definitions 
        # ==============================================================
        system = _data.system
        _data.simulation_set.current_test = _contTest
        _data.apply_load_factor()
        _data.apply_gen_factor()
        _data.apply_circuit_factor()
 
        N_samples_PT  = _data.simulation_set.N_samples_PT[_contTest]
        n_epochs_PT   = _data.simulation_set.n_epochs_PT[_contTest]
        batch_size_PT = _data.simulation_set.batch_size_PT[_contTest]
        lr_actor_PT   = _data.simulation_set.lr_actor_PT[_contTest]
        lr_critic_PT  = _data.simulation_set.lr_critic_PT[_contTest]
 
        lr_actor_RL     = _data.simulation_set.lr_actor_RL[_contTest]
        lr_critic_RL    = _data.simulation_set.lr_critic_RL[_contTest]
        batch_size_RL   = _data.simulation_set.batch_size_RL[_contTest]
        n_episodes      = _data.simulation_set.n_episodes[_contTest]
        entropy_coef_RL = _data.simulation_set.entropy_coef_RL[_contTest]
        ppo_epochs      = _data.simulation_set.ppo_epochs[_contTest]
        clip_param      = _data.simulation_set.ppo_clip[_contTest]
 
        hidden_dim_actor  = _data.simulation_set.hidden_dim_actor[_contTest]
        n_heads_actor     = _data.simulation_set.n_heads_actor[_contTest]
        n_layers_actor    = _data.simulation_set.n_layers_actor[_contTest]
        hidden_dim_critic = _data.simulation_set.hidden_dim_critic[_contTest]
        n_heads_critic    = _data.simulation_set.n_heads_critic[_contTest]
        n_layers_critic   = _data.simulation_set.n_layers_critic[_contTest]
        n_eval_performancemetrics = _data.simulation_set.samples_performance[_contTest]
        dropout = _data.simulation_set.dropout[_contTest]
 
        alpha = 0.05
        n_eval = 10
 
        std_min = [0.0001, 0.0001, 0.0001]
        std_max = [0.1000, 0.2000, 0.1000]
        fine_tune_start_frac = 0.5
        std_min_start = std_min
        std_min_final = [1e-1 * x for x in std_min]
        std_max_start = std_max
        std_max_final = [1e-1 * x for x in std_max]
 
        # ==============================================================
        # LAGRANGIAN HYPERPARAMETERS 
        # ==============================================================
        n_costs       = 4
        cost_names    = ["V", "QG", "CL", "PG"]
 
        # Constraint thresholds d_k
        cost_thresh   = np.array([1e-5, 1e-5, 1e-5, 1e-5], dtype=np.float64)
 
        # Initial multipliers (>=0)
        lambda_init   = 1.0
        lambda_max    = 20.0                      
 
        # Dual (multiplier) 
        eta_lambda    = 1.0
 
        # EMA smoothing of the batch cost feeding the dual update 
        cost_ema_beta = 0.80
 
        lr_cost_critic = lr_critic_RL              
 
        lambdas   = np.full(n_costs, float(lambda_init), dtype=np.float64)
        cost_ema  = np.zeros(n_costs, dtype=np.float64)
 
        # ==============================================================
        # Seed + environment
        # ==============================================================
        self.set_global_seed(_data.simulation_set.seed_RL[_contTest], _data.simulation_set.seed)
        env = DRL_reliab_env_1episode.CompositeReliabilityEnv_1episode(_data)
 
        # Actor + OBJECTIVE critic (self.baseline) + per-constraint COST critics
        self.model = GNN_ACTOR(node_feat_dim=env.node_feat_dim, edge_feat_dim=env.edge_feat_dim,
                               hidden_dim=hidden_dim_actor, n_heads=n_heads_actor,
                               n_layers=n_layers_actor, std_min=std_min, std_max=std_max)
        self.baseline = GNN_Baseline(node_feat_dim=env.node_feat_dim, edge_feat_dim=env.edge_feat_dim,
                                     hidden_dim=hidden_dim_critic, n_heads=n_heads_critic,
                                     n_layers=n_layers_critic)
        cost_critics = [GNN_Baseline(node_feat_dim=env.node_feat_dim, edge_feat_dim=env.edge_feat_dim,
                                     hidden_dim=hidden_dim_critic, n_heads=n_heads_critic,
                                     n_layers=n_layers_critic) for _ in range(n_costs)]
 
        # AC-PF & AC-OPF base case (same as the fixed-weight method)
        flow_AC_obj = flow_AC.flow_AC_class()
        flow_AC_obj.run_flow_AC(_data)
        flow_AC_obj.print_flow_AC(_data)
        OPF_AC_obj = OPF_AC.OPF_AC_class()
        OPF_AC_obj.run_OPF_AC(_data)
        OPF_AC_obj.print_OPF_AC(_data)
 
        # ==============================================================
        # Initialisation: warm-start OR supervised pretraining
        # ==============================================================
        log("[Lagrangian] Pretraining GNN with SL (OPF targets)...")
        t_s = perf_counter()
        self.pretraining_MCS_SL_GNN_AC(
            _data, _contTest, _mainDir, _outputDir, env,
            N_samples=N_samples_PT, lr_ac=lr_actor_PT, lr_baseline=lr_critic_PT,
            batch_size=batch_size_PT, n_epochs=n_epochs_PT, device=device)
        log("[Pretrain] Spent time {:.2f} [s]".format(perf_counter() - t_s))
 
        # Pre-Lagrangian evaluation (same hooks as fixed-weight method)
        current_state_seed_MCS_TEST = self.current_state_seed_MCS
        self.set_global_seed(_data.simulation_set.seed_RL[_contTest] + 50, _data.simulation_set.seed + 50)
        self.test_model(_data, env, n_eval=n_eval, stage_name="PT",
                        print_OPF_results=True, device=device)
        self.test_model_performance_metrics(_data, env, n_eval=n_eval_performancemetrics,
                                            stage_name="PT", print_OPF_results=True, device=device)
        
        # Save PT Models
        pt_state_dict = {
            "model_state_dict": self.model.state_dict(),
            "model_config": {
                "node_feat_dim": env.node_feat_dim,
                "edge_feat_dim": env.edge_feat_dim,
                "hidden_dim": hidden_dim_actor,
                "n_heads": n_heads_actor,
                "n_layers": n_layers_actor,
                "action_dim": 3,
                "std_min": std_min,
                "std_max": std_max,
            }
        }
        # Save to the main model directory
        torch.save(pt_state_dict, os.path.join(_modelDir, "gnn_actor_PT.pt"))
        # Save a copy to the result test directory
        torch.save(pt_state_dict, os.path.join(_outputDir, f'Test-{_contTest + 1}', "gnn_actor_PT.pt"))
 
        # ==============================================================
        # Lagrangian PPO training
        # ==============================================================
        print_model_norm(self.model, "Before Lagrangian")
 
        self.model.to(device);    self.model.train()
        self.baseline.to(device); self.baseline.train()
        for c in cost_critics:
            c.to(device); c.train()
 
        optimizer          = optim.Adam(self.model.parameters(),    lr=lr_actor_RL)
        optimizer_baseline = optim.Adam(self.baseline.parameters(), lr=lr_critic_RL)
        cost_optimizers    = [optim.Adam(c.parameters(), lr=lr_cost_critic) for c in cost_critics]
 
        self.NS = 0; self.NS_discarded = 0; self.N_flowAC = 0; self.NS_RLtraining = 0
        episode = 0
        running_reward = None
 
        buffer_graphs, buffer_actions, buffer_log_probs = [], [], []
        buffer_combined, buffer_obj, buffer_costs = [], [], []   # combined (log only), objective, costs
 
        t_s_rltrain = perf_counter()
        scaler = torch.amp.GradScaler(device.type)
 
        while True:
            self.NS += 1
            n_unavailableG, n_unavailableC, _, _ = self.generate_new_state(_data)
            if n_unavailableG == system.n_total_generation_units:
                self.NS_discarded += 1; self.NS -= 1; continue
 
            success = env.flow_AC_obj.run_flow_AC(_data)
            if not success:
                self.NS_discarded += 1; self.NS -= 1; continue
 
            self.N_flowAC += 1
            env.flow_AC_obj.colect_flow_AC_results(_data)
            cv  = env.flow_AC_obj.circuit_violation
            vv  = env.flow_AC_obj.voltage_violation
            rv  = env.flow_AC_obj.reactive_violation
            pv  = env.flow_AC_obj.activeSW_violation
            ils = env.flow_AC_obj.island_load_shedding
 
            # Train only on samples that present operational issues
            if not (cv > 0.0 or vv > 0.0 or rv > 0.0 or pv > 0.0 or ils > 0.0):
                if episode >= n_episodes:
                    break
                continue
 
            self.NS_RLtraining += 1
            episode += 1
 
            # Reset (refreshes edge cache) and rollout the policy
            state_graph, _ = env.reset()
            state_graph = state_graph.to(device)
            mask = state_graph.mask
 
            with torch.no_grad():
                with torch.amp.autocast(device.type):
                    mean, std = self.model(state_graph)
                mean = mean.float(); std = std.float()
                dist = torch.distributions.Normal(mean, std)
                action = dist.sample()
                log_prob_node = dist.log_prob(action) * mask
                log_prob_per_sample = log_prob_node.sum().unsqueeze(0)
 
            action_clamped = torch.clamp(action, 0.0, 1.0)
            action_np = action_clamped.detach().cpu().numpy()
 
            # Step env (runs post-action PF, sets self.last_pf_success)
            _, combined_reward, _, _, _ = env.step(action_np)
 
            # Decompose into objective + per-constraint costs
            obj_reward, costs = env.compute_costs_pf()       
 
            buffer_graphs.append(state_graph)
            buffer_actions.append(action)
            buffer_log_probs.append(log_prob_per_sample)
            buffer_combined.append(float(combined_reward))
            buffer_obj.append(float(obj_reward))
            buffer_costs.append(np.asarray(costs, dtype=np.float64))
 
            # ==========================================================
            # PPO + DUAL UPDATE when the buffer is full
            # ==========================================================
            if len(buffer_obj) == batch_size_RL:
 
                # Linear LR decay (over the full run) 
                frac = max(0.0, 1.0 - (episode / n_episodes))
                cur_lr_actor  = max(1e-5, lr_actor_RL  * frac)
                cur_lr_critic = max(1e-5, lr_critic_RL * frac)
                for g in optimizer.param_groups:          g['lr'] = cur_lr_actor
                for g in optimizer_baseline.param_groups: g['lr'] = cur_lr_critic
                for opt in cost_optimizers:
                    for g in opt.param_groups:            g['lr'] = max(1e-5, lr_cost_critic * frac)
 
                # Batched tensors 
                old_log_probs   = torch.cat(buffer_log_probs).detach().float()
                batched_actions = torch.cat(buffer_actions).detach()
                super_batch     = Batch.from_data_list(buffer_graphs).to(device)
 
                obj_tensor   = torch.tensor(buffer_obj, dtype=torch.float32, device=device)           # [B]
                costs_np     = np.stack(buffer_costs, axis=0)                                         # [B, n_costs]
                costs_tensor = torch.tensor(costs_np, dtype=torch.float32, device=device)             # [B, n_costs]
                lambdas_t    = torch.tensor(lambdas, dtype=torch.float32, device=device)              # [n_costs]
 
                # Static advantages (computed once, like standard PPO)
                with torch.no_grad():
                    with torch.amp.autocast(device.type):
                        V_obj  = self.baseline(super_batch).squeeze(-1)
                        V_cost = torch.stack([c(super_batch).squeeze(-1) for c in cost_critics], dim=1)
                    V_obj  = V_obj.float()
                    V_cost = V_cost.float()
                    A_obj  = obj_tensor - V_obj                                  # [B]
                    A_cost = costs_tensor - V_cost                               # [B, n_costs]
                    # Lagrangian advantage: maximise objective, penalise costs
                    advantages = A_obj - (A_cost * lambdas_t).sum(dim=1)       # [B]
                    # advantages = A_obj - (costs_tensor * lambdas_t).sum(dim=1)         # [B]
                    if advantages.std() > 1e-8:
                        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
 
                # PPO epochs
                for ppo_it in range(ppo_epochs):
                    with torch.amp.autocast(device.type):
                        mean, std    = self.model(super_batch)
                        V_obj_fresh  = self.baseline(super_batch).squeeze(-1)
                        V_cost_fresh = torch.stack([c(super_batch).squeeze(-1) for c in cost_critics], dim=1)
 
                    mean = mean.float(); std = std.float()
                    V_obj_fresh  = V_obj_fresh.float()
                    V_cost_fresh = V_cost_fresh.float()
 
                    dist = torch.distributions.Normal(mean, std)
                    new_log_prob_node = dist.log_prob(batched_actions) * super_batch.mask
                    entropy_node      = dist.entropy() * super_batch.mask
                    if new_log_prob_node.dim() > 1:
                        new_log_prob_node = new_log_prob_node.sum(dim=-1)
                        entropy_node      = entropy_node.sum(dim=-1)
                    new_log_probs = global_add_pool(new_log_prob_node, super_batch.batch)
 
                    num_active_actions = super_batch.mask.sum()
                    mean_entropy = (dist.entropy() * super_batch.mask).sum() / (num_active_actions + 1e-8)
 
                    log_ratio = new_log_probs - old_log_probs
                    ratio     = torch.exp(log_ratio)
 
                    surr1 = ratio * advantages
                    surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantages
                    actor_loss = -torch.min(surr1, surr2).mean() - (entropy_coef_RL * mean_entropy)
 
                    baseline_loss = F.mse_loss(V_obj_fresh, obj_tensor)
                    cost_losses   = [F.mse_loss(V_cost_fresh[:, k], costs_tensor[:, k]) for k in range(n_costs)]
 
                    # Single combined backward
                    total_loss = actor_loss + baseline_loss + sum(cost_losses)
 
                    optimizer.zero_grad()
                    optimizer_baseline.zero_grad()
                    for opt in cost_optimizers:
                        opt.zero_grad()
 
                    scaler.scale(total_loss).backward()
 
                    # Unscale every optimizer before clipping
                    scaler.unscale_(optimizer)
                    scaler.unscale_(optimizer_baseline)
                    for opt in cost_optimizers:
                        scaler.unscale_(opt)
 
                    # Per-network gradient clipping
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    torch.nn.utils.clip_grad_norm_(self.baseline.parameters(), 1.0)
                    for k in range(n_costs):
                        torch.nn.utils.clip_grad_norm_(cost_critics[k].parameters(), 1.0)
 
                    # Step every optimizer
                    scaler.step(optimizer)
                    scaler.step(optimizer_baseline)
                    for opt in cost_optimizers:
                        scaler.step(opt)
 
                    scaler.update()
 
                # DUAL ASCENT on the multipliers (once per batch) - Exponential Moving Average (EMA)
                mean_costs = costs_np.mean(axis=0)                                  # [n_costs]
                cost_ema   = cost_ema_beta * cost_ema + (1.0 - cost_ema_beta) * mean_costs
                lambdas    = np.clip(lambdas + eta_lambda * (cost_ema - cost_thresh), 1.0, lambda_max)
 
                # Std annealing 
                frac_episode = episode / n_episodes
                if frac_episode > fine_tune_start_frac:
                    progress   = (frac_episode - fine_tune_start_frac) / (1.0 - fine_tune_start_frac)
                    cos_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
                    a_std_min = [std_min_final[j] + (std_min_start[j] - std_min_final[j]) * cos_factor for j in range(3)]
                    a_std_max = [std_max_final[j] + (std_max_start[j] - std_max_final[j]) * cos_factor for j in range(3)]
                    self.model.set_std_bounds(a_std_min, a_std_max)
 
                # Logging
                avg_combined = float(np.mean(buffer_combined))
                avg_obj      = float(np.mean(buffer_obj))
                if running_reward is None:
                    running_reward = avg_combined
                else:
                    running_reward = alpha * avg_combined + (1 - alpha) * running_reward
 
                lam_str  = "[" + ", ".join(f"{l:6.2f}" for l in lambdas) + "]"
                cost_str = "[" + ", ".join(f"{c:7.4f}" for c in mean_costs) + "]"
                log(f"[LAG] Ep {episode:5d} | R(comb): {avg_combined:7.4f} | Smooth: {running_reward:7.4f} | "
                    f"Obj: {avg_obj:7.4f} | Cost{cost_names}: {cost_str} | lam: {lam_str} | "
                    f"Actor L: {actor_loss.item():7.4f} | Ent: {mean_entropy.item():6.3f}")
 
                buffer_graphs, buffer_actions, buffer_log_probs = [], [], []
                buffer_combined, buffer_obj, buffer_costs = [], [], []
 
            if episode >= n_episodes:
                log(f"[Lagrangian] Training finished after {episode} episodes.")
                break
 
        print_model_norm(self.model, "After Lagrangian")
        log("[Lagrangian] Spent time {:.2f} [s]".format(perf_counter() - t_s_rltrain))
 
        # ==============================================================
        # Save models (distinct names; includes critics + final lambdas)
        # ==============================================================
        rl_state_dict = {
            "model_state_dict": self.model.state_dict(),
            "baseline_state_dict": self.baseline.state_dict(),
            "cost_critics_state_dict": [c.state_dict() for c in cost_critics],
            "lambdas": lambdas.tolist(),
            "cost_thresh": cost_thresh.tolist(),
            "model_config": {
                "node_feat_dim": env.node_feat_dim, "edge_feat_dim": env.edge_feat_dim,
                "hidden_dim": hidden_dim_actor, "n_heads": n_heads_actor,
                "n_layers": n_layers_actor, "action_dim": 3,
                "std_min": std_min, "std_max": std_max,
            }
        }
        torch.save(rl_state_dict, os.path.join(_modelDir, "gnn_actor_RL.pt"))
        torch.save(rl_state_dict, os.path.join(_outputDir, f'Test-{_contTest + 1}', "gnn_actor_RL.pt"))
 
        # ==============================================================
        # Evaluate after Lagrangian training
        # ==============================================================
        self.current_state_seed_MCS = current_state_seed_MCS_TEST
        self.set_global_seed(_data.simulation_set.seed_RL[_contTest] + 50, _data.simulation_set.seed + 50)
        self.test_model(_data, env, n_eval=n_eval, stage_name="RL",
                        print_OPF_results=False, device=device)
        self.test_model_performance_metrics(_data, env, n_eval=n_eval_performancemetrics,
                                            stage_name="RL", print_OPF_results=True, device=device)
 
        self.file.close()

    # -----------------------------------------
    # Method to pretrain the GNN - Monte Carlo simulation - Supervised Learning
    def pretraining_MCS_SL_GNN_AC(self, _data, _contTest, _mainDir, _outputDir, env, N_samples = 1000, lr_ac=1e-3, lr_baseline=1e-3, batch_size=64, n_epochs=500, device=None):
        
        system = _data.system
        start_time = perf_counter()
        _data.simulation_set.current_test = _contTest
        OPF_AC_obj = OPF_AC.OPF_AC_class()

        # To print
        def log(msg):
            print(msg)
            self.file.write(msg + '\n')

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
       
        # =================================================================
        # Sampling states
        # =================================================================        
        dataset = []
        NS_train = 0
        self.NS = 0
        self.N_OPF = 0
        self.N_flowAC = 0
        self.NS_discarded = 0
        violation_types = {"circuit_only": 0, "voltage_only": 0, "reactive_only": 0, "activeSW_only": 0, "mixed": 0,"n_unavailableG": 0, "n_unavailableC": 0}
        while(self.NS < 1000 * _data.simulation_set.NS_max):
            
            # -----------------------------------------
            # Sampling new state of the system
            self.NS +=  1
            n_unavailableG, n_unavailableC, unavailable_gstat, unavailable_circ = self.generate_new_state(_data)
            if(n_unavailableG == system.n_total_generation_units):
                self.NS_discarded += 1
                self.NS -= 1
                continue
            
            '''# -----------------------------------------
            # Skip samples with unavailable components
            if(n_unavailableG > 0 or n_unavailableC > 0):
                continue'''
            
            # -----------------------------------------
            # Evaluating the new state with AC power flow - without optimization
            success = env.flow_AC_obj.run_flow_AC(_data)
            #env.flow_AC_obj.print_flow_AC(_data)

            if not success:
                self.NS_discarded += 1
                self.NS -= 1
                continue            
             
            # Colecting power flow results
            self.N_flowAC += 1
            env.flow_AC_obj.colect_flow_AC_results(_data)
            circuit_violation = env.flow_AC_obj.circuit_violation
            voltage_violation = env.flow_AC_obj.voltage_violation
            reactive_violation = env.flow_AC_obj.reactive_violation
            activeSW_violation  = env.flow_AC_obj.activeSW_violation
            island_load_shedding = env.flow_AC_obj.island_load_shedding

            # -----------------------------------------
            # Skip samples with no violations
            if(circuit_violation > 0.0 or voltage_violation > 0.0 or reactive_violation > 0.0 or activeSW_violation > 0.0 or island_load_shedding > 0.0):

                # -----------------------------------------
                # Logging violations
                n_types = (circuit_violation > 0) + (voltage_violation > 0) + (reactive_violation > 0)
                if n_types > 1:
                    violation_types["mixed"] += 1
                elif circuit_violation > 0:
                    violation_types["circuit_only"] += 1
                elif voltage_violation > 0:
                    violation_types["voltage_only"] += 1
                elif activeSW_violation > 0:
                    violation_types["activeSW_only"] += 1
                else:
                    violation_types["reactive_only"] += 1
                if(n_unavailableG > 0): violation_types["n_unavailableG"] += 1
                if(n_unavailableC > 0): violation_types["n_unavailableC"] += 1

                # -----------------------------------------
                # Colecting state to DRL agent train
                NS_train += 1
                graph, _ = env.reset()
                mask = graph.mask

                # -----------------------------------------
                # Evaluating the state with AC OPF
                OPF_AC_obj.run_OPF_AC(_data)
                OPF_AC_obj.colect_flow_OPFAC_results(_data)
                self.N_OPF += 1

                '''OPF_AC_obj.run_OPF_AC(_data)
                OPF_AC_obj.print_OPF_AC(_data)'''

                # --------------------------
                # Target - OPF actions
                target_actions = []

                for bus in system.dbus:
                    V_adj = (bus.V_OPFAC - bus.Vmin) / (bus.Vmax - bus.Vmin) if bus.type in ['PV', 'SW'] else 0.0
                    if (bus.type in ['PV', 'SW'] and bus.PG_max_current > 0.0):
                        if(bus.PG_max_current != bus.PG_min_current):
                            Pg_adj = (bus.PG_OPFAC - bus.PG_min_current)  / (bus.PG_max_current - bus.PG_min_current)
                        else:
                            Pg_adj = (bus.PG_OPFAC)  / (bus.PG_max_current)
                    else:
                        Pg_adj = 0.0
                    Pr_adj = bus.Pr_OPFAC / bus.PL_current if bus.PL > 0 else 0.0
                    
                    target_actions.append([V_adj, Pg_adj, Pr_adj])

                graph.y = torch.tensor(target_actions, dtype=torch.float32)

                # --------------------------
                # Graph-level critic target (can be used in the future)
                reward_opf = env._compute_reward_opf(OPF_AC_obj.results["success"])
                graph.y_reward = torch.tensor([reward_opf], dtype=torch.float32)

                # --------------------------
                # Adding the graph to the dataset
                dataset.append(graph)

                if NS_train % 20 == 0:
                    log(f"[Pretrain] NS {self.NS:6d} | Samples for training: {NS_train:5d}")

            if(NS_train == N_samples):
                log(f"[Pretrain] NS {self.NS:6d} | Samples for training: {NS_train:5d}")
                break
        
        # Printing information about violations in dataset collection
        log(f"[Pretrain] Violation type distribution: {violation_types}")
        
        # =================================================================
        # Training the GNN - Supervised Learning
        # =================================================================
        self.model.to(device)
        self.model.train()
        self.baseline.to(device) 
        self.baseline.train()

        optimizer = optim.Adam(self.model.parameters(), lr_ac)
        optimizer_baseline = optim.Adam(self.baseline.parameters(), lr=lr_baseline)
        scheduler_actor    = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
        scheduler_baseline = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_baseline, mode='min', factor=0.5, patience=10, min_lr=1e-5)

        # Simple DataLoader
        g = torch.Generator()
        g.manual_seed(42)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, worker_init_fn=self.seed_worker, generator=g, pin_memory=(device.type == "cuda"))        

        for epoch in range(n_epochs):
            epoch_loss_actor    = 0.0
            epoch_loss_baseline = 0.0

            fist_batch = True
            for batch in loader:
                batch = batch.to(device)

                # --------------------------
                # Forward pass
                mean, _ = self.model(batch)
                action_mask = batch.mask
                
                # --------------------------
                # Loss
                diff = (mean - batch.y) * action_mask                # [n_bus_total, action_dim]
                
                num_active_actions = action_mask.sum()
                actor_loss = (diff ** 2).sum() / (num_active_actions + 1e-8)

                # --------------------------
                # Optimization - actor
                optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

                # --------------------------
                # Optimization - baseline
                value_pred  = self.baseline(batch)                    
                value_target = batch.y_reward.view(-1)               
                baseline_loss = F.mse_loss(value_pred, value_target)

                optimizer_baseline.zero_grad()
                baseline_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.baseline.parameters(), 1.0)
                optimizer_baseline.step()

                epoch_loss_actor    += actor_loss.item()
                epoch_loss_baseline += baseline_loss.item()

                if epoch == 0 and fist_batch:
                    print(f"Active action fraction: {action_mask.mean().item():.3f}")
                    fist_batch = False

            scheduler_actor.step()
            scheduler_baseline.step(epoch_loss_baseline / len(loader))
            
            if epoch % 20 == 0:
                log(f"[Pretrain] Epoch {epoch+1:4d}/{n_epochs:4d} | "f"Actor loss: {epoch_loss_actor/len(loader):12.6f} | "f"Baseline loss: {epoch_loss_baseline/len(loader):12.6f}")

        log(f"[Pretrain] Epoch {epoch+1:4d}/{n_epochs:4d} | "f"Actor loss: {epoch_loss_actor/len(loader):12.6f} | "f"Baseline loss: {epoch_loss_baseline/len(loader):12.6f}")
        log("[Pretrain] Finished!")        
        
    # -----------------------------------------
    # Method to run Monte Carlo simulation - AC-OPF-NS-MCS (benchmark)
    def run_MCS_OPF_AC(self, _data, _mainDir, _outputDir, _contTest):

        self.set_global_seed(_data.simulation_set.seed_RL[_contTest], _data.simulation_set.seed + 20)
        
        print(". \n. \n. \n. \n")
        print("[AC-OPF-NS-MCS] ...")
        
        # -----------------------------------------
        system = _data.system
        start_time = perf_counter()
        _data.simulation_set.current_test = _contTest
        _data.apply_load_factor()
        _data.apply_gen_factor()
        _data.apply_circuit_factor()
        self.store_states = False      # In a crude MCS execution, there is no need to store states

        # Defining seed
        self.set_global_seed(_data.simulation_set.seed_RL[_contTest], _data.simulation_set.seed + 20)
        env = DRL_reliab_env_1episode.CompositeReliabilityEnv_1episode(_data)

        _mainDir = _data.simulation_set.mainDir
        _outputDir = _data.simulation_set.outputDir
                
        # -----------------------------------------
        # Base case
        flow_AC_obj = flow_AC.flow_AC_class()    # Object for AC power flow
        '''flow_AC_obj.run_flow_AC(_data)
        #flow_AC_obj.print_flow_AC(_data)'''

        OPF_AC_obj = OPF_AC.OPF_AC_class()       # Object for AC-OPF
        '''OPF_AC_obj.run_OPF_AC(_data)
        #OPF_AC_obj.print_OPF_AC(_data)'''

        '''#flow_DC_obj.run_flow_DC(_data)
        #flow_DC_obj.print_flow_DC(_data)
        OPF_DC_obj = OPF_DC.OPF_DC_class()       # Object for OPF DC  
        if(_data.simulation_set.reoptimization[0] == 'True'):
            OPF_DC_obj.run_OPF_DC(_data)
            #self.N_OPF += 1
            #OPF_DC_obj.print_OPF_DC(_data)
            system.x0 = OPF_DC_obj.x0'''
        
        # -----------------------------------------
        # Initial conditions
        self.NS = 0                              # Sampled state counter (simulations)
        self.NS_discarded = 0                    # Number of discarded states (islanding system)
        AF_LOLP = 0.0                            # LOLP cumulative test function - general
        AF2_LOLP = 0.0                           # LOLP cumulative squared test function - general
        AF_EPNS = 0.0                            # EPNS cumulative test function - general
        AF2_EPNS = 0.0                           # EPNS cumulative squared test function - general
        for bus in system.dbus_load:
            bus.reset_values_reliability_indices()

        # -----------------------------------------
        # Simulation
        self.spent_time_OPFAC = 0.0    # Mean time - AC-flow
        self.spent_time_flowAC = 0.0   # Mean time - AC-flow
        self.spent_time_GNN = 0.0      # Mean time - GNN prediction
        self.N_flowAC = 0              # Number of flow AC runs
        self.N_OPF = 0                 # Number of OPF runs
        while(self.NS < _data.simulation_set.NS_max):
            
            # =================================================================
            # Sampling new state
            # =================================================================
            self.NS +=  1
            n_unavailableG, n_unavailableC, unavailable_gstat, unavailable_circ = self.generate_new_state(_data)
            if(n_unavailableG == system.n_total_generation_units):
                self.NS_discarded += 1
                self.NS -= 1
                continue

            # =================================================================
            # State adequacy assessment
            # =================================================================                        
            # -----------------------------------------
            # OPF evaluation
            '''for bus in system.dbus:
                bus.V_FAC = 1.0
                bus.Theta_FAC = 0.0
                bus.PG_FAC = bus.PG_desp_orig'''
            t_s_acopf = perf_counter()
            OPF_AC_obj.run_OPF_AC(_data)
            OPF_AC_obj.colect_flow_OPFAC_results(_data)
            t_e_acopf = perf_counter()
            self.N_OPF += 1
            self.spent_time_OPFAC += t_e_acopf - t_s_acopf
            #OPF_AC_obj.print_OPF_AC(_data, _file_name='-OPF-'+str(self.NS))
                            
            # =================================================================
            # Updating reliability indices
            # =================================================================
            load_shedding_total_OPF = OPF_AC_obj.loadshedding_total
            if(load_shedding_total_OPF > 0.000001):      
                AF_LOLP += 1
                AF2_LOLP += 1          # In fact, 1^2
                self.LOLP = AF_LOLP / self.NS
                AF_EPNS += load_shedding_total_OPF
                AF2_EPNS += math.pow(load_shedding_total_OPF, 2)
                self.EPNS = AF_EPNS / self.NS

                #OPF_AC_obj.print_OPF_AC(_data, _file_name='-FAIL-OPF-'+str(self.NS))

                # -----------------------------------------
                # Indices per bus:
                for bus in system.dbus_load:
                    if(bus.Pr_OPFAC >= 0.000001):
                        bus.AF_LOLP += 1
                        bus.AF2_LOLP += 1          # In fact, 1^2
                        bus.AF_EPNS += bus.Pr_OPFAC
                        bus.AF2_EPNS += math.pow(bus.Pr_OPFAC, 2)

                if(self.NS > _data.simulation_set.NS_min):
                    V_LOLP = ((AF2_LOLP - self.NS*(math.pow(self.LOLP, 2))) / (self.NS * (self.NS - 1)))      
                    self.beta_LOLP = (math.sqrt(V_LOLP)) / (AF_LOLP / self.NS) 
                    V_EPNS = ((AF2_EPNS - self.NS*(math.pow(self.EPNS, 2))) / (self.NS * (self.NS - 1)))      
                    self.beta_EPNS = (math.sqrt(V_EPNS)) / (AF_EPNS / self.NS)

                    if(max(self.beta_LOLP, self.beta_EPNS) <= _data.simulation_set.tol and self.NS >= _data.simulation_set.NS_min):
                        break

            if(self.N_OPF % 100 == 0):
                self.LOLP = AF_LOLP / self.NS
                self.LOLE = self.LOLP * 8760
                self.EPNS = AF_EPNS / self.NS
                self.EENS = self.EPNS * 8760
                print('-----------------------------------------------------------------------------------------------')   
                print(' NS:           {:9d} '.format(self.NS))
                print(' NS OPF:       {:9d} '.format(self.N_OPF))
                print(' NS discarded: {:9d} '.format(self.NS_discarded))
                print('                    LOLP (pu):            LOLE (h/y):        EPNS (MW):            EENS (NWh/y):')
                print(' Composed:          {:.4e} ({:6.2f}%)  {:.4e}         {:.4e} ({:6.2f}%)  {:.4e} \n'.format(self.LOLP, self.beta_LOLP * 100, self.LOLE, self.EPNS, self.beta_EPNS * 100, self.EENS))
                    
        if(self.NS > 0):
            # -----------------------------------------
            # Composed indices:
            self.LOLP = AF_LOLP / self.NS
            self.LOLE = self.LOLP * 8760
            self.EPNS = AF_EPNS / self.NS
            self.EENS = self.EPNS * 8760
            print('-----------------------------------------------------------------------------------------------')   
            print(' NS:           {:9d} '.format(self.NS))
            print(' NS OPF:       {:9d} '.format(self.N_OPF))
            print(' NS discarded: {:9d} '.format(self.NS_discarded))
            print('                    LOLP (pu):            LOLE (h/y):        EPNS (MW):            EENS (NWh/y):')
            print(' Composed:          {:.4e} ({:6.2f}%)  {:.4e}         {:.4e} ({:6.2f}%)  {:.4e} \n'.format(self.LOLP, self.beta_LOLP * 100, self.LOLE, self.EPNS, self.beta_EPNS * 100, self.EENS))        
            # -----------------------------------------
            # Indices per bus:
            for bus in system.dbus_load:
                if(bus.AF_LOLP > 0.0):
                    bus.LOLP = bus.AF_LOLP / self.NS
                    bus.LOLE = bus.LOLP * 8760
                    bus.EPNS = bus.AF_EPNS / self.NS
                    bus.EENS = bus.EPNS * 8760
                    V_LOLP = ((bus.AF2_LOLP - self.NS*(math.pow(bus.LOLP, 2))) / (self.NS * (self.NS - 1)))      
                    bus.beta_LOLP = (math.sqrt(V_LOLP)) / (bus.AF_LOLP / self.NS) 
                    V_EPNS = ((bus.AF2_EPNS - self.NS*(math.pow(bus.EPNS, 2))) / (self.NS * (self.NS - 1)))      
                    bus.beta_EPNS = (math.sqrt(V_EPNS)) / (bus.AF_EPNS / self.NS)
        
        end_time = perf_counter()
        self.spent_time = end_time - start_time

    # -----------------------------------------
    # Method to print results of Monte Carlo simulation - AC-OPF-NS-MCS (benchmark)
    def print_results_MCS_OPF_AC(self, _data, _mainDir, _outputDir):

        os.chdir(_outputDir)

        # Creating file
        system = _data.simulation_set.system_assessment[0]
        file_name = '{}_{}_OPF_NS_MCS.txt'.format(_data.simulation_set.current_test + 1, system)
        file =  open(file_name, 'w')

        file.write('-----------------------------------\n')
        file.write('MONTE CARLO SIMULATION - CRUDE_OPF-AC\n')
        file.write('-----------------------------------\n')
        file.write(' System: {}\n'.format(_data.simulation_set.system_assessment[0]))
        file.write(' Region of Assessment: {}\n'.format(_data.simulation_set.region_assessment))
        file.write(' Initial seed: {:6d}\n'.format(_data.simulation_set.seed))
        file.write(' Tol: {:.2f}\n\n'.format(_data.simulation_set.tol))

        file.write('-----------------------------------------\n')
        file.write('SIMULATION INFORMATION:\n')
        file.write('-----------------------------------------\n')
        file.write(' Spent time:   {:.2f} [sec]\n'.format(self.spent_time))
        if(self.N_OPF > 0):
            file.write(' Mean time AC-OPF:   {:.4e} [sec]\n'.format(self.spent_time_OPFAC/self.N_OPF))
        file.write(' Number of sampled states:   {:9d}\n'.format(self.NS))
        file.write(' Number of failure states:   {:9d}\n'.format(round(self.NS * self.LOLP)))
        file.write(' Number of success states:   {:9d}\n'.format((self.NS - round(self.NS * self.LOLP))))
        file.write(' Number of OPFs:             {:9d}\n'.format(self.N_OPF)) 
        file.write(' Number of discarded states: {:9d}\n'.format(self.NS_discarded)) 
        file.write('-----------------------------------------\n\n') 
        
        file.write('-----------------------------------------------------------------------------------------------\n')
        file.write('RELIABILITY INDICES:\n')
        file.write('-----------------------------------------------------------------------------------------------\n')
        file.write('                    LOLP (pu):            LOLE (h/y):        EPNS (MW):            EENS (NWh/y):\n')
        file.write(' Composed:          {:.4e}            {:.4e}         {:.4e}            {:.4e} \n'.format(self.LOLP, self.LOLE, self.EPNS, self.EENS))
        file.write('                    ({:5.2f}%)                                 ({:5.2f}%)  \n'.format(self.beta_LOLP * 100, self.beta_EPNS * 100))
        file.write('-----------------------------------------------------------------------------------------------\n')
        file.write('      BUS           LOLP (pu):            LOLE (h/y):        EPNS (MW):            EENS (NWh/y):\n')
        for bus in _data.system.dbus_load:
            file.write('  {:8d}          {:.4e} ({:6.2f}%)  {:.4e}         {:.4e} ({:6.2f}%)  {:.4e} \n'.format(bus.number, bus.LOLP, bus.beta_LOLP * 100, bus.LOLE, bus.EPNS, bus.beta_EPNS * 100, bus.EENS))
        file.write('-----------------------------------------------------------------------------------------------\n')

        print('\n\n------------------------------')
        print('MONTE CARLO SIMULATION - CRUDE')
        print('------------------------------')
        print(' System: {}'.format(_data.simulation_set.system_assessment[0]))
        print(' Region of Assessment: {}'.format(_data.simulation_set.region_assessment))
        print(' Initial seed: {:6d}'.format(_data.simulation_set.seed))
        print(' Tol: {:.2f}\n'.format(_data.simulation_set.tol))

        print('-----------------------------------------')
        print('SIMULATION INFORMATION:')
        print('-----------------------------------------')
        print(' Spent time:   {:.2f} [sec]'.format(self.spent_time))
        if(self.N_OPF > 0):
            print(' Mean time AC-OPF:   {:.4e} [sec]'.format(self.spent_time_OPFAC/self.N_OPF))
        print(' Number of sampled states:   {:9d}'.format(self.NS))
        print(' Number of failure states:   {:9d}'.format(int(self.NS * self.LOLP)))
        print(' Number of success states:   {:9d}'.format(int(self.NS - (self.NS * self.LOLP))))
        print(' Number of OPFs:             {:9d}'.format(self.N_OPF)) 
        print(' Number of discarded states: {:9d}'.format(self.NS_discarded)) 
        print('-----------------------------------------\n') 
        
        print('-----------------------------------------------------------------------------------------------')
        print('RELIABILITY INDICES:')
        print('-----------------------------------------------------------------------------------------------')
        print('                    LOLP (pu):            LOLE (h/y):        EPNS (MW):            EENS (NWh/y):')
        print(' Composed:          {:.4e} ({:6.2f}%)  {:.4e}         {:.4e} ({:6.2f}%)  {:.4e} '.format(self.LOLP, self.beta_LOLP * 100, self.LOLE, self.EPNS, self.beta_EPNS * 100, self.EENS))
        print('-----------------------------------------------------------------------------------------------\n')

        # Closing file
        file.close()
        os.chdir(_mainDir)

    # -----------------------------------------
    # Method to run Monte Carlo simulation - GNN model + OPF - AC (BATCHED INFERENCE) - RL-GNN-NS-MCS or SL-GNN-NS-MCS
    def run_MCS_GNN_OPF_AC(self, _data, _mainDir, _outputDir, _contTest, _modelDir, model_fase):

        def load_gnn_actor_for_testing(model_path, device):

            # Load the dictionary. 
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)            
            # Extract the configuration and initialize the model
            config = checkpoint["model_config"]            
            # We use **config to unpack the dictionary directly into the __init__ arguments
            model = GNN_ACTOR(**config)            
            # Load the saved weights (state_dict) into the model
            model.load_state_dict(checkpoint["model_state_dict"])            
            # Move to the correct device
            model.to(device)            
            # Set the model to evaluation mode.
            model.eval()
            
            return model

        def _save_snapshot(data, load_level, cv, vv, rv, ils):

            bus_snap = {}
            for bus in data.system.dbus:
                bus_snap[bus.id] = {
                    'isolated':              bus.isolated,
                    'type_current':          bus.type_current,
                    'nu_available_current':  bus.nu_available_current, 
                    'PL_current':            bus.PL_current,
                    'QL_current':            bus.QL_current,
                    'PG_desp_current':       bus.PG_desp_current,
                    'PG_max_current':        bus.PG_max_current,
                    'PG_min_current':        bus.PG_min_current,
                    'QG_max_current':        bus.QG_max_current,
                    'QG_min_current':        bus.QG_min_current,    
                }
            gstat_snap = {}
            for gstat in data.system.dgstat:
                gstat_snap[gstat.id] = {
                    'factor_disp':           gstat.factor_disp,
                    'nu_available':          gstat.nu_available,
                }
            cir_snap = {}
            for cir in data.system.dcir:
                cir_snap[cir.id] = {
                    'available': cir.available,
                }
            return {
                'bus':                  bus_snap,
                'cir':                  cir_snap,
                'gstat':                gstat_snap,
                'load_level':           load_level,
                'circuit_violation':    cv,
                'voltage_violation':    vv,
                'reactive_violation':   rv,
                'island_load_shedding': ils,
                'SW_bus':               data.system.bus_sw_current,
            }

        def _restore_snapshot(data, snap):

            data.system.bus_sw_current = snap['SW_bus'] 
            for bus in data.system.dbus:
                s = snap['bus'][bus.id]
                bus.isolated             = s['isolated']
                bus.type_current         = s['type_current']
                bus.nu_available_current = s['nu_available_current']
                bus.PL_current           = s['PL_current']
                bus.QL_current           = s['QL_current']
                bus.PG_desp_current      = s['PG_desp_current']
                bus.PG_max_current = s['PG_max_current']
                bus.PG_min_current = s['PG_min_current']
                bus.QG_max_current = s['QG_max_current']
                bus.QG_min_current = s['QG_min_current']
            for gstat in data.system.dgstat:
                s = snap['gstat'][gstat.id]
                gstat.factor_disp  = s['factor_disp']
                gstat.nu_available = s['nu_available']
            for cir in data.system.dcir:
                s = snap['cir'][cir.id]
                cir.available = s['available']             
        
        self.set_global_seed(_data.simulation_set.seed_RL[_contTest], _data.simulation_set.seed + 20)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Device] Using: {device}")
        if device.type == "cuda":
            torch.backends.cudnn.benchmark = True

        print(". \n. \n. \n. \n")
        print("[GNN-NS-MCS] ...")
        
        system = _data.system
        start_time = perf_counter()
        _data.simulation_set.current_test = _contTest
        _data.apply_load_factor()
        _data.apply_gen_factor()
        _data.apply_circuit_factor()
        self.store_states = False      

        _mainDir = _data.simulation_set.mainDir
        _outputDir = _data.simulation_set.outputDir
        outputDir_test = os.path.join(_outputDir, f'Test-{_contTest + 1}')
        env = DRL_reliab_env_1episode.CompositeReliabilityEnv_1episode(_data)

        # Loading model for deployment
        model_file_path = os.path.join(_modelDir, "gnn_actor_"+model_fase+".pt")
        # Assuming load_gnn_actor_for_testing is moved outside
        self.model = load_gnn_actor_for_testing(model_file_path, device)
                
        flow_AC_obj = flow_AC.flow_AC_class()    
        OPF_AC_obj = OPF_AC.OPF_AC_class()      
        
        self.NS = 0                              
        self.NS_discarded = 0                    
        AF_LOLP = 0.0                            
        AF2_LOLP = 0.0                           
        AF_EPNS = 0.0                            
        AF2_EPNS = 0.0                           
        
        for bus in system.dbus_load:
            bus.reset_values_reliability_indices()

        self.spent_time_OPFAC = 0.0    
        self.spent_time_flowAC = 0.0   
        self.spent_time_GNN = 0.0      
        self.N_flowAC = 0              
        self.N_OPF = 0                 
        self.N_GNN = 0                 

        batch_size = 100
        pending = []                     
        self.NS_processed = 0            
        break_simulation = False

        # Cache convergence parameters to avoid dot notation lookups in tight loops
        ns_max = _data.simulation_set.NS_max
        ns_min = _data.simulation_set.NS_min
        tol = _data.simulation_set.tol
        n_bus = env.n_bus

        while self.NS < ns_max and not break_simulation:
            
            # Sample and collect data
            while len(pending) < batch_size and self.NS < ns_max:
                n_unavailableG, n_unavailableC, unavailable_gstat, unavailable_circ = self.generate_new_state(_data)
                if n_unavailableG == system.n_total_generation_units:
                    self.NS_discarded += 1
                    continue

                t_s_flowAC = perf_counter()
                success = env.flow_AC_obj.run_flow_AC(_data)
                #env.flow_AC_obj.print_flow_AC(_data)
                self.spent_time_flowAC += perf_counter() - t_s_flowAC
                self.N_flowAC += 1

                if not success:
                    self.NS_discarded += 1
                    continue            
                 
                self.NS += 1
                
                env.flow_AC_obj.colect_flow_AC_results(_data)
                circuit_violation = env.flow_AC_obj.circuit_violation
                voltage_violation = env.flow_AC_obj.voltage_violation
                reactive_violation = env.flow_AC_obj.reactive_violation
                activeSW_violation  = env.flow_AC_obj.activeSW_violation
                island_load_shedding = env.flow_AC_obj.island_load_shedding

                load_level = _data.system.current_total_load / _data.system.peak_load

                # If no violation, bypass GNN
                if not (circuit_violation > 0.0 or voltage_violation > 0.0 or reactive_violation > 0.0 or activeSW_violation > 0.0 or island_load_shedding > 0.000001):       
                    continue

                t_s_GNN = perf_counter()
                graph, info = env.reset()
                self.spent_time_GNN += perf_counter() - t_s_GNN

                # Assuming _save_snapshot is moved outside
                snap = _save_snapshot(_data, load_level, circuit_violation, voltage_violation, reactive_violation, island_load_shedding)
                pending.append((graph, snap))

            if not pending:
                continue

            # Batched GNN forward pass
            pyg_batch = Batch.from_data_list([g for g, _ in pending]).to(device)

            t_s_GNN = perf_counter()
            with torch.no_grad():
                mean_batch, _ = self.model(pyg_batch)
            self.spent_time_GNN += perf_counter() - t_s_GNN

            # Mask and clamp entirely on GPU
            actions_batch = (mean_batch * pyg_batch.mask).clamp(0.0, 1.0)
            
            # Move to CPU but DO NOT flatten yet - This keeps the shape as (Total_Nodes_in_Batch, action_dim)
            actions_numpy_all = actions_batch.cpu().numpy()

            # Process results sequentially
            for k, (graph, snap) in enumerate(pending):
                
                self.NS_processed += 1
                _restore_snapshot(_data, snap)

                action_k = actions_numpy_all[k * n_bus : (k + 1) * n_bus].flatten()

                t_s_GNN = perf_counter()
                obs, reward_DRL, done, truncated, info = env.step(action_k)
                self.spent_time_GNN += perf_counter() - t_s_GNN
                self.N_GNN += 1

                if (env.flow_AC_obj.max_Pr > 0.005 or 
                    env.flow_AC_obj.max_V_viol > 0.001 or 
                    env.flow_AC_obj.max_PGSW_viol > 0.000 or 
                    env.flow_AC_obj.max_QG_viol > 0.001 or 
                    env.flow_AC_obj.max_Sij_viol > 0.001):
                    
                    load_level = snap['load_level']
                    for bus in _data.system.dbus:
                        bus.PL_current = load_level * bus.PL
                    
                    t_s_acopf = perf_counter()
                    OPF_AC_obj.run_OPF_AC(_data)
                    self.spent_time_OPFAC += perf_counter() - t_s_acopf
                    self.N_OPF += 1
                                    
                    load_shedding_total_OPF = OPF_AC_obj.loadshedding_total
                    if load_shedding_total_OPF > 0.000001:      
                        AF_LOLP += 1
                        AF2_LOLP += 1          
                        AF_EPNS += load_shedding_total_OPF
                        AF2_EPNS += load_shedding_total_OPF * load_shedding_total_OPF 

                        for bus in system.dbus_load:
                            if bus.Pr_OPFAC >= 0.000001:
                                bus.AF_LOLP += 1
                                bus.AF2_LOLP += 1          
                                bus.AF_EPNS += bus.Pr_OPFAC
                                bus.AF2_EPNS += bus.Pr_OPFAC * bus.Pr_OPFAC

                        #env.flow_AC_obj.print_flow_AC(_data, _file_name='FAIL-GNN-'+str(self.NS))
                        #OPF_AC_obj.print_OPF_AC(_data, _file_name='-FAIL-OPF-GNN-'+str(self.NS))

                if self.N_GNN % 100 == 0:
                    self.LOLP = AF_LOLP / self.NS
                    self.EPNS = AF_EPNS / self.NS
                    self.LOLE = self.LOLP * 8760
                    self.EENS = self.EPNS * 8760
                    print('-----------------------------------------------------------------------------------------------')   
                    print(' NS:           {:9d} '.format(self.NS))
                    print(' NS GNN:       {:9d} '.format(self.N_GNN))
                    print(' NS OPF:       {:9d} '.format(self.N_OPF))
                    print(' NS discarded: {:9d} '.format(self.NS_discarded))
                    print('                    LOLP (pu):            LOLE (h/y):        EPNS (MW):            EENS (NWh/y):')
                    print(' Composed:          {:.4e} ({:6.2f}%)  {:.4e}         {:.4e} ({:6.2f}%)  {:.4e} \n'.format(self.LOLP, self.beta_LOLP * 100, self.LOLE, self.EPNS, self.beta_EPNS * 100, self.EENS))

            pending = []
            if self.NS > ns_min:
                
                self.LOLP = AF_LOLP / self.NS
                self.EPNS = AF_EPNS / self.NS

                V_LOLP = ((AF2_LOLP - self.NS*(math.pow(self.LOLP, 2))) / (self.NS * (self.NS - 1)))      
                self.beta_LOLP = (math.sqrt(V_LOLP)) / (AF_LOLP / self.NS) if AF_LOLP > 0 else 1.0
                V_EPNS = ((AF2_EPNS - self.NS*(math.pow(self.EPNS, 2))) / (self.NS * (self.NS - 1)))      
                self.beta_EPNS = (math.sqrt(V_EPNS)) / (AF_EPNS / self.NS) if AF_EPNS > 0.0 else 1.0

                if(max(self.beta_LOLP, self.beta_EPNS) <= _data.simulation_set.tol and self.NS >= _data.simulation_set.NS_min):
                    break

        # Finalization & Per-Bus Metric Compilation
        final_ns = self.NS if self.NS > 0 else 1
        self.LOLP = AF_LOLP / final_ns
        self.LOLE = self.LOLP * 8760
        self.EPNS = AF_EPNS / final_ns
        self.EENS = self.EPNS * 8760

        for bus in system.dbus_load:
            if bus.AF_LOLP > 0.0:
                bus.LOLP = bus.AF_LOLP / final_ns
                bus.LOLE = bus.LOLP * 8760
                bus.EPNS = bus.AF_EPNS / final_ns
                bus.EENS = bus.EPNS * 8760
                V_LOLP = ((bus.AF2_LOLP - final_ns * (bus.LOLP * bus.LOLP)) / (final_ns * (final_ns - 1)))      
                bus.beta_LOLP = (math.sqrt(V_LOLP)) / (bus.AF_LOLP / final_ns) 
                V_EPNS = ((bus.AF2_EPNS - final_ns * (bus.EPNS * bus.EPNS)) / (final_ns * (final_ns - 1)))      
                bus.beta_beta_EPNS = (math.sqrt(V_EPNS)) / (bus.AF_EPNS / final_ns)
        
        end_time = perf_counter()
        self.spent_time = end_time - start_time    

    # -----------------------------------------
    # Method to print results of Monte Carlo simulation - RL-GNN-NS-MCS or SL-GNN-NS-MCS
    def print_results_MCS_GNN_OPF_AC(self, _data, _mainDir, _outputDir, _versionGNN):
    
        # Changing the directory 
        outputDir_test = _outputDir+'\Test-'+str( _data.simulation_set.current_test + 1)
        path = Path(outputDir_test)
        if path.is_dir():
            print("Directory exists")
        else:
            print("Directory doesn't exist. Creating...")
            os.makedirs(outputDir_test)
        os.chdir(outputDir_test)

        # Creating file
        system = _data.simulation_set.system_assessment[0]
        file_name = '{}_{}_{}_GNN_NS_MCS.txt'.format(_data.simulation_set.current_test + 1, system, _versionGNN)
        file =  open(file_name, 'w')

        file.write('------------------------------------------\n')
        file.write('MONTE CARLO SIMULATION - GNN-{} - OPF-AC\n'.format(_versionGNN))
        file.write('------------------------------------------\n')
        file.write(' System: {}\n'.format(_data.simulation_set.system_assessment[0]))
        file.write(' Region of Assessment: {}'.format(_data.simulation_set.region_assessment))
        file.write(' Initial seed: {:6d}\n'.format(_data.simulation_set.seed))
        file.write(' Tol: {:.2f}\n\n'.format(_data.simulation_set.tol))

        file.write('-----------------------------------------\n')
        file.write('SIMULATION INFORMATION:\n')
        file.write('-----------------------------------------\n')
        file.write(' Spent time:   {:.2f} [sec]\n'.format(self.spent_time))
        if(self.N_flowAC > 0):
            file.write(' Mean time AC-PF:    {:.4e} [sec]\n'.format(self.spent_time_flowAC/self.N_flowAC))
        if(self.N_OPF > 0):
            file.write(' Mean time AC-OPF:   {:.4e} [sec]\n'.format(self.spent_time_OPFAC/self.N_OPF))
        if(self.N_GNN > 0):
            file.write(' Mean time GNN:      {:.4e} [sec]\n'.format(self.spent_time_GNN/self.N_GNN))
        file.write(' Number of sampled states:   {:9d}\n'.format(self.NS))
        file.write(' Number of failure states:   {:9d}\n'.format(round(self.NS * self.LOLP)))
        file.write(' Number of success states:   {:9d}\n'.format((self.NS - round(self.NS * self.LOLP))))
        file.write(' Number of flowACs:          {:9d}\n'.format(self.N_flowAC))  
        file.write(' Number of OPFs:             {:9d}\n'.format(self.N_OPF))
        file.write(' Number of GNN predictions:  {:9d}\n'.format(self.N_GNN)) 
        file.write(' Number of discarded states: {:9d}\n'.format(self.NS_discarded)) 
        file.write('-----------------------------------------\n\n')         
        file.write('-----------------------------------------------------------------------------------------------\n')
        file.write('RELIABILITY INDICES:\n')
        file.write('-----------------------------------------------------------------------------------------------\n')
        file.write('                    LOLP (pu):            LOLE (h/y):        EPNS (MW):            EENS (NWh/y):\n')
        file.write(' Composed:          {:.4e}            {:.4e}         {:.4e}            {:.4e} \n'.format(self.LOLP, self.LOLE, self.EPNS, self.EENS))
        file.write('                    ({:5.2f}%)                                 ({:5.2f}%)  \n'.format(self.beta_LOLP * 100, self.beta_EPNS * 100))
        file.write('-----------------------------------------------------------------------------------------------\n')
        file.write('      BUS           LOLP (pu):            LOLE (h/y):        EPNS (MW):            EENS (NWh/y):\n')
        for bus in _data.system.dbus_load:
            file.write('  {:8d}          {:.4e} ({:6.2f}%)  {:.4e}         {:.4e} ({:6.2f}%)  {:.4e} \n'.format(bus.number, bus.LOLP, bus.beta_LOLP * 100, bus.LOLE, bus.EPNS, bus.beta_EPNS * 100, bus.EENS))
        file.write('-----------------------------------------------------------------------------------------------\n')

        print('-----------------------------------------')
        print('MONTE CARLO SIMULATION - GNN-{} - OPF-AC'.format(_versionGNN))
        print('-----------------------------------------')
        print(' System: {}'.format(_data.simulation_set.system_assessment[0]))
        print(' Region of Assessment: {}'.format(_data.simulation_set.region_assessment))
        print(' Initial seed: {:6d}'.format(_data.simulation_set.seed))
        print(' Tol: {:.2f}\n'.format(_data.simulation_set.tol))

        print('-----------------------------------------')
        print('SIMULATION INFORMATION:')
        print('-----------------------------------------')
        print(' Spent time:   {:.2f} [sec]'.format(self.spent_time))
        if(self.N_flowAC > 0):
            print(' Mean time AC-PF:    {:.4e} [sec]'.format(self.spent_time_flowAC/self.N_flowAC))
        if(self.N_OPF > 0):
            print(' Mean time AC-OPF:   {:.4e} [sec]'.format(self.spent_time_OPFAC/self.N_OPF))
        if(self.N_GNN > 0):
            print(' Mean time GNN:      {:.4e} [sec]'.format(self.spent_time_GNN/self.N_GNN))
        print(' Number of sampled states:   {:9d}'.format(self.NS))
        print(' Number of failure states:   {:9d}'.format(int(self.NS * self.LOLP)))
        print(' Number of success states:   {:9d}'.format(int(self.NS - (self.NS * self.LOLP))))
        print(' Number of flowACs:          {:9d}'.format(self.N_flowAC))  
        print(' Number of OPFs:             {:9d}'.format(self.N_OPF)) 
        print(' Number of GNN predictions:  {:9d}'.format(self.N_GNN))
        print(' Number of discarded states: {:9d}'.format(self.NS_discarded)) 
        print('-----------------------------------------\n') 
        
        print('-----------------------------------------------------------------------------------------------')
        print('RELIABILITY INDICES:')
        print('-----------------------------------------------------------------------------------------------')
        print('                    LOLP (pu):            LOLE (h/y):        EPNS (MW):            EENS (NWh/y):')
        print(' Composed:          {:.4e} ({:6.2f}%)  {:.4e}         {:.4e} ({:6.2f}%)  {:.4e} '.format(self.LOLP, self.beta_LOLP * 100, self.LOLE, self.EPNS, self.beta_EPNS * 100, self.EENS))
        print('-----------------------------------------------------------------------------------------------\n')

        # Closing file
        file.close()
        os.chdir(_mainDir)

    # -----------------------------------------
    # Method to generate a new state of the system
    def generate_new_state(self, _data):

        #random.setstate(self.current_state_seed_MCS)
        sim = _data.simulation_set
        sys = _data.system         

        unavailable_gstat = {'gstat':[], 'n_u': []}        # Dictionary with unavailable generating stations
        unavailable_circ = {'circ':[], 'n_u': []}          # Dictionary with unavailable circuits
        
        # -----------------------------------------
        # Load state:
        x = random.randint(0, len(_data.loadcurve) - 1)
        load_level = _data.loadcurve[x] / 100.0
        self.state_load = 0.0 
        for bus in sys.dbus_load:
            bus.PL_current = load_level * bus.PL
            bus.QL_current = load_level * bus.QL
            self.state_load += bus.PL_current        
        sys.current_total_load        = self.state_load
        
        for bus in sys.dbus_gstat:
            bus.PG_max_current = 0.0
            bus.PG_min_current = 0.0
            bus.QG_max_current = 0.0
            bus.QG_min_current = 0.0
            bus.nu_available_current = 0.0
            bus.cap_slack = 0.0
        
        # -----------------------------------------
        # Generating units:       
        n_unavailableG = 0             # Total number of unavailable generating units          
        self.state_generationcapacity = 0.0
        for gstat in sys.dgstat:
            bus = gstat.bus            
            gstat.factor_disp = 0.0
            bus.V = bus.V_orig
            bus.type_current = bus.type
            gstat.nu_available = 0               # Defining all generating units initially unavailable
            gstat.generationCapacity = 0.0       # Defining generation capacity null
            x = random.random()
            # Checking availability in the current state and updating station dispatch
            for s in range(gstat.nu, 0, -1):                    
                if(x > gstat.state_space[s - 1]):
                    gstat.nu_available = s
                    cap = s * gstat.P_max
                    self.state_generationcapacity += cap
                    gstat.generationCapacity += cap
                    if(gstat.nu_available < gstat.nu):
                        shortfall = gstat.nu - s
                        unavailable_gstat['gstat'].append(gstat)
                        unavailable_gstat['n_u'].append(shortfall)
                        n_unavailableG += shortfall
                    break
            if(gstat.nu_available == 0):
                nu_unavailable = gstat.nu - gstat.nu_available
                unavailable_gstat['gstat'].append(gstat)
                unavailable_gstat['n_u'].append(nu_unavailable)
                n_unavailableG += nu_unavailable
                bus.type_current = "PQ"

            nu_available = gstat.nu_available
            bus.PG_max_current += nu_available * gstat.P_max
            bus.PG_min_current += nu_available * gstat.P_min
            bus.QG_max_current += nu_available * gstat.Q_max
            bus.QG_min_current += nu_available * gstat.Q_min
            bus.nu_available_current += nu_available 

        # Adjusting current generation dispatch to the load level and available generation units
        total_cap_slack = 0.00         # Total capacity slack of generation buses   
        PG_desp_total = 0.0 
        for bus in sys.dbus_gstat:

            desp = bus.PG_desp * load_level
            pmax = bus.PG_max_current
            pmin = bus.PG_min_current

            if desp < pmin:
                desp = pmin
            elif desp > pmax:
                desp = pmax

            bus.PG_desp_current = desp
            slack = pmax - desp
            bus.cap_slack = slack
            PG_desp_total  += desp
            total_cap_slack += slack

        # Iteratively distribute missing generation across buses with slack
        for _ in range(2):
            missing = self.state_load - PG_desp_total
            if missing <= 1e-6 or total_cap_slack < missing:
                break

            PG_desp_total = 0.0
            new_total_cap_slack = 0.0
            for bus in sys.dbus_gstat:
                if bus.cap_slack > 0.0:
                    delta = (bus.cap_slack / total_cap_slack) * missing 
                    bus.PG_desp_current = min(bus.PG_desp_current + delta, bus.PG_max_current)
                    bus.cap_slack = bus.PG_max_current - bus.PG_desp_current
                PG_desp_total += bus.PG_desp_current
                new_total_cap_slack += bus.cap_slack

            total_cap_slack = new_total_cap_slack

        # Dispatch factors
        for bus in sys.dbus_gstat:
            inv_pmax = 1.0 / bus.PG_max_current if bus.PG_max_current > 0.0 else 0.0
            for gstat in bus.gstat:
                gstat.factor_disp = gstat.nu_available * gstat.P_max * inv_pmax
        
        # -----------------------------------------
        # Circuits:
        for bus in sys.dbus:
            bus.isolated = True
            bus.cir_available = 0
        n_unavailableC = 0   # Total number of unavailable circuits
        for cir in sys.dcir:
            cir.available = True
            x = random.random()
            if(x < cir.FOR):
                cir.available = False
                n_unavailableC += 1
                unavailable_circ['circ'].append(cir)
                unavailable_circ['n_u'].append(1)
            if(cir.available):
                cir.bF.isolated = False
                cir.bT.isolated = False
                cir.bF.cir_available += 1
                cir.bT.cir_available += 1
        
        # -----------------------------------------
        # Swing bus check:
        sw = sys.bus_sw
        if(sw.nu_available_current > 0 and not sw.isolated):
            sw.type_current = sw.type
            sys.bus_sw_current = sw
        else:
            sw.type_current = "PQ"
            best_bus = sys.dbus_gstat[0]
            best_slack = -1.0
            for bus in sys.dbus_gstat:
                if(bus.nu_available_current > 0 and bus.cap_slack > best_slack and not bus.isolated):
                    best_bus = bus
                    best_slack = bus.cap_slack
            best_bus.type_current = "SW"
            sys.bus_sw_current = best_bus

        # -----------------------------------------
        # Returning the current state of the random generator
        self.current_state_seed_MCS = random.getstate()

        return n_unavailableG, n_unavailableC, unavailable_gstat, unavailable_circ    
           
    # -----------------------------------------
    # Method for current sample preparation
    def sample_preparation(self, _state, _features):

        amostra = np.concatenate((np.array([_state.F_load_level]), np.array([_state.F_generation_reserveDeficit]), np.array(_state.F_generationCapacityArea),
                                  np.array([_state.F_IF]), np.array([_state.F_IS]), np.array(_state.F_IUgk), np.array(_state.F_IUck), np.array(_state.F_IAg), np.array(_state.F_IAc))) 
        # Selecting only the features to be used for ML application
        x = np.array(amostra)
        x = x[_features]
        x = np.array(x).reshape(1, -1)

        return x    

    # -----------------------------------------
    # Method for calculating performance metrics
    def classification_metrics(self):
        self.Accuracy = (self.TP + self.TN) / (self.TP + self.TN + self.FP + self.FN)
        self.Precision = (self.TP) / (self.TP + self.FP)
        self.Recall = (self.TP) / (self.TP + self.FN)
        self.F1Score = 2*(self.Recall) / (self.Recall + self.Precision)
        self.Specificity = (self.TN) / (self.TN + self.FP)
        self.FPR = (self.FP) / (self.TN + self.FP)
   
    # -----------------------------------------
    # Method to test the model
    def test_model(self, _data, env, n_eval=10, stage_name='', print_OPF_results=False, device=None):
        
        print('==================================')
        self.file.write('==================================\n')
        print('CURRENT GNN TEST - {} STATE EVALUATIONS:'.format(n_eval))
        self.file.write('CURRENT GNN TEST - {} STATE EVALUATIONS:\n'.format(n_eval))
        x = [random.random() for _ in range(3)]
        print(x)
        
        # -----------------------------------------
        # Testing the GNN
        OPF_AC_obj = OPF_AC.OPF_AC_class()
        rewards = []
        NS_samples = 0
        self.model.eval()
        cont_eval = 0
        cont_failure = 0
        min_failure = 0

        spent_time_acpf = 0.0
        spent_time_GNN = 0.0
        spent_time_acopf = 0.0

        while cont_eval < n_eval or cont_failure < min_failure:

            # -----------------------------------------
            # Sampling a new state
            n_unavailableG, n_unavailableC, unavailable_gstat, unavailable_circ = self.generate_new_state(_data)

            '''# -----------------------------------------
            # Skip samples with unavailable components
            if(n_unavailableG > 0 or n_unavailableC > 0):
                continue'''

            # -----------------------------------------
            # Evaluating the new state with AC power flow - without optimization
            NS_samples += 1
            t_s_acpf = perf_counter()
            success = env.flow_AC_obj.run_flow_AC(_data)
            t_e_acpf = perf_counter()
            spent_time_acpf += t_e_acpf - t_s_acpf
            #self.flow_AC_obj.print_flow_AC(_data)

            if not success:
                continue            
            else: 
                # Colecting power flow results
                env.flow_AC_obj.colect_flow_AC_results(_data)
                circuit_violation = env.flow_AC_obj.circuit_violation
                voltage_violation = env.flow_AC_obj.voltage_violation
                reactive_violation = env.flow_AC_obj.reactive_violation
                activeSW_violation  = env.flow_AC_obj.activeSW_violation
                island_load_shedding = env.flow_AC_obj.island_load_shedding

                # -----------------------------------------
                # Skip samples with no violations
                if(circuit_violation > 0.0 or voltage_violation > 0.0 or reactive_violation > 0.0 or activeSW_violation > 0.0 or island_load_shedding > 0.0):

                    cont_eval += 1
                    #self.flow_AC_obj.print_flow_AC(_data)

                    if cont_eval == 4:
                        aaaa = 1

                    # -----------------------------------------
                    # GNN Prediction
                    t_s_GNN = perf_counter()
                    graph, info = env.reset()
                    graph = graph.to(device)
                    mask = graph.mask                 
                    with torch.no_grad():
                        #mean, std, value = self.model(graph)
                        mean, std = self.model(graph)
                        #action, log_prob = self.sample_action(mean, std, mask)
                        action = mean * mask
                        action = torch.clamp(action, 0.0, 1.0)
                        action_np = action.cpu().numpy().flatten()        # Convertint to numpy for Gym
                    obs, reward_DRL, done, truncated, info = env.step(action_np)
                    t_e_GNN = perf_counter()
                    spent_time_GNN += t_e_GNN - t_s_GNN
                    env.flow_AC_obj.print_flow_AC(_data, _file_name='-GNN-'+stage_name+'-'+str(cont_eval))

                    # -----------------------------------------
                    # OPF evaluation
                    load_level = _data.system.current_total_load / _data.system.peak_load
                    for bus in _data.system.dbus:
                        bus.V = bus.V_orig
                        bus.PL_current = load_level * bus.PL
                    t_s_acopf = perf_counter()
                    OPF_AC_obj.run_OPF_AC(_data)
                    OPF_AC_obj.colect_flow_OPFAC_results(_data)
                    t_e_acopf = perf_counter()

                    if not OPF_AC_obj.results["success"]:
                        aaaa = 1

                    spent_time_acopf += t_e_acopf - t_s_acopf
                    reward_OPF = env._compute_reward_opf(OPF_AC_obj.results["success"])
                    if(print_OPF_results):
                        OPF_AC_obj.print_OPF_AC(_data, _file_name='-'+str(cont_eval))

                    if(OPF_AC_obj.loadshedding_total > 0.0):
                        cont_failure += 1

                    print("{:2d} - Reward GNN: {:10.6f}  |  Reward OPF: {:10.6f}".format(cont_eval, reward_DRL, reward_OPF))
                    self.file.write("{:2d} - Reward GNN: {:10.6f}  |  Reward OPF: {:10.6f}\n".format(cont_eval, reward_DRL, reward_OPF))

        print('==================================')
        print('MEAN TIMES FOR {} STATE EVALUATIONS:'.format(n_eval))
        print('AC-OPF: {:.4e}'.format(spent_time_acopf / n_eval))
        print('GNN:    {:.4e}'.format(spent_time_GNN / n_eval)) 
        print(' ---------------------------------')  
        print('AC-PF:  {:.4e} - NS samples: {}'.format(spent_time_acpf / NS_samples, NS_samples))  
        print('==================================')
        self.file.write('==================================\n')
        self.file.write('MEAN TIMES FOR {} STATE EVALUATIONS:\n'.format(n_eval))
        self.file.write('AC-OPF: {:.4e}\n'.format(spent_time_acopf / n_eval))
        self.file.write('GNN:    {:.4e}\n'.format(spent_time_GNN / n_eval)) 
        self.file.write('----------------------------------\n')  
        self.file.write('AC-PF:  {:.4e} - NS samples: {}\n'.format(spent_time_acpf / NS_samples, NS_samples))  
        self.file.write('==================================\n') 
    
    # -----------------------------------------
    # Method to test the model - Performance Metrics
    def test_model_performance_metrics(self, _data, env, n_eval=1000, stage_name='', print_OPF_results=False, device=None):
        
        def _save_snapshot(data, load_level, cv, vv, rv, ils):
            bus_snap = {}
            for bus in data.system.dbus:
                bus_snap[bus.id] = {
                    'isolated':              bus.isolated,
                    'type_current':          bus.type_current,
                    'nu_available_current':  bus.nu_available_current, 
                    'PL_current':            bus.PL_current,
                    'QL_current':            bus.QL_current,
                    'PG_desp_current':       bus.PG_desp_current,
                    'PG_max_current':        bus.PG_max_current,
                    'PG_min_current':        bus.PG_min_current,
                    'QG_max_current':        bus.QG_max_current,
                    'QG_min_current':        bus.QG_min_current,    
                }
            gstat_snap = {}
            for gstat in data.system.dgstat:
                gstat_snap[gstat.id] = {
                    'factor_disp':           gstat.factor_disp,
                    'nu_available':          gstat.nu_available,
                }
            cir_snap = {}
            for cir in data.system.dcir:
                cir_snap[cir.id] = {
                    'available': cir.available,
                }
            return {
                'bus':                  bus_snap,
                'cir':                  cir_snap,
                'gstat':                gstat_snap,
                'load_level':           load_level,
                'circuit_violation':    cv,
                'voltage_violation':    vv,
                'reactive_violation':   rv,
                'island_load_shedding': ils,
                'SW_bus':               data.system.bus_sw_current,
            }

        def _restore_snapshot(data, snap):
            data.system.bus_sw_current = snap['SW_bus'] 
            for bus in data.system.dbus:
                s = snap['bus'][bus.id]
                bus.isolated             = s['isolated']
                bus.type_current         = s['type_current']
                bus.nu_available_current = s['nu_available_current']
                bus.PL_current           = s['PL_current']
                bus.QL_current           = s['QL_current']
                bus.PG_desp_current      = s['PG_desp_current']
                bus.PG_max_current = s['PG_max_current']
                bus.PG_min_current = s['PG_min_current']
                bus.QG_max_current = s['QG_max_current']
                bus.QG_min_current = s['QG_min_current']
            for gstat in data.system.dgstat:
                s = snap['gstat'][gstat.id]
                gstat.factor_disp  = s['factor_disp']
                gstat.nu_available = s['nu_available']
            for cir in data.system.dcir:
                s = snap['cir'][cir.id]
                cir.available = s['available'] 
        
        print('==================================')
        self.file.write('==================================\n')
        print('CURRENT GNN TEST - {} STATE EVALUATIONS:'.format(n_eval))
        self.file.write('CURRENT GNN TEST - {} STATE EVALUATIONS - PERFORMANCE METRICS:\n'.format(n_eval))   

        t_initial = perf_counter()

        batch_size = 200
        
        # -----------------------------------------
        # Performance metrics
        OPT_LS = 0.0
        FEAS_V = 0.0
        FEAS_PGSW = 0.0
        FEAS_QG = 0.0
        FEAS_SCIR = 0.0
        CONT_LS = 0
        CONT_V = 0
        CONT_PGSW = 0
        CONT_QG = 0
        CONT_SCIR = 0
        CONT_SAMPLES_V = 0
        CONT_SAMPLES_QG = 0
        CONT_SAMPLES_SCIR = 0
        CONT_SAMPLES_PGSW = 0
        CONT_MIX = 0
        cont_bettersolution = 0

        spent_time_acpf = 0.0
        spent_time_GNN = 0.0
        spent_time_acopf = 0.0
        
        # -----------------------------------------
        # Testing the GNN
        OPF_AC_obj = OPF_AC.OPF_AC_class()
        pending   = []   # list of (graph: Data, snap: dict)
        cont_eval = 0
        NS_samples = 0
        cont_failure = 0

        while cont_eval < n_eval:

            # -----------------------------------------
            # Collect up to batch_size samples with violations
            while len(pending) < batch_size and (cont_eval + len(pending)) < n_eval:

                self.generate_new_state(_data)
                NS_samples += 1

                t0 = perf_counter()
                success = env.flow_AC_obj.run_flow_AC(_data)
                spent_time_acpf += perf_counter() - t0

                if not success:
                    continue

                env.flow_AC_obj.colect_flow_AC_results(_data)
                cv  = env.flow_AC_obj.circuit_violation
                vv  = env.flow_AC_obj.voltage_violation
                rv  = env.flow_AC_obj.reactive_violation
                pv  = env.flow_AC_obj.activeSW_violation
                ils = env.flow_AC_obj.island_load_shedding

                if not (cv > 0.0 or vv > 0.0 or rv > 0.0 or pv > 0.0 or ils > 0.0):
                    continue   # no violations → skip (does NOT consume an eval slot)

                load_level = _data.system.current_total_load / _data.system.peak_load

                # Build graph — reads system state, does NOT modify it
                t0 = perf_counter()
                graph, _ = env.reset()
                spent_time_GNN += perf_counter() - t0   # obs-build component

                snap = _save_snapshot(_data, load_level, cv, vv, rv, ils)
                pending.append((graph, snap))

            if not pending:
                break   # no more valid samples available

            # -----------------------------------------
            # Single batched GNN forward pass for all pending samples
            pyg_batch = Batch.from_data_list([g for g, _ in pending]).to(device)

            t0 = perf_counter()
            with torch.no_grad():
                mean_batch, _ = self.model(pyg_batch)   # [B*n_bus, action_dim]
            spent_time_GNN += perf_counter() - t0       # inference component

            # Apply mask and clamp for all samples at once
            actions_batch = (mean_batch * pyg_batch.mask).clamp(0.0, 1.0)
            n_bus = env.n_bus

            # -----------------------------------------
            # Sequential per-sample evaluation (restore → step → OPF)
                        
            for k, (graph, snap) in enumerate(pending):
                if cont_eval >= n_eval:
                    break

                load_level = snap['load_level']

                # Restore system to post-initial-AC-PF state for this sample
                _restore_snapshot(_data, snap)

                # Extract this sample's action from the batched output
                action_k = actions_batch[k * n_bus : (k + 1) * n_bus].cpu().numpy().flatten()

                # Apply action + post-action AC-PF (same as original env.step)
                t0 = perf_counter()
                obs, reward_DRL, done, truncated, info = env.step(action_k)
                spent_time_GNN += perf_counter() - t0   # post-action PF component

                # GNN load-shedding metrics 
                loadshedding_total_GNN = 0.0
                bus_LS_greater5e3 = False
                for bus in _data.system.dbus_load:
                    if not bus.isolated:
                        bus.Pr_PFAC = max(0.0, bus.PL * load_level - bus.PL_current)
                    else:
                        bus.Pr_PFAC = bus.PL_current
                    if bus.Pr_PFAC > 0.005:
                        loadshedding_total_GNN += bus.Pr_PFAC
                        bus_LS_greater5e3 = True

                # Restore and run OPF for comparison 
                for bus in _data.system.dbus:
                    bus.V_FAC      = bus.V_orig
                    bus.PG_FAC     = bus.PG_desp_orig
                    bus.PL_current = load_level * bus.PL

                t0 = perf_counter()
                OPF_AC_obj.run_OPF_AC(_data)
                OPF_AC_obj.colect_flow_OPFAC_results(_data)
                spent_time_acopf += perf_counter() - t0

                if env._compute_reward_opf(OPF_AC_obj.results["success"]) < -9.9:
                    continue   # OPF infeasible — discard (does NOT consume eval slot)

                if OPF_AC_obj.loadshedding_total > 0.000001:
                    cont_failure += 1

                # Per-sample performance metrics (logic unchanged) 
                _restore_snapshot(_data, snap)          # back to clean state
                for bus in _data.system.dbus:
                    bus.PL_current = load_level * bus.PL
                env.step(action_k)
                viol = viol_V = viol_QG = viol_SCIR = viol_PGSW = False

                diff_LS = 0.0
                for bus in _data.system.dbus:
                    if bus.PL > 0.0:
                        OPT_LS += abs(bus.Pr_OPFAC - bus.Pr_PFAC)

                    V_viol = max(0.0, bus.Vmin - bus.V_FAC, bus.V_FAC - bus.Vmax)
                    if V_viol > 0.001:
                        FEAS_V += V_viol; CONT_V += 1
                        viol_V = viol = True

                for bus in _data.system.dbus_gstat:
                    QG_viol = max(0.0, bus.QG_min_current - bus.QG_FAC, bus.QG_FAC - bus.QG_max_current)
                    if QG_viol > 0.001:
                        FEAS_QG += QG_viol; CONT_QG += 1
                        viol_QG = viol = True

                for cir in _data.system.dcir:
                    Scir_viol = max(1.0, abs(cir.Sij_FAC) / cir.cap_n, abs(cir.Sji_FAC) / cir.cap_n)
                    if Scir_viol - 1.0 > 0.001:
                        FEAS_SCIR += max(0.0, abs(cir.Sij_FAC) - cir.cap_n, abs(cir.Sji_FAC) - cir.cap_n); CONT_SCIR += 1
                        viol_SCIR = viol = True

                bus_sw = _data.system.bus_sw_current
                sw_viol = max(0.0, bus_sw.PG_min_current - bus_sw.PG_FAC, bus_sw.PG_FAC - bus_sw.PG_max_current)                
                if sw_viol > 0.0:                          
                    FEAS_PGSW += sw_viol; CONT_PGSW += 1
                    viol_PGSW = viol = True

                if viol_V:    CONT_SAMPLES_V    += 1
                if viol_QG:   CONT_SAMPLES_QG   += 1
                if viol_SCIR: CONT_SAMPLES_SCIR += 1
                if viol_PGSW: CONT_SAMPLES_PGSW += 1

                if viol or bus_LS_greater5e3: CONT_MIX += 1
                if abs(loadshedding_total_GNN - OPF_AC_obj.loadshedding_total) > 0.01:
                    CONT_LS += 1

                # "GNN beats OPF" diagnostic (restore snapshot before re-step)
                if loadshedding_total_GNN < OPF_AC_obj.loadshedding_total and not viol:
                    cont_bettersolution += 1
                    OPF_AC_obj.print_OPF_AC(_data, _file_name=f'-OPF_BETTER-{cont_eval}')
                    _restore_snapshot(_data, snap)          # back to clean state
                    for bus in _data.system.dbus:
                        bus.PL_current = load_level * bus.PL
                    env.step(action_k)
                    env.flow_AC_obj.print_flow_AC(_data, _file_name=f'-GNN_BETTER-{stage_name}-{cont_eval}')

                cont_eval += 1
                if cont_eval % 100 == 0:
                    print(f'[PerformanceEvaluation] {cont_eval:6d} samples...')

            pending = []   # clear buffer; next outer iteration fills a new batch

        OPT_LS /= cont_eval
        FEAS_V /= cont_eval
        FEAS_PGSW /= cont_eval
        FEAS_QG /= cont_eval
        FEAS_SCIR /= cont_eval
        CONT_V /= cont_eval
        CONT_QG /= cont_eval
        CONT_SCIR /= cont_eval

        t_total = perf_counter() - t_initial

        print("LOAD SHEDDING (OPTIMALITY): \n  Absolute deviation = {:10.4f} [MW]  |  Number of samples where the GNN and AC-OPF solutions differ by more than 0.01 MW = {:5d}".format(OPT_LS, CONT_LS))
        print("FEASIBILITY: ")
        print("   Mean V error per sample (load buses) = {:10.4f} [pu]    |  Mean number of buses with violations per sample:           {:5.2f} ({:.2f}%)  |  Number of samples with V violation:   {:5d}".format(FEAS_V, CONT_V, CONT_V/len(_data.system.dbus_load), CONT_SAMPLES_V))
        print("   Mean QG error per sample (PV and SW) = {:10.4f} [MVAr]  |  Mean number of buses with violations per sample:           {:5.2f} ({:.2f}%)  |  Number of samples with QG violation:  {:5d}".format(FEAS_QG, CONT_QG, CONT_QG/len(_data.system.dbus_gstat), CONT_SAMPLES_QG))
        print("   Mean Sij error per sample            = {:10.4f} [MVA]   |  Mean number of circuits with violations per sample:        {:5.2f} ({:.2f}%)  |  Number of samples with Sij violation: {:5d}".format(FEAS_SCIR, CONT_SCIR, CONT_SCIR/len(_data.system.dcir), CONT_SAMPLES_SCIR))
        print("   PG error per sample - (SW bus)       = {:10.4f} [MW]                                                                                 |  Number of samples with PG violation:  {:5d}".format(FEAS_PGSW, CONT_SAMPLES_PGSW))
        print("   Sampes with at least one issue  = {:6d}".format(CONT_MIX))
        print(' ---------------------------------')
        print("Number of solutions where the GNN outperforms the AC-OPF solver: {:3d}".format(cont_bettersolution))
        print("Number of samples with load shedding: {:3d}".format(cont_failure))
        self.file.write("LOAD SHEDDING (OPTIMALITY): \n  Absolute deviation = {:10.4f} [MW]  |  Number of samples where the GNN and AC-OPF solutions differ by more than 0.01 MW = {:5d}\n".format(OPT_LS, CONT_LS))
        self.file.write("FEASIBILITY: \n")
        self.file.write("   Mean V error per sample (load buses) = {:10.4f} [pu]    |  Mean number of buses with violations per sample:           {:5.2f} ({:.2f}%)  |  Number of samples with V violation:   {:5d}\n".format(FEAS_V, CONT_V, CONT_V/len(_data.system.dbus_load), CONT_SAMPLES_V))
        self.file.write("   Mean QG error per sample (PV and SW) = {:10.4f} [MVAr]  |  Mean number of buses with violations per sample:           {:5.2f} ({:.2f}%)  |  Number of samples with QG violation:  {:5d}\n".format(FEAS_QG, CONT_QG, CONT_QG/len(_data.system.dbus_gstat), CONT_SAMPLES_QG))
        self.file.write("   Mean Sij error per sample            = {:10.4f} [MVA]   |  Mean number of circuits with violations per sample:        {:5.2f} ({:.2f}%)  |  Number of samples with Sij violation: {:5d}\n".format(FEAS_SCIR, CONT_SCIR, CONT_SCIR/len(_data.system.dcir), CONT_SAMPLES_SCIR))
        self.file.write("   PG error per sample - (SW bus)       = {:10.4f} [MW]                                                                                 |  Number of samples with PG violation:  {:5d}\n".format(FEAS_PGSW, CONT_SAMPLES_PGSW))
        self.file.write("   Sampes with at least one issue  = {:6d}\n".format(CONT_MIX))
        self.file.write("Number of solutions where the GNN outperforms the AC-OPF solver: {:3d}\n".format(cont_bettersolution))
        self.file.write("Number of samples with load shedding: {:3d}\n".format(cont_failure))
        print('==================================')
        print('MEAN TIMES FOR {} STATE EVALUATIONS:'.format(n_eval))
        print('AC-OPF: {:.4e}'.format(spent_time_acopf / n_eval))
        print('GNN:    {:.4e}'.format(spent_time_GNN / n_eval)) 
        print(' ---------------------------------')  
        print('AC-PF:  {:.4e} - NS samples: {}'.format(spent_time_acpf / NS_samples, NS_samples)) 
        print(' ---------------------------------')  
        print('Total eval time:  {:.4e}'.format(t_total))   
        print('==================================')
        self.file.write('==================================\n')
        self.file.write('MEAN TIMES FOR {} STATE EVALUATIONS:\n'.format(n_eval))
        self.file.write('AC-OPF: {:.4e}\n'.format(spent_time_acopf / n_eval))
        self.file.write('GNN:    {:.4e}\n'.format(spent_time_GNN / n_eval)) 
        self.file.write('----------------------------------\n')  
        self.file.write('AC-PF:  {:.4e} - NS samples: {}\n'.format(spent_time_acpf / NS_samples, NS_samples))  
        self.file.write(' ---------------------------------\n')  
        self.file.write('Total eval time:  {:.4e}\n'.format(t_total))
        self.file.write('==================================\n') 

    # -----------------------------------------
    # Methods to define seed
    def set_global_seed(self, seed_RL, seed_MCS):
        
        random.seed(seed_MCS)
        np.random.seed(seed_MCS)
        
        torch.manual_seed(seed_RL)
        torch.cuda.manual_seed(seed_RL)
        torch.cuda.manual_seed_all(seed_RL)
        
        # Deterministic GPU operations
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        # Force deterministic algorithms (PyTorch 1.8+)
        torch.use_deterministic_algorithms(True)

    # -----------------------------------------
    # Methods to define seed worker
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

# =================================================================================== #
# Class for system state
class System_state:

    # -----------------------------------------
    # Constructor
    def __init__(self, _areas, _NS):
        # -----------------------------------------
        # Attributes
        self.ID = _NS                            # State general ID
        self.ID_train = 0                        # State train ID
        self.OPFrun = False                      # Has the OPF been executed for this state?
        self.failure_state = False               # Failure state?
        self.correct_ML_classif = False          # Correct ML classification?
        self.n_unavailableG = 0                  # Total number of unavailable generating units
        self.n_unavailableC = 0                  # Total number of unavailable circuits
        self.load_shedding_G = 0.0               # Load shedding due to generation deficit (G failures)
        self.load_shedding_T = 0.0               # Load shedding due to insufficient transmission capacity (T failures)
        self.load_shedding_total = 0.0           # Total load shedding
        self.unavailable_gstat = {'gstat':[], 'n_u': []}   # Dictionary with unavailable generating stations
        self.unavailable_circ = {'circ':[], 'n_u': []}     # Dictionary with unavailable circuits
        self.prob_loadsheddingOPF = 0.0          
        # Features
        # Existent features
        self.F_load_level = 0.0                  # Feature: State load level
        self.F_generation_reserveDeficit = 0.0   # Feature: State generation reserve/deficit (i.e., the difference between the available generation capacity and the load to be served)
        self.F_generationCapacityArea = []       # Feature: State generation capacity per area (bus.area - bus object attibute)
        for area in _areas:
            self.F_generationCapacityArea.append(0.0)
        # New features
        self.F_IF = 0.0                          # Feature: State feature with the relationship between the unavailabilities of its components and system failure
        self.F_IS = 0.0                          # Feature: State feature with the relationship between the unavailabilities of its components and the non-occurrence of system operational failure
        self.F_IUgk = []                         # Feature: State feature with genartion unit outage information for each class, of the K classes defined using a clustering tool
        self.F_IUck = []                         # Feature: State feature with circuit outage information for each class, of the K classes defined using a clustering tool
        self.F_IAg = []                          # Feature: State feature with individual unavailability for generating stations - composed by ng elements (ng = number of generating stations)
        self.F_IAc = []                          # Feature: State feature with individual unavailability for for circuits - composed by nc elements (nc = number of circuits)

    # -----------------------------------------
    # Method to define state load level
    def set_load_level(self, _load_level):
        self.F_load_level = _load_level

    # -----------------------------------------
    # Method to define state generation reserve/deficit
    def set_generation_reserveDeficit(self, _state_generationcapacity, _state_load):
        self.F_generation_reserveDeficit = _state_generationcapacity - _state_load

    # -----------------------------------------
    # Method to define state number of unavailable generating units
    def set_n_unavailableG(self, _n_unavailableG):
        self.n_unavailableG = _n_unavailableG

    # -----------------------------------------
    # Method to define state number of unavailable circuits
    def set_n_unavailableC(self, _n_unavailableC):
        self.n_unavailableC = _n_unavailableC
    
    # -----------------------------------------
    # Method to define state generation capacity per area
    def set_generationCapacityArea(self, _dgstat):
        for gstat in _dgstat:
            self.F_generationCapacityArea[gstat.bus.area-1] += gstat.generationCapacity
    
    # -----------------------------------------
    # String representation while debugging Python
    def __repr__(self) -> str:
        return f"Person(id: {self.id}, un_G: {self.n_unavailableG}, un_C: {self.n_unavailableC}, OPFrun: {self.OPFrun}, Loadshedding: {self.load_shedding_total})"

# =============================================================================
# GATBlock — single GAT layer with residual + LayerNorm
class GATBlock(nn.Module):

    def __init__(self, in_dim, out_dim, n_heads, edge_dim, dropout=0.0, concat=True):
        super().__init__()
        self.concat = concat

        self.conv = GATv2Conv(
            in_channels=in_dim,
            out_channels=out_dim,
            heads=n_heads,
            edge_dim=edge_dim,       
            concat=concat,
            dropout=dropout,
            add_self_loops=True,     
            share_weights=False,     
        )

        conv_out_dim = out_dim * n_heads if concat else out_dim

        self.residual_proj = (
            nn.Linear(in_dim, conv_out_dim, bias=False)
            if in_dim != conv_out_dim
            else nn.Identity()
        )

        self.norm = nn.LayerNorm(conv_out_dim)
        self.act  = nn.ReLU()

    def forward(self, x, edge_index, edge_emb):
        out = self.conv(x, edge_index, edge_attr=edge_emb)  
        out = self.act(out + self.residual_proj(x))         
        out = self.norm(out)                                
        return out                                          

# =============================================================================
# GNNEncoder — shared backbone for actor and critic
class GNNEncoder(nn.Module):

    def __init__(self, node_feat_dim, edge_feat_dim, hidden_dim, n_heads, n_layers, dropout=0.0):
        super().__init__()
        assert hidden_dim % n_heads == 0, \
            f"hidden_dim ({hidden_dim}) must be divisible by n_heads ({n_heads})"
        assert n_layers >= 1

        # Edge projector (Shared across all layers)
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Input node projection
        self.input_proj = nn.Sequential(
            nn.Linear(node_feat_dim, hidden_dim),
            nn.ReLU(),
        )

        # GAT Layers
        layers = []
        for _ in range(n_layers):

            layers.append(GATBlock(
                in_dim=hidden_dim, 
                out_dim=hidden_dim // n_heads,
                n_heads=n_heads, 
                edge_dim=hidden_dim,
                dropout=dropout, 
                concat=True, 
            ))

        self.layers = nn.ModuleList(layers)

    def forward(self, data):
        x          = data.x           
        edge_index = data.edge_index  
        edge_attr  = data.edge_attr   

        # Project features
        edge_emb = self.edge_encoder(edge_attr)  
        x = self.input_proj(x)                   

        # Message passing
        for layer in self.layers:
            x = layer(x, edge_index, edge_emb)   

        return x  

# =============================================================================
# GNN_ACTOR (Policy Network)
class GNN_ACTOR(nn.Module):

    def __init__(
        self,
        node_feat_dim,
        edge_feat_dim,
        hidden_dim  = 128,
        n_heads     = 4,
        n_layers    = 3,
        action_dim  = 3,
        # Per-group bounds: [V, Pg, Pr]
        std_min     = (0.01,  0.02,  0.005),
        std_max     = (0.30,  0.40,  0.30 ),
        dropout     = 0.0,
    ):
        super().__init__()
        self.action_dim = action_dim

        # Register as buffers: shape [1, action_dim] for broadcasting
        self.register_buffer('std_min', torch.tensor(std_min, dtype=torch.float32).view(1, -1))
        self.register_buffer('std_max', torch.tensor(std_max, dtype=torch.float32).view(1, -1))

        self.encoder = GNNEncoder(
            node_feat_dim=node_feat_dim,
            edge_feat_dim=edge_feat_dim,
            hidden_dim=hidden_dim,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
        )

        self.mean_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim),
        )
        self.std_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim),
        )

        self._init_weights()

    def _init_weights(self):
        # Initialize mean head to output 0 (sigmoid(0) = 0.5)
        nn.init.xavier_uniform_(self.mean_head[0].weight)
        nn.init.zeros_(self.mean_head[0].bias)
        nn.init.xavier_uniform_(self.mean_head[2].weight, gain=0.01)
        nn.init.zeros_(self.mean_head[2].bias)

        # Initialize std head to output 0 
        nn.init.xavier_uniform_(self.std_head[0].weight)
        nn.init.zeros_(self.std_head[0].bias)
        nn.init.xavier_uniform_(self.std_head[2].weight, gain=0.01)
        nn.init.zeros_(self.std_head[2].bias)

    def set_std_bounds(self, std_min: list, std_max: list):
        floor = 1e-4
        self.std_min.copy_(torch.tensor([max(v, floor) for v in std_min], dtype=torch.float32))
        self.std_max.copy_(torch.tensor([max(v, floor) for v in std_max], dtype=torch.float32))

    def forward(self, data):
        node_emb = self.encoder(data)  

        # Mean calculation
        mean = torch.sigmoid(self.mean_head(node_emb))
        if hasattr(data, 'mask'):
            mean = mean * data.mask

        # Std calculation
        std_raw = self.std_head(node_emb)
        std = self.std_min + (self.std_max - self.std_min) * torch.sigmoid(std_raw)        
        if hasattr(data, 'mask'):
            std = std * data.mask + (1.0 - data.mask) * 1e-6

        return mean, std

# =============================================================================
# GNN_Baseline (Critic)
class GNN_Baseline(nn.Module):

    def __init__(
        self,
        node_feat_dim,
        edge_feat_dim,
        hidden_dim  = 128,
        n_heads     = 4,
        n_layers    = 3,
        dropout     = 0.0,
    ):
        super().__init__()

        self.encoder = GNNEncoder(
            node_feat_dim=node_feat_dim,
            edge_feat_dim=edge_feat_dim,
            hidden_dim=hidden_dim,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
        )

        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
        )

    def forward(self, data):
        node_emb  = self.encoder(data)                              
        graph_emb = global_mean_pool(node_emb, data.batch)         
        value     = self.value_head(graph_emb).squeeze(-1)         
        return value