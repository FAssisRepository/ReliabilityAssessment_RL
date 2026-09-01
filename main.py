# =================================================================================== #
# Accelerated Composite Reliability Assessment Using a
# Reinforcement Learning-Driven Graph Neural Network Surrogate
# =================================================================================== #
import os
import auxiliar_classes
import input_data_class
import reliab_assessment

# =================================================================================== #
# Defining directories paths
dir = auxiliar_classes.handling_dir()
mainDir = dir.get_mainDir()
inputDir = dir.get_inputDir()
modelDir = dir.get_modelDir()
outputDir = dir.get_outputDir()
del dir

# =================================================================================== #
# Loading input data
data = input_data_class.inputData_system(mainDir, inputDir, outputDir)
os.chdir(mainDir)

# =================================================================================== #
# Reliability assessment
MCS_AC_OPF = reliab_assessment.Reliab_assessment(data)
MCS_GNN = reliab_assessment.Reliab_assessment(data)

# -----------------------------------------------------------------------
# AC-OPF-NS-MCS (benchmark)
MCS_AC_OPF.run_MCS_OPF_AC(data, mainDir, outputDir, 0)
MCS_AC_OPF.print_results_MCS_OPF_AC(data, mainDir, outputDir)

# -----------------------------------------------------------------------
# RL-GNN-NS-MCS or SL-GNN-NS-MCS
for contTest in range(0, len(data.simulation_set.testeID_RL)):

    # Training
    MCS_GNN.run_MCS_DRL_PPO_GNN_Lagrangian_AC(data, contTest, mainDir, outputDir, modelDir)

    # Deployment - RL-GNN-NS-MCS or SL-GNN-NS-MCS
    if(data.simulation_set.RA_GNN_PT_eval[contTest] == 'True'):
        MCS_GNN.run_MCS_GNN_OPF_AC(data, mainDir, outputDir, contTest, modelDir, 'PT')
        MCS_GNN.print_results_MCS_GNN_OPF_AC(data, mainDir, outputDir, 'PT')
    if(data.simulation_set.RA_GNN_RL_eval[contTest] == 'True'):
        MCS_GNN.run_MCS_GNN_OPF_AC(data, mainDir, outputDir, contTest, modelDir, 'RL')
        MCS_GNN.print_results_MCS_GNN_OPF_AC(data, mainDir, outputDir, 'RL')