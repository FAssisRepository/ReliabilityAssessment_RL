# =================================================================================== #
# COMPOSITE RELIABILITY ASSESSMENT
# FERNANDO ASSIS
# =================================================================================== #
import os
import auxiliar_classes
import input_data_class
import OPF_DC
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
if(data.simulation_set.system_assessment[0][-3:] == "_AC"):    

    # -----------------------------------------
    # Reliability Assessment
    MCS_GNN = reliab_assessment.Reliab_assessment(data)

    # -----------------------------------------
    # Tests - models
    for contTest in range(0, len(data.simulation_set.testeID_RL)):

        # -----------------------------------------
        # Training via RL - PPO
        MCS_GNN.run_MCS_DRL_PPO_GNN_Lagrangian_AC(data, contTest, mainDir, outputDir, modelDir)

        # -----------------------------------------
        # Reliability Assessment via RL-GNN
        if(data.simulation_set.RA_GNN_PT_eval[contTest] == 'True'):
            MCS_GNN.run_MCS_GNN_OPF_AC_PROPOSAL(data, mainDir, outputDir, contTest, modelDir, 'PT')
            MCS_GNN.print_results_MCS_GNN_OPF_AC_Buses_PROPOSAL(data, mainDir, outputDir, 'PT')
        if(data.simulation_set.RA_GNN_RL_eval[contTest] == 'True'):
            MCS_GNN.run_MCS_GNN_OPF_AC_PROPOSAL(data, mainDir, outputDir, contTest, modelDir, 'RL')
            MCS_GNN.print_results_MCS_GNN_OPF_AC_Buses_PROPOSAL(data, mainDir, outputDir, 'RL')