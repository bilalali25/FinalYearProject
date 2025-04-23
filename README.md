# FinalYearProject

Overview

This repository contains the code and resources for my Final Year Project (FYP) at City University of Hong Kong, Department of Mechanical Engineering. The project, titled "Predicting Stacking Fault Energies in NiCoCr Alloys Using Machine Learning", investigates the use of Graph Neural Networks (GNNs) to predict stacking fault energy (SFE) in NiCoCr medium-entropy alloys. Two models were developed: ThermoGNN (a GAT-based model) and CGCNN (a convolutional GNN), trained on a dataset generated via LAMMPS molecular dynamics simulations. The project was supervised by Dr. Zhao Shijun and completed on April 24, 2025.

Key Objectives





Generate a dataset of NiCoCr alloy atomic configurations using LAMMPS.



Develop and train GNN models (ThermoGNN and CGCNN) to predict SFE.



Evaluate model performance and identify limitations for future improvements.

Repository Structure





data/: Contains precomputed graphs (graph_idx.pt), scalers (node_scaler.pkl, edge_scaler.pkl), and the SFE dataset (SFE.xlsx).



scripts/: Python scripts for data generation and visualization.





lammps_input.in: LAMMPS script for generating atomic configurations (Appendix 7.1).



extract_sfe.py: Extracts SFE values from LAMMPS output (Appendix 7.2).



plot_gaussian.py: Plots Gaussian distribution of SFE values (Appendix 7.3).



plot_gnn_architecture.py: Visualizes the ThermoGNN architecture (Appendix 7.5).



src/: Main code for model implementation and training.





train.py: Implements and trains the ThermoGNN model (Appendix 7.4).



results/: Stores model outputs, including predictions, metrics, and plots.



docs/: Contains the final report PDF (Bilal_Ali_57131705_Final_Report_FYP.pdf).

Installation





Clone the repository:

git clone https://github.com/bilalali25/FinalYearProject.git
cd FinalYearProject



Install dependencies:

pip install -r requirements.txt

Note: Requires Python 3.8+, PyTorch, PyTorch Geometric, ASE, Pymatgen, scikit-learn, pandas, and Graphviz. See requirements.txt for details.



Install LAMMPS for dataset generation (see LAMMPS documentation).

Usage

1. Dataset Generation

Generate atomic configurations and calculate SFE using LAMMPS:

lmp_serial < scripts/lammps_input.in

Extract SFE values:

python scripts/extract_sfe.py

2. Model Training

Train the ThermoGNN model using precomputed graphs:

python src/train.py

Note: Ensure data/ contains graph_idx.pt, node_scaler.pkl, and edge_scaler.pkl. Modify RUN_PREPROCESS_SCALERS, RUN_PREPROCESS_GRAPHS, and RUN_TRAINING flags in train.py as needed.

3. Visualization

Generate SFE distribution plots:

python scripts/plot_gaussian.py

Visualize the ThermoGNN architecture:

python scripts/plot_gnn_architecture.py

Results





The dataset includes 3,000 configurations (used 400 for training/testing) with 3,600 atoms each, at 33.33% Ni, Co, and Cr.



Both ThermoGNN and CGCNN showed limited accuracy due to dataset constraints (fixed composition, small size) and feature engineering challenges.



Detailed results, including training/validation losses, predicted vs. true SFE plots, and performance metrics (MAE, RMSE, R2), are in the results/ directory.

Future Improvements





Use larger, more diverse datasets with varied compositions.



Incorporate advanced feature engineering (e.g., physics-informed features).



Explore hybrid models combining GNNs with DFT/MD.

Citation

For more details, refer to the final report in docs/:





Bilal Ali (57131705), "Predicting Stacking Fault Energies in NiCoCr Alloys Using Machine Learning," Final Year Project Report, City University of Hong Kong, 2025.

Acknowledgments

Dr. Zhao Shijun for supervision and guidance.

Wenyu Lu for feedback on concepts, coding, and report review.



Family and friends for their support.

License

This project is licensed under the MIT License - see the LICENSE file for details.
