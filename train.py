# import os
# os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'  # Add before any torch imports

# import torch
# import torch.nn as nn
# import torch.optim as optim
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from torch_geometric.data import Data
# from torch_geometric.loader import DataLoader
# from torch_geometric.nn import GATConv, global_mean_pool
# from pymatgen.core.structure import Structure
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# from sklearn.model_selection import KFold
# from ase.io import read
# from pymatgen.io.ase import AseAtomsAdaptor
# import joblib
# import logging
# import time

# # Configure device
# device = torch.device("cpu")
# torch.set_default_device(device)

# # Configure logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# # ===================== 1. Configuration =====================
# class Config:
#     THERMO_DATA = {
#         28: {'sgte_lse': 2.89, 'allen_en': 1.88, 'ionization_energy': 737.1, 'specific_heat': 26.1, 'atomic_mass': 58.693, 'metallic_radius': 1.24, 'symbol': 'Ni'},
#         27: {'sgte_lse': -0.43, 'allen_en': 1.84, 'ionization_energy': 760.4, 'specific_heat': 24.8, 'atomic_mass': 58.933, 'metallic_radius': 1.25, 'symbol': 'Co'},
#         24: {'sgte_lse': -2.85, 'allen_en': 1.65, 'ionization_energy': 652.9, 'specific_heat': 23.3, 'atomic_mass': 51.996, 'metallic_radius': 1.28, 'symbol': 'Cr'}
#     }
#     for z, data in THERMO_DATA.items():
#         if 'symbol' not in data:
#             if z == 28: data['symbol'] = 'Ni'
#             elif z == 27: data['symbol'] = 'Co'
#             elif z == 24: data['symbol'] = 'Cr'
#             else: data['symbol'] = f'X{z}'
#     N_FOLDS = 3
#     EPOCHS = 300
#     EARLY_STOP_PATIENCE = 75
#     SFE_FILE = "temp_SFE.xlsx"
#     LAMMPS_DATA_DIR = "New_results"
#     LAMMPS_DATA_PREFIX = "NiCoCr_faulted_"
#     PRECOMPUTED_DIR = "precomputed_graphs"
#     SCALER_DIR = "scalers"
#     OUTPUT_DIR = "results/"
#     SF_Z_MIN = 15.0
#     SF_Z_MAX = 19.0

# config = Config()
# os.makedirs(config.OUTPUT_DIR, exist_ok=True)
# os.makedirs(config.PRECOMPUTED_DIR, exist_ok=True)
# os.makedirs(config.SCALER_DIR, exist_ok=True)

# # ===================== 2. Data Loading and Processing =====================
# class ThermoDataLoader:
#     def __init__(self, load_scalers=True):
#         self.node_scaler = None
#         self.edge_scaler = None
#         if load_scalers:
#             node_scaler_path = os.path.join(config.SCALER_DIR, "node_scaler.pkl")
#             edge_scaler_path = os.path.join(config.SCALER_DIR, "edge_scaler.pkl")
#             try:
#                 self.node_scaler = joblib.load(node_scaler_path)
#                 logging.info(f"Loaded node scaler from {node_scaler_path}")
#             except FileNotFoundError:
#                 logging.error(f"Node scaler not found at {node_scaler_path}. Run preprocess_scalers() first.")
#                 raise
#             except Exception as e:
#                 logging.error(f"Error loading node scaler: {e}", exc_info=True)
#                 raise
#             if os.path.exists(edge_scaler_path):
#                 try:
#                     self.edge_scaler = joblib.load(edge_scaler_path)
#                     logging.info(f"Loaded edge scaler from {edge_scaler_path}")
#                 except Exception as e:
#                     logging.error(f"Error loading edge scaler: {e}", exc_info=True)
#                     logging.warning("Proceeding without edge scaler due to loading error.")
#             else:
#                 logging.warning("Edge scaler file not found; edge features will not be scaled.")
#         else:
#             self.node_scaler = StandardScaler()
#             self.edge_scaler = StandardScaler()

#     def load_data(self, num_structures=200):
#         logging.info(f"Loading precomputed graphs...")
#         try:
#             data_SF = pd.read_excel(config.SFE_FILE)
#         except FileNotFoundError:
#             logging.error(f"SFE file not found: {config.SFE_FILE}")
#             raise
#         num_available = len(data_SF)
#         num_to_load = min(num_structures, num_available)
#         logging.info(f"Found {num_available} samples in {config.SFE_FILE}, attempting to load up to {num_to_load}")

#         graphs = []
#         loaded_indices = []
#         failed_load_count = 0
#         for idx in range(1, num_to_load + 1):
#             graph_path = os.path.join(config.PRECOMPUTED_DIR, f"graph_{idx}.pt")
#             try:
#                 if os.path.exists(graph_path):
#                     graph = torch.load(graph_path, map_location=device, weights_only=False)
#                     if not hasattr(graph, 'x') or not hasattr(graph, 'edge_index') or not hasattr(graph, 'y'):
#                         logging.warning(f"Graph {idx} from {graph_path} seems incomplete or corrupted. Skipping.")
#                         failed_load_count += 1
#                         continue
#                     if graph.x.device != device: graph = graph.to(device)
#                     graphs.append(graph)
#                     loaded_indices.append(idx)
#                 else:
#                     logging.debug(f"Precomputed graph {graph_path} not found. Skipping index {idx}.")
#                     failed_load_count += 1
#             except Exception as e:
#                 logging.error(f"Error loading graph {idx} from {graph_path}: {str(e)}", exc_info=True)
#                 failed_load_count += 1
#                 continue

#         if not graphs:
#             if failed_load_count == num_to_load and num_to_load > 0:
#                 error_msg = f"No precomputed graph files found in {config.PRECOMPUTED_DIR} (checked up to index {num_to_load}). Run graph preprocessing."
#             else:
#                 error_msg = f"No valid graphs loaded. Loaded {len(graphs)}, Failed/Missing: {failed_load_count}. Check precomputed files and logs in {config.PRECOMPUTED_DIR}."
#             raise ValueError(error_msg)

#         logging.info(f"Successfully loaded {len(graphs)} graphs.")
#         if loaded_indices:
#             logging.info(f" Indices loaded: {min(loaded_indices)}...{max(loaded_indices)}")
#         if failed_load_count > 0:
#             logging.warning(f"Failed to load or missing {failed_load_count} graphs up to index {num_to_load}.")

#         n_splits_actual = min(config.N_FOLDS, len(graphs))
#         kf = KFold(n_splits=n_splits_actual, shuffle=True, random_state=42)
#         if n_splits_actual < config.N_FOLDS:
#             logging.warning(f"Reduced KFold splits from {config.N_FOLDS} to {n_splits_actual} because only {len(graphs)} graphs were loaded.")
#         return graphs, kf

#     def _create_graph(self, idx, sfe_value):
#         lammps_file = os.path.join(config.LAMMPS_DATA_DIR, f"{config.LAMMPS_DATA_PREFIX}{idx}.data")
#         logging.debug(f"CreateGraph [{idx}]: Reading {lammps_file}")
#         try:
#             ase_atoms = read(lammps_file, format="lammps-data", atom_style="atomic")
#             full_structure = AseAtomsAdaptor.get_structure(ase_atoms)
#             logging.debug(f"CreateGraph [{idx}]: Read {len(full_structure)} atoms.")
#         except FileNotFoundError:
#             logging.warning(f"CreateGraph [{idx}]: LAMMPS data file not found: {lammps_file}. Skipping.")
#             return None
#         except Exception as e:
#             logging.error(f"CreateGraph [{idx}]: Error reading/converting {lammps_file}: {e}", exc_info=True)
#             return None

#         try:
#             cart_coords = full_structure.cart_coords
#             original_indices = np.arange(len(full_structure))
#             sf_filter_mask = (cart_coords[:, 2] >= config.SF_Z_MIN) & (cart_coords[:, 2] <= config.SF_Z_MAX)
#             sf_original_indices = original_indices[sf_filter_mask]
#             num_filtered = len(sf_original_indices)
#             if num_filtered == 0:
#                 logging.warning(f"CreateGraph [{idx}]: No atoms found in SF Z-region [{config.SF_Z_MIN}, {config.SF_Z_MAX}]. Skipping.")
#                 return None
#             logging.debug(f"CreateGraph [{idx}]: Found {num_filtered} atoms in Z-region.")
#             sort_key = np.argsort(sf_original_indices)
#             sf_original_indices_sorted = sf_original_indices[sort_key]
#             sf_species = [full_structure.species[i] for i in sf_original_indices_sorted]
#             sf_frac_coords = [full_structure.frac_coords[i] for i in sf_original_indices_sorted]
#             sf_structure = Structure(
#                 lattice=full_structure.lattice,
#                 species=sf_species,
#                 coords=sf_frac_coords,
#                 coords_are_cartesian=False
#             )
#             num_graph_nodes = len(sf_structure)
#             logging.debug(f"CreateGraph [{idx}]: Subgraph created with {num_graph_nodes} nodes.")
#             if abs(num_graph_nodes - 350) > 50:
#                 logging.warning(f"CreateGraph [{idx}]: Atoms in SF region ({num_graph_nodes}) differs significantly from ~350.")
#             logging.debug(f"CreateGraph [{idx}]: Extracting node features...")
#             node_features_raw = np.array([self._get_node_features(site) for site in sf_structure])
#             if node_features_raw.ndim != 2 or node_features_raw.shape[1] != 7:
#                 logging.error(f"CreateGraph [{idx}]: Raw node features have incorrect shape {node_features_raw.shape}. Expected ({num_graph_nodes}, 7). Skipping graph.")
#                 return None
#             node_features_processed = None
#             if self.node_scaler and hasattr(self.node_scaler, 'mean_') and self.node_scaler.mean_ is not None:
#                 try:
#                     atomic_numbers_col = node_features_raw[:, 0:1]
#                     features_to_scale = node_features_raw[:, 1:]
#                     if features_to_scale.shape[1] != len(self.node_scaler.mean_):
#                         logging.error(f"CreateGraph [{idx}]: Mismatch between features to scale ({features_to_scale.shape[1]}) and scaler dimensions ({len(self.node_scaler.mean_)}). Skipping scaling.")
#                         node_features_processed = node_features_raw
#                     else:
#                         scaled_features = self.node_scaler.transform(features_to_scale)
#                         logging.debug(f"CreateGraph [{idx}]: Node features (cols 1-6) scaled.")
#                         node_features_processed = np.hstack((atomic_numbers_col, scaled_features))
#                 except Exception as e:
#                     logging.error(f"CreateGraph [{idx}]: Error applying node scaler: {e}. Using raw features.", exc_info=True)
#                     node_features_processed = node_features_raw
#             else:
#                 is_fitting = getattr(self, '_fitting_scalers', False)
#                 if not is_fitting:
#                     logging.warning(f"CreateGraph [{idx}]: Node scaler not available/fitted. Using raw node features.")
#                 else:
#                     logging.debug(f"CreateGraph [{idx}]: Node scaler not applied (fitting phase).")
#                 node_features_processed = node_features_raw
#             logging.debug(f"CreateGraph [{idx}]: Extracting edge features...")
#             edges, edge_features_array = self._get_edge_features(sf_structure)
#             num_edges = len(edges)
#             edge_features_scaled = None
#             logging.debug(f"CreateGraph [{idx}]: Found {num_edges} edges within cutoff.")
#             if edges:
#                 if edge_features_array.size > 0:
#                     if edge_features_array.ndim != 2 or edge_features_array.shape[1] != 4:
#                         logging.warning(f"CreateGraph [{idx}]: Raw edge features have unexpected shape {edge_features_array.shape}. Expected (*, 4). Using raw.")
#                         edge_features_scaled = edge_features_array
#                     elif self.edge_scaler and hasattr(self.edge_scaler, 'mean_') and self.edge_scaler.mean_ is not None:
#                         try:
#                             if edge_features_array.shape[1] != len(self.edge_scaler.mean_):
#                                 logging.error(f"CreateGraph [{idx}]: Edge feature/scaler dimension mismatch! Features: {edge_features_array.shape[1]}, Scaler expects: {len(self.edge_scaler.mean_)}. Using raw edge features.")
#                                 edge_features_scaled = edge_features_array
#                             else:
#                                 edge_features_scaled = self.edge_scaler.transform(edge_features_array)
#                                 logging.debug(f"CreateGraph [{idx}]: Edge features scaled.")
#                         except Exception as e:
#                             logging.error(f"CreateGraph [{idx}]: Error applying edge scaler: {e}. Using raw edge features.", exc_info=True)
#                             edge_features_scaled = edge_features_array
#                     else:
#                         is_fitting = getattr(self, '_fitting_scalers', False)
#                         if not is_fitting:
#                             logging.warning(f"CreateGraph [{idx}]: Edge scaler not available/fitted. Using raw edge features.")
#                         else:
#                             logging.debug(f"CreateGraph [{idx}]: Edge scaler not applied (fitting phase).")
#                         edge_features_scaled = edge_features_array
#                 else:
#                     logging.warning(f"CreateGraph [{idx}]: Edges found ({num_edges}), but edge feature array is empty.")
#             logging.debug(f"CreateGraph [{idx}]: Calculating composition...")
#             full_comp = full_structure.composition
#             composition = np.array([full_comp.get_atomic_fraction(Z) for Z in [28, 27, 24]])
#             logging.debug(f"CreateGraph [{idx}]: Creating tensors...")
#             node_features_tensor = torch.tensor(node_features_processed, dtype=torch.float, device=device)
#             if edges:
#                 edges_tensor = torch.tensor(edges, dtype=torch.long, device=device).t().contiguous()
#                 if num_graph_nodes > 0 and edges_tensor.max() >= num_graph_nodes:
#                     logging.error(f"CreateGraph [{idx}]: Invalid edge index {edges_tensor.max()} >= num_nodes {num_graph_nodes}.")
#                     return None
#                 edge_attr_tensor = torch.tensor(edge_features_scaled, dtype=torch.float, device=device) if edge_features_scaled is not None else None
#             else:
#                 edges_tensor = torch.empty((2, 0), dtype=torch.long, device=device)
#                 edge_attr_tensor = None
#         except Exception as e:
#             logging.error(f"CreateGraph [{idx}]: Unhandled error during graph construction: {e}", exc_info=True)
#             return None
#         graph_data = Data(
#             x=node_features_tensor,
#             edge_index=edges_tensor,
#             edge_attr=edge_attr_tensor,
#             y=torch.tensor([sfe_value], dtype=torch.float, device=device),
#             composition=torch.tensor(composition, dtype=torch.float, device=device).view(1, -1),
#             structure_id=torch.tensor([idx], dtype=torch.long, device=device)
#         )
#         logging.debug(f"CreateGraph [{idx}]: Data object created successfully.")
#         return graph_data

#     def _get_node_features(self, site):
#         Z = site.specie.Z
#         try:
#             return [float(Z), config.THERMO_DATA[Z]['sgte_lse'], config.THERMO_DATA[Z]['allen_en'],
#                     config.THERMO_DATA[Z]['ionization_energy'], config.THERMO_DATA[Z]['specific_heat'],
#                     config.THERMO_DATA[Z]['atomic_mass'], config.THERMO_DATA[Z]['metallic_radius']]
#         except KeyError:
#             logging.error(f"Atomic number {Z} not found in config.THERMO_DATA")
#             raise

#     def _get_edge_features(self, structure):
#         edges, edge_features = [], []
#         max_neighbors, cutoff = 12, 3.2
#         try:
#             all_neighbors = structure.get_all_neighbors(cutoff, include_index=True)
#         except Exception as e:
#             logging.warning(f"Pymatgen neighbor finding failed for structure: {e}. Returning no edges.")
#             return [], np.array([])
#         if not all_neighbors:
#             logging.debug("No neighbors found within cutoff distance in the provided structure.")
#             return [], np.array([])
#         for i, neighbors_of_i in enumerate(all_neighbors):
#             sorted_neighbors = sorted(neighbors_of_i, key=lambda x: x[1])[:max_neighbors]
#             for neighbor_site, dist, j, image_offset in sorted_neighbors:
#                 sender_Z = int(structure[i].specie.Z)
#                 receiver_Z = int(structure[j].specie.Z)
#                 try:
#                     allen_diff = float(config.THERMO_DATA[sender_Z]['allen_en'] - config.THERMO_DATA[receiver_Z]['allen_en'])
#                     ion_avg = float((config.THERMO_DATA[sender_Z]['ionization_energy'] + config.THERMO_DATA[receiver_Z]['ionization_energy']) / 2)
#                     sgte_diff = float(abs(config.THERMO_DATA[sender_Z]['sgte_lse'] - config.THERMO_DATA[receiver_Z]['sgte_lse']))
#                     edge_feature = [float(dist), allen_diff, ion_avg, sgte_diff]
#                 except KeyError as e:
#                     logging.error(f"Atomic number {sender_Z} or {receiver_Z} not in THERMO_DATA: {e}")
#                     raise
#                 edge_features.append(edge_feature)
#                 edges.append((i, j))
#         return edges, np.array(edge_features) if edge_features else np.array([])

# # ===================== 3. GNN Model Architecture =====================
# class ThermoGNN(nn.Module):
#     def __init__(self, node_in=7, edge_in=4, comp_in=3, hidden_dim=128, dropout=0.2,
#                  node_feature_indices=None, edge_feature_indices=None):
#         super().__init__()
#         self.node_feature_indices = node_feature_indices if node_feature_indices is not None else list(range(node_in))
#         self.edge_feature_indices = edge_feature_indices if edge_feature_indices is not None else list(range(edge_in))
#         node_in = len(self.node_feature_indices)
#         edge_in = len(self.edge_feature_indices)
#         self.node_encoder = nn.Linear(node_in, hidden_dim)
#         self.edge_encoder = nn.Linear(edge_in, hidden_dim) if edge_in > 0 else None
#         self.comp_head = nn.Linear(comp_in, hidden_dim)
#         gat_edge_dim = hidden_dim if self.edge_encoder else None
#         self.conv1 = GATConv(hidden_dim, hidden_dim, heads=4, dropout=dropout, edge_dim=gat_edge_dim, add_self_loops=False)
#         self.bn1 = nn.BatchNorm1d(hidden_dim * 4)
#         self.conv2 = GATConv(hidden_dim * 4, hidden_dim, heads=4, dropout=dropout, edge_dim=gat_edge_dim, add_self_loops=False)
#         self.bn2 = nn.BatchNorm1d(hidden_dim * 4)
#         self.conv3 = GATConv(hidden_dim * 4, hidden_dim, heads=1, concat=False, dropout=dropout, edge_dim=gat_edge_dim, add_self_loops=False)
#         self.bn3 = nn.BatchNorm1d(hidden_dim)
#         self.predictor = nn.Sequential(
#             nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
#             nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.Dropout(dropout),
#             nn.Linear(hidden_dim // 2, 1)
#         )
#         self.dropout_rate = dropout

#     def forward(self, data):
#         x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
#         x = x[:, self.node_feature_indices]
#         x = self.node_encoder(x)
#         if self.edge_encoder and edge_attr is not None and edge_attr.numel() > 0 and len(self.edge_feature_indices) > 0:
#             edge_attr = edge_attr[:, self.edge_feature_indices]
#             edge_attr = self.edge_encoder(edge_attr)
#         else:
#             edge_attr = None
#         x = self.bn1(self.conv1(x, edge_index, edge_attr))
#         x = torch.relu(x)
#         x = nn.functional.dropout(x, p=self.dropout_rate, training=self.training)
#         x = self.bn2(self.conv2(x, edge_index, edge_attr))
#         x = torch.relu(x)
#         x = nn.functional.dropout(x, p=self.dropout_rate, training=self.training)
#         x = self.bn3(self.conv3(x, edge_index, edge_attr))
#         x = torch.relu(x)
#         if batch is None:
#             batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
#         x_pool = global_mean_pool(x, batch)
#         comp_feat = self.comp_head(data.composition.to(x_pool.device))
#         x_out = torch.cat([x_pool, comp_feat], dim=1)
#         return self.predictor(x_out).squeeze(-1)
# # ===================== 4. Training Utilities =====================
# class EarlyStopper:
#     def __init__(self, patience=75, min_delta=0.0):
#         self.patience = patience
#         self.min_delta = min_delta
#         self.counter = 0
#         self.min_validation_loss = float('inf')
#     def early_stop(self, validation_loss):
#         if validation_loss < self.min_validation_loss - self.min_delta:
#             self.min_validation_loss = validation_loss
#             self.counter = 0
#             return False
#         else:
#             self.counter += 1
#             logging.debug(f"Early stopping counter: {self.counter}/{self.patience}")
#             return self.counter >= self.patience

# def train_epoch(model, loader, optimizer, criterion, device):
#     model.train()
#     total_loss = 0
#     total_mae = 0
#     num_batches = 0
#     for batch_idx, batch in enumerate(loader):
#         for param in model.parameters():
#             param.requires_grad = True
#         optimizer.zero_grad()
#         try:
#             pred = model(batch)
#             target = batch.y.squeeze()
#             if pred.shape != target.shape:
#                 logging.warning(f"Train shape mismatch: pred {pred.shape}, target {target.shape}.")
#                 if pred.ndim == 1 and target.ndim == 0:
#                     target = target.unsqueeze(0)
#                 elif pred.ndim == 0 and target.ndim == 1:
#                     pred = pred.unsqueeze(0)
#             loss = criterion(pred, target)
#             mae = torch.mean(torch.abs(pred - target))
#             if torch.isnan(loss) or torch.isnan(mae):
#                 logging.warning(f"NaN loss/MAE in train batch {batch_idx}. Skipping.")
#                 continue
#             loss.backward()
#             optimizer.step()
#             total_loss += loss.item()
#             total_mae += mae.item()
#             num_batches += 1
#         except Exception as e:
#             logging.error(f"Error train batch {batch_idx}: {e}", exc_info=True)
#             continue
#     avg_loss = total_loss / num_batches if num_batches > 0 else float('nan')
#     avg_mae = total_mae / num_batches if num_batches > 0 else float('nan')
#     return avg_loss, avg_mae

# def validate(model, loader, criterion, device):
#     model.eval()
#     total_loss = 0
#     total_mae = 0
#     num_batches = 0
#     with torch.no_grad():
#         for batch_idx, batch in enumerate(loader):
#             try:
#                 pred = model(batch)
#                 target = batch.y.squeeze()
#                 if pred.shape != target.shape:
#                     logging.warning(f"Val shape mismatch: pred {pred.shape}, target {target.shape}.")
#                     if pred.ndim == 1 and target.ndim == 0:
#                         target = target.unsqueeze(0)
#                     elif pred.ndim == 0 and target.ndim == 1:
#                         pred = pred.unsqueeze(0)
#                 loss = criterion(pred, target)
#                 mae = torch.mean(torch.abs(pred - target))
#                 if torch.isnan(loss) or torch.isnan(mae):
#                     logging.warning(f"NaN loss/MAE in validation batch {batch_idx}.")
#                     continue
#                 total_loss += loss.item()
#                 total_mae += mae.item()
#                 num_batches += 1
#             except Exception as e:
#                 logging.error(f"Error val batch {batch_idx}: {e}", exc_info=True)
#                 continue
#     avg_loss = total_loss / num_batches if num_batches > 0 else float('nan')
#     avg_mae = total_mae / num_batches if num_batches > 0 else float('nan')
#     return avg_loss, avg_mae

# # ===================== 5. Evaluation & Visualization =====================
# def evaluate(model, loader, device):
#     model.eval()
#     y_true, y_pred, ids = [], [], []
#     with torch.no_grad():
#         for batch in loader:
#             try:
#                 pred = model(batch)
#                 y_true.extend(batch.y.cpu().numpy().flatten())
#                 y_pred.extend(pred.cpu().numpy().flatten())
#                 ids.extend(batch.structure_id.cpu().numpy().flatten() if hasattr(batch, 'structure_id') else [np.nan] * batch.num_graphs)
#             except Exception as e:
#                 logging.error(f"Error eval batch: {e}", exc_info=True)
#     y_true, y_pred, ids = np.array(y_true), np.array(y_pred), np.array(ids)
#     valid_idx = ~np.isnan(y_pred)
#     y_true, y_pred, ids = y_true[valid_idx], y_pred[valid_idx], ids[valid_idx]
#     metrics = {
#         'MAE': mean_absolute_error(y_true, y_pred) if len(y_true) else np.nan,
#         'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)) if len(y_true) else np.nan,
#         'R2': r2_score(y_true, y_pred) if len(y_true) else np.nan
#     }
#     results = pd.DataFrame({'Structure_ID': ids, 'True_SFE': y_true, 'Predicted_SFE': y_pred})
#     return metrics, results

# def plot_true_vs_pred(true, pred, fold, split, output_dir):
#     valid_idx = ~np.isnan(pred) & ~np.isnan(true)
#     true_valid, pred_valid = true[valid_idx], pred[valid_idx]
#     if len(true_valid) == 0:
#         logging.warning(f"Fold {fold} {split}: No valid points to plot.")
#         return
#     plt.figure(figsize=(8, 6))
#     plt.scatter(true_valid, pred_valid, alpha=0.6, label=f'{split} data', color='blue' if split == 'Train' else 'red')
#     min_val = min(np.min(true_valid), np.min(pred_valid))
#     max_val = max(np.max(true_valid), np.max(pred_valid))
#     plt.plot([min_val, max_val], [min_val, max_val], 'k--', label='Ideal')
#     plt.xlabel('True SFE (mJ/m²)')
#     plt.ylabel('Predicted SFE (mJ/m²)')
#     plt.title(f'{split} Set: Fold {fold} Predictions')
#     plt.legend()
#     plt.grid(True)
#     plt.axis('equal')
#     plt.tight_layout()
#     filename = os.path.join(output_dir, f"fold_{fold}_{split.lower()}_pred_vs_true.png")
#     plt.savefig(filename)
#     plt.close()
#     logging.info(f"Saved {split} prediction plot for fold {fold} at: {filename}")

# def plot_loss_and_mae(train_losses, val_losses, train_maes, val_maes, fold, output_dir):
#     if not train_losses or not val_losses or not train_maes or not val_maes:
#         logging.warning(f"Fold {fold}: No metrics to plot for loss/MAE.")
#         return
    
#     epochs = range(1, len(train_losses) + 1)
    
#     # Plot Loss
#     plt.figure(figsize=(8, 6))
#     plt.plot(epochs, train_losses, label='Train Loss', color='blue')
#     plt.plot(epochs, val_losses, label='Validation Loss', color='red')
#     plt.xlabel('Epoch')
#     plt.ylabel('Huber Loss')
#     plt.title(f'Fold {fold}: Loss vs Epoch')
#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()
#     loss_filename = os.path.join(output_dir, f"fold_{fold}_loss_vs_epoch.png")
#     plt.savefig(loss_filename)
#     plt.close()
#     logging.info(f"Saved loss plot for fold {fold} at: {loss_filename}")
    
#     # Plot MAE
#     plt.figure(figsize=(8, 6))
#     plt.plot(epochs, train_maes, label='Train MAE', color='blue')
#     plt.plot(epochs, val_maes, label='Validation MAE', color='red')
#     plt.xlabel('Epoch')
#     plt.ylabel('Mean Absolute Error (mJ/m²)')
#     plt.title(f'Fold {fold}: MAE vs Epoch')
#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()
#     mae_filename = os.path.join(output_dir, f"fold_{fold}_mae_vs_epoch.png")
#     plt.savefig(mae_filename)
#     plt.close()
#     logging.info(f"Saved MAE plot for fold {fold} at: {mae_filename}")

# # ===================== Preprocessing Functions =====================
# def preprocess_graphs(sfe_file=config.SFE_FILE,
#                       lammps_dir=config.LAMMPS_DATA_DIR,
#                       lammps_prefix=config.LAMMPS_DATA_PREFIX,
#                       output_dir=config.PRECOMPUTED_DIR,
#                       num_structures=200,
#                       start_from=1):
#     os.makedirs(output_dir, exist_ok=True)
#     logging.info(f"Starting graph preprocessing...")
#     logging.info(f"Reading SFE data from: {sfe_file}")
#     try:
#         data_SF = pd.read_excel(sfe_file)
#         energy_values = data_SF.iloc[:, -1].values
#         logging.info(f"Loaded {len(energy_values)} SFE values.")
#     except FileNotFoundError:
#         logging.error(f"SFE file not found: {sfe_file}. Aborting graph preprocessing.")
#         return
#     except Exception as e:
#         logging.error(f"Error reading SFE file {sfe_file}: {e}", exc_info=True)
#         return
#     num_available_sfe = len(energy_values)
#     if num_structures > num_available_sfe:
#         logging.warning(f"Requested {num_structures} structures, but only {num_available_sfe} SFE entries found. Processing up to {num_available_sfe}.")
#         num_structures = num_available_sfe
#     if start_from > num_structures:
#         logging.warning(f"Start_from index ({start_from}) is > number of available structures ({num_structures}). No graphs need generation.")
#         return
#     if start_from < 1:
#         logging.warning("Correcting start_from index to 1.")
#         start_from = 1
#     logging.info("Loading scalers...")
#     try:
#         loader = ThermoDataLoader(load_scalers=True)
#         loader._fitting_scalers = False
#         if loader.node_scaler is None:
#             logging.error("Node scaler failed to load. Cannot proceed.")
#             return
#         if loader.edge_scaler is None:
#             logging.warning("Edge scaler not loaded. Edge features will not be scaled.")
#     except Exception as e:
#         logging.error(f"Failed to prepare ThermoDataLoader with scalers: {e}", exc_info=True)
#         return
#     logging.info(f"Preprocessing graphs for indices {start_from} to {num_structures}...")
#     logging.info(f"Reading LAMMPS data from: {lammps_dir} (prefix: {lammps_prefix})")
#     logging.info(f"Saving precomputed graphs to: {output_dir}")
#     graphs_created = 0
#     graphs_skipped = 0
#     graphs_failed = 0
#     start_time_loop = time.time()
#     for idx in range(start_from, num_structures + 1):
#         graph_path = os.path.join(output_dir, f"graph_{idx}.pt")
#         logging.info(f"--- Processing Index {idx}/{num_structures} ---")
#         if os.path.exists(graph_path):
#             logging.info(f"Graph {idx}: Already exists at {graph_path}. Skipping.")
#             graphs_skipped += 1
#             continue
#         try:
#             if idx - 1 < num_available_sfe:
#                 sfe_value = energy_values[idx - 1]
#                 if pd.isna(sfe_value):
#                     logging.warning(f"Graph {idx}: SFE value is NaN. Skipping.")
#                     graphs_failed += 1
#                     continue
#                 logging.debug(f"Graph {idx}: Target SFE = {sfe_value:.4f}")
#             else:
#                 logging.error(f"Graph {idx}: Index out of bounds for SFE values ({num_available_sfe}). Skipping.")
#                 graphs_failed += 1
#                 continue
#             graph = loader._create_graph(idx, sfe_value)
#             if graph is not None:
#                 num_nodes = graph.num_nodes if hasattr(graph, 'num_nodes') else graph.x.shape[0]
#                 num_edges_in_graph = graph.edge_index.shape[1] if graph.edge_index is not None else 0
#                 logging.info(f"Graph {idx}: Successfully created. Nodes={num_nodes}, Edges={num_edges_in_graph}.")
#                 if num_nodes == 0:
#                     logging.warning(f"Graph {idx}: Created with 0 nodes. Skipping save.")
#                     graphs_failed += 1
#                     continue
#                 torch.save(graph, graph_path)
#                 logging.info(f"Graph {idx}: Saved to {graph_path}")
#                 graphs_created += 1
#             else:
#                 logging.warning(f"Graph {idx}: Creation failed (returned None). Check previous logs.")
#                 graphs_failed += 1
#         except Exception as e:
#             logging.error(f"Graph {idx}: Unhandled error during processing: {str(e)}", exc_info=True)
#             graphs_failed += 1
#             if os.path.exists(graph_path):
#                 try:
#                     os.remove(graph_path)
#                     logging.info(f"Removed potentially corrupt file: {graph_path}")
#                 except OSError:
#                     pass
#             continue
#     loop_duration = time.time() - start_time_loop
#     logging.info(f"--- Finished graph preprocessing loop ({loop_duration:.2f} seconds) ---")
#     logging.info(f"Summary: Range={start_from}-{num_structures}, Skipped Existing={graphs_skipped}, Created New={graphs_created}, Failed/Skipped={graphs_failed}.")

# def preprocess_scalers(sfe_file=config.SFE_FILE,
#                        lammps_dir=config.LAMMPS_DATA_DIR,
#                        lammps_prefix=config.LAMMPS_DATA_PREFIX,
#                        output_dir=config.SCALER_DIR,
#                        num_structures=200):
#     os.makedirs(output_dir, exist_ok=True)
#     logging.info(f"Starting scaler preprocessing...")
#     logging.info(f"Reading SFE data from {sfe_file} (for structure count)...")
#     try:
#         data_SF = pd.read_excel(sfe_file)
#     except FileNotFoundError:
#         logging.error(f"SFE file not found: {sfe_file}. Cannot determine structure count.")
#         return
#     except Exception as e:
#         logging.error(f"Error reading SFE file {sfe_file}: {e}", exc_info=True)
#         return
#     num_available = len(data_SF)
#     num_structures = min(num_structures, num_available)
#     logging.info(f"Preparing to collect features for scaler fitting from up to {num_structures} structures.")
#     logging.info(f"Reading LAMMPS data from: {lammps_dir} (prefix: {lammps_prefix})")
#     logging.info(f"Saving scalers to: {output_dir}")
#     temp_loader = ThermoDataLoader(load_scalers=False)
#     temp_loader._fitting_scalers = True
#     all_node_features_list = []
#     all_edge_features_list = []
#     structures_processed = 0
#     structures_failed = 0
#     total_nodes_collected = 0
#     total_edges_collected = 0
#     start_time_scaler_loop = time.time()
#     for idx in range(1, num_structures + 1):
#         logging.debug(f"Scaler: Processing index {idx}/{num_structures}...")
#         try:
#             lammps_file = os.path.join(lammps_dir, f"{lammps_prefix}{idx}.data")
#             if not os.path.exists(lammps_file):
#                 logging.warning(f"Scaler: LAMMPS file {lammps_file} not found. Skipping index {idx}.")
#                 structures_failed += 1
#                 continue
#             ase_atoms = read(lammps_file, format="lammps-data", atom_style="atomic")
#             full_structure = AseAtomsAdaptor.get_structure(ase_atoms)
#             logging.debug(f"Scaler [{idx}]: Read structure with {len(full_structure)} total atoms.")
#             cart_coords = full_structure.cart_coords
#             original_indices = np.arange(len(full_structure))
#             sf_filter_mask = (cart_coords[:, 2] >= config.SF_Z_MIN) & (cart_coords[:, 2] <= config.SF_Z_MAX)
#             sf_original_indices = original_indices[sf_filter_mask]
#             num_filtered_nodes = len(sf_original_indices)
#             if num_filtered_nodes == 0:
#                 logging.warning(f"Scaler [{idx}]: No atoms in SF region. Skipping.")
#                 structures_failed += 1
#                 continue
#             logging.debug(f"Scaler [{idx}]: Found {num_filtered_nodes} nodes in Z-region.")
#             nodes_in_struct = 0
#             for original_idx in sf_original_indices:
#                 site = full_structure[original_idx]
#                 try:
#                     raw_node_f = temp_loader._get_node_features(site)
#                     if len(raw_node_f) == 7:
#                         all_node_features_list.append(raw_node_f)
#                         nodes_in_struct += 1
#                     else:
#                         logging.warning(f"Scaler [{idx}]: Node {original_idx} has incorrect number of features ({len(raw_node_f)}). Skipping.")
#                 except KeyError as e:
#                     logging.error(f"Scaler [{idx}]: Skipping node (idx {original_idx}) due to KeyError {e} in THERMO_DATA.")
#                     continue
#                 except Exception as inner_e:
#                     logging.error(f"Scaler [{idx}]: Error getting features for node {original_idx}: {inner_e}")
#                     continue
#             total_nodes_collected += nodes_in_struct
#             if num_filtered_nodes > 0:
#                 sort_key = np.argsort(sf_original_indices)
#                 sf_original_indices_sorted = sf_original_indices[sort_key]
#                 sf_species = [full_structure.species[i] for i in sf_original_indices_sorted]
#                 sf_frac_coords = [full_structure.frac_coords[i] for i in sf_original_indices_sorted]
#                 sf_structure = Structure(full_structure.lattice, sf_species, sf_frac_coords, coords_are_cartesian=False)
#                 _, raw_edge_features_array = temp_loader._get_edge_features(sf_structure)
#                 if raw_edge_features_array.size > 0:
#                     if raw_edge_features_array.ndim == 2 and raw_edge_features_array.shape[1] == 4:
#                         all_edge_features_list.append(raw_edge_features_array)
#                         num_edges_in_struct = raw_edge_features_array.shape[0]
#                         total_edges_collected += num_edges_in_struct
#                         logging.debug(f"Scaler [{idx}]: Collected {num_edges_in_struct} edge features.")
#                     else:
#                         logging.warning(f"Scaler [{idx}]: Edge features array shape {raw_edge_features_array.shape} != (*, 4). Skipping.")
#             structures_processed += 1
#         except Exception as e:
#             logging.error(f"Scaler: Unhandled error processing structure {idx}: {str(e)}", exc_info=True)
#             structures_failed += 1
#             continue
#     scaler_loop_duration = time.time() - start_time_scaler_loop
#     logging.info(f"Scaler feature collection loop finished ({scaler_loop_duration:.2f} seconds).")
#     logging.info(f"Processed: {structures_processed}, Failed/Skipped: {structures_failed}")
#     logging.info(f"Collected total: {total_nodes_collected} node features, {total_edges_collected} edge features.")
#     if not all_node_features_list:
#         logging.error("No node features collected. Cannot fit node scaler.")
#     else:
#         logging.info(f"Fitting node scaler on features 1-6 from {total_nodes_collected} nodes...")
#         try:
#             node_features_np = np.array(all_node_features_list)
#             if node_features_np.ndim != 2 or node_features_np.shape[1] != 7:
#                 logging.error(f"Final node features array has unexpected shape {node_features_np.shape}. Expected (?, 7). Node scaler not saved.")
#             else:
#                 features_to_scale = node_features_np[:, 1:]
#                 node_scaler = StandardScaler()
#                 node_scaler.fit(features_to_scale)
#                 node_scaler_path = os.path.join(output_dir, "node_scaler.pkl")
#                 joblib.dump(node_scaler, node_scaler_path)
#                 logging.info(f"Saved node scaler (fitted on features 1-6) to {node_scaler_path}")
#                 if hasattr(node_scaler, 'mean_'):
#                     logging.debug(f"Node scaler fitted means (features 1-6): {node_scaler.mean_}")
#                     logging.debug(f"Node scaler fitted scales (features 1-6): {node_scaler.scale_}")
#         except Exception as e:
#             logging.error(f"Error fitting or saving node scaler: {e}", exc_info=True)
#     if not all_edge_features_list:
#         logging.warning("No edge features collected across all structures. Edge scaler will not be created.")
#     else:
#         logging.info(f"Fitting edge scaler on {total_edges_collected} edge features...")
#         try:
#             all_edge_features_np = np.vstack(all_edge_features_list)
#             if all_edge_features_np.shape[0] > 0:
#                 if all_edge_features_np.ndim == 2 and all_edge_features_np.shape[1] == 4:
#                     edge_scaler = StandardScaler()
#                     edge_scaler.fit(all_edge_features_np)
#                     edge_scaler_path = os.path.join(output_dir, "edge_scaler.pkl")
#                     joblib.dump(edge_scaler, edge_scaler_path)
#                     logging.info(f"Saved edge scaler to {edge_scaler_path}")
#                 else:
#                     logging.error(f"Final edge features array shape {all_edge_features_np.shape} != (?, 4). Edge scaler not saved.")
#             else:
#                 logging.warning("Edge feature array is empty after stacking. Edge scaler not saved.")
#         except ValueError as ve:
#             logging.warning(f"ValueError during edge scaler fitting (likely empty list): {ve}. Edge scaler not saved.")
#         except Exception as e:
#             logging.error(f"Error fitting or saving edge scaler: {e}", exc_info=True)
#     if hasattr(temp_loader, '_fitting_scalers'):
#         delattr(temp_loader, '_fitting_scalers')

# # ===================== 6. Main Execution =====================
# def main():
#     TOTAL_STRUCTURES = 800
#     START_GRAPH_INDEX = 1
#     logging.info(f"\n=== Starting Execution ===")
#     logging.info(f"Using device: {device}")
#     logging.info(f"Targeting {TOTAL_STRUCTURES} total structures.")

#     # Workflow Control
#     RUN_PREPROCESS_SCALERS = False
#     RUN_PREPROCESS_GRAPHS = False
#     RUN_TRAINING = True

#     if RUN_PREPROCESS_SCALERS:
#         logging.info("--- STAGE 1: Running Preprocessing Scalers ---")
#         logging.warning("Ensure old scaler files are deleted if starting fresh.")
#         preprocess_scalers(num_structures=TOTAL_STRUCTURES)
#         logging.info("--- Scaler Preprocessing Complete ---")
#         logging.info("Exiting after Stage 1. Set RUN_PREPROCESS_SCALERS=False and RUN_PREPROCESS_GRAPHS=True for Stage 2.")
#         return

#     if RUN_PREPROCESS_GRAPHS:
#         logging.info(f"--- STAGE 2: Running Incremental Graph Preprocessing (Starting from {START_GRAPH_INDEX}) ---")
#         logging.info("Ensure scalers exist and are compatible with SF region 15.0-19.0.")
#         preprocess_graphs(num_structures=TOTAL_STRUCTURES, start_from=START_GRAPH_INDEX)
#         logging.info("--- Graph Preprocessing Complete ---")
#         logging.info("Exiting after Stage 2. Set RUN_PREPROCESS_GRAPHS=False and RUN_TRAINING=True for Stage 3.")
#         return

#     if RUN_TRAINING:
#         logging.info("--- STAGE 3: Starting Training Phase ---")
#         loader = ThermoDataLoader(load_scalers=True)
#         try:
#             graphs, kf = loader.load_data(num_structures=TOTAL_STRUCTURES)
#         except Exception as e:
#             logging.error(f"Failed to load data for training: {e}. Try RUN_PREPROCESS_SCALERS=True.", exc_info=True)
#             return

#         if not graphs:
#             logging.error("No graphs were loaded for training. Exiting.")
#             return

#         # Define experiments with different node and edge feature combinations
#         experiments = [
#             {
#                 'name': 'baseline_allfeatures_xx',
#                 'node_feature_indices': [0, 1, 2, 3, 4, 5, 6],
#                 'edge_feature_indices': [0, 1, 2, 3]
#             },
#             # {
#             #     'name': 'edge:only_Z',
#             #     'node_feature_indices': [0],
#             #     'edge_feature_indices': [0, 1, 2, 3]
#             # },
#             # {
#             #     'name': 'Z+sgte_lse+allen_en',
#             #     'node_feature_indices': [0, 1, 2],
#             #     'edge_feature_indices': [0, 1, 2, 3]
#             # },
#             # {
#             #     'name': 'Z+sgte_lse+allen_en+IE+Specific_heat',
#             #     'node_feature_indices': [0, 1, 2, 3, 4],
#             #     'edge_feature_indices': [0, 1, 2, 3]
#             # },
#             # {
#             #     'name': 'no_edge_features',
#             #     'node_feature_indices': [0, 1, 2, 3, 4, 5, 6],
#             #     'edge_feature_indices': []
#             # },
#             # {
#             #     'name': 'dist_only',
#             #     'node_feature_indices': [0, 1, 2, 3, 4, 5, 6],
#             #     'edge_feature_indices': [0]
#             # }
#         ]

#         # Hyperparameters
#         best_params = {
#             'lr': 0.0001,
#             'hidden_dim': 128,
#             'batch_size': 16,
#             'dropout': 0.35,
#             'epochs': 600,
#             'early_stop_patience': 400,
#             'weight_decay': 5e-4
#         }
#         logging.info(f"Using hyperparameters: {best_params}")

#         for exp in experiments:
#             logging.info(f"\n=== Running Experiment: {exp['name']} ===")
#             config.OUTPUT_DIR = f"results/{exp['name']}"
#             os.makedirs(config.OUTPUT_DIR, exist_ok=True)

#             all_fold_results = []
#             fold_metrics_summary = []
#             fold_metrics_history = []
#             n_splits = kf.get_n_splits()
#             logging.info(f"Starting {n_splits}-Fold Cross Validation for {exp['name']}...")

#             for fold, (train_idx, test_idx) in enumerate(kf.split(graphs)):
#                 logging.info(f"\n=== Fold {fold + 1}/{n_splits} ===")
#                 train_data = [graphs[i] for i in train_idx]
#                 test_data = [graphs[i] for i in test_idx]
#                 if not train_data or not test_data:
#                     logging.warning(f"Fold {fold+1}: Not enough data for train/test split. Skipping.")
#                     continue

#                 effective_train_batch_size = min(best_params['batch_size'], len(train_data))
#                 effective_test_batch_size = min(best_params['batch_size'], len(test_data))
#                 train_loader = DataLoader(train_data, batch_size=effective_train_batch_size, shuffle=True)
#                 test_loader = DataLoader(test_data, batch_size=effective_test_batch_size, shuffle=False)

#                 model = ThermoGNN(
#                     hidden_dim=best_params['hidden_dim'],
#                     dropout=best_params['dropout'],
#                     node_feature_indices=exp['node_feature_indices'],
#                     edge_feature_indices=exp['edge_feature_indices']
#                 ).to(device)
#                 optimizer = optim.AdamW(model.parameters(), lr=best_params['lr'], weight_decay=best_params['weight_decay'])
#                 criterion = nn.HuberLoss()
#                 stopper = EarlyStopper(patience=best_params['early_stop_patience'], min_delta=0.001)

#                 best_val_loss = float('inf')
#                 best_epoch = 0
#                 model_save_path = os.path.join(config.OUTPUT_DIR, f"best_model_fold{fold+1}.pth")
#                 train_losses, val_losses, train_maes, val_maes = [], [], [], []

#                 for epoch in range(best_params['epochs']):
#                     start_time = time.time()
#                     train_loss, train_mae = train_epoch(model, train_loader, optimizer, criterion, device)
#                     val_loss, val_mae = validate(model, test_loader, criterion, device)
#                     epoch_time = time.time() - start_time

#                     if np.isnan(train_loss) or np.isnan(val_loss) or np.isnan(train_mae) or np.isnan(val_mae):
#                         logging.warning(f"Epoch {epoch+1}: NaN metrics (Train Loss:{train_loss:.4f}, Val Loss:{val_loss:.4f}, "
#                                       f"Train MAE:{train_mae:.4f}, Val MAE:{val_mae:.4f}). Stop fold.")
#                         break

#                     train_losses.append(train_loss)
#                     val_losses.append(val_loss)
#                     train_maes.append(train_mae)
#                     val_maes.append(val_mae)

#                     logging.info(f"Epoch {epoch+1}/{best_params['epochs']} | Time:{epoch_time:.2f}s | "
#                                 f"Train Loss:{train_loss:.4f} | Val Loss:{val_loss:.4f} | "
#                                 f"Train MAE:{train_mae:.4f} | Val MAE:{val_mae:.4f}")

#                     if val_loss < best_val_loss:
#                         best_val_loss = val_loss
#                         best_epoch = epoch + 1
#                         try:
#                             torch.save(model.state_dict(), model_save_path)
#                             logging.info(f"Saved new best model fold {fold+1} epoch {best_epoch} (Val Loss: {best_val_loss:.4f})")
#                         except Exception as e:
#                             logging.error(f"Error saving model fold {fold+1}: {e}")

#                     if stopper.early_stop(val_loss):
#                         logging.info(f"Early stop epoch {epoch+1} fold {fold+1}. Best Val Loss: {best_val_loss:.4f} at epoch {best_epoch}.")
#                         break

#                 fold_metrics_history.append({
#                     'fold': fold + 1,
#                     'train_losses': train_losses,
#                     'val_losses': val_losses,
#                     'train_maes': train_maes,
#                     'val_maes': val_maes
#                 })

#                 logging.info(f"Evaluating fold {fold + 1} (epoch {best_epoch})")
#                 try:
#                     model.load_state_dict(torch.load(model_save_path, map_location=device))
#                     metrics_test, results_test = evaluate(model, test_loader, device)
#                     results_test['Fold'] = fold + 1
#                     metrics_train, results_train = evaluate(model, train_loader, device)
#                     results_train['Fold'] = fold + 1

#                     all_fold_results.append(results_test)
#                     metrics = {
#                         'Fold': fold + 1,
#                         'Train_MAE': metrics_train['MAE'],
#                         'Train_RMSE': metrics_train['RMSE'],
#                         'Train_R2': metrics_train['R2'],
#                         'Test_MAE': metrics_test['MAE'],
#                         'Test_RMSE': metrics_test['RMSE'],
#                         'Test_R2': metrics_test['R2'],
#                         'BestEpoch': best_epoch,
#                         'BestValLoss': best_val_loss
#                     }
#                     fold_metrics_summary.append(metrics)
#                     logging.info(f"Fold {fold + 1} Metrics - Train: MAE={metrics_train['MAE']:.4f}, R2={metrics_train['R2']:.4f} | "
#                                 f"Test: MAE={metrics_test['MAE']:.4f}, R2={metrics_test['R2']:.4f}")
#                 except FileNotFoundError:
#                     logging.error(f"Model file {model_save_path} not found. Skipping eval fold {fold + 1}.")
#                 except Exception as e:
#                     logging.error(f"Error eval fold {fold + 1}: {e}", exc_info=True)

#             if not all_fold_results:
#                 logging.error(f"No folds completed successfully for experiment {exp['name']}.")
#                 continue

#             final_results_df = pd.concat(all_fold_results, ignore_index=True)
#             final_csv_path = os.path.join(config.OUTPUT_DIR, "all_predictions_cv.csv")
#             final_results_df.to_csv(final_csv_path, index=False)
#             logging.info(f"Saved test set predictions to: {final_csv_path}")

#             metrics_df = pd.DataFrame(fold_metrics_summary)
#             metric_cols = [col for col in metrics_df.columns if col not in ['Fold', 'BestEpoch', 'BestValLoss']]
#             mean_row = metrics_df[metric_cols].mean().round(4)
#             mean_row['Fold'] = 'Mean'
#             mean_row['BestEpoch'] = ''
#             mean_row['BestValLoss'] = ''
#             metrics_df = pd.concat([metrics_df, pd.DataFrame([mean_row])], ignore_index=True)
#             summary_csv_path = os.path.join(config.OUTPUT_DIR, "fold_metrics_summary.csv")
#             metrics_df.to_csv(summary_csv_path, index=False)
#             logging.info(f"Saved fold-wise metrics (with mean row) to: {summary_csv_path}")
#             logging.info(f"\n=== Fold Metrics with Mean for {exp['name']} ===")
#             logging.info(metrics_df.to_string(index=False))

#             filtered_df = metrics_df[metrics_df['Fold'] != 'Mean']
#             if not filtered_df.empty:
#                 best_fold_row = filtered_df.loc[filtered_df['Test_MAE'].idxmin()]
#                 best_fold = int(best_fold_row['Fold'])
#                 logging.info(f"\n=== Best Fold for {exp['name']}: Fold {best_fold} (Test MAE = {best_fold_row['Test_MAE']:.4f}, "
#                             f"Test R2 = {best_fold_row['Test_R2']:.4f}) ===")

#                 logging.info(f"Generating train/test and loss/MAE plots for best fold ({best_fold})...")
#                 try:
#                     model = ThermoGNN(
#                         hidden_dim=best_params['hidden_dim'],
#                         dropout=best_params['dropout'],
#                         node_feature_indices=exp['node_feature_indices'],
#                         edge_feature_indices=exp['edge_feature_indices']
#                     ).to(device)
#                     best_model_path = os.path.join(config.OUTPUT_DIR, f"best_model_fold{best_fold}.pth")
#                     model.load_state_dict(torch.load(best_model_path, map_location=device))
#                     train_loader = DataLoader([graphs[i] for i in train_idx], batch_size=best_params['batch_size'])
#                     test_loader = DataLoader([graphs[i] for i in test_idx], batch_size=best_params['batch_size'])
#                     train_metrics, train_results = evaluate(model, train_loader, device)
#                     test_metrics, test_results = evaluate(model, test_loader, device)
#                     plot_true_vs_pred(train_results['True_SFE'].values, train_results['Predicted_SFE'].values,
#                                     best_fold, 'Train', config.OUTPUT_DIR)
#                     plot_true_vs_pred(test_results['True_SFE'].values, test_results['Predicted_SFE'].values,
#                                     best_fold, 'Test', config.OUTPUT_DIR)
#                     best_fold_metrics = next(m for m in fold_metrics_history if m['fold'] == best_fold)
#                     plot_loss_and_mae(
#                         best_fold_metrics['train_losses'],
#                         best_fold_metrics['val_losses'],
#                         best_fold_metrics['train_maes'],
#                         best_fold_metrics['val_maes'],
#                         best_fold,
#                         config.OUTPUT_DIR
#                     )
#                 except FileNotFoundError:
#                     logging.error(f"Best model file {best_model_path} not found. Skipping plots for fold {best_fold}.")
#                 except Exception as e:
#                     logging.error(f"Error generating plots for fold {best_fold}: {e}", exc_info=True)
#             else:
#                 logging.error(f"No valid folds for {exp['name']}. Cannot determine best fold or generate plots.")

#         logging.info(f"\n=== Training and Evaluation Complete ===")
#         logging.info(f"Results saved to {config.OUTPUT_DIR}")
#     else:
#         if not RUN_PREPROCESS_SCALERS and not RUN_PREPROCESS_GRAPHS:
#             logging.warning("All run flags are False. Nothing to do.")

# if __name__ == "__main__":
#     try:
#         import ase
#         import torch_geometric
#         import pymatgen
#         import joblib
#         import sklearn
#         import matplotlib
#         import pandas
#         import openpyxl
#     except ImportError as e:
#         print(f"Error: Missing required library - {e}. See installation instructions.")
#     else:
#         main()




import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'  # Add before any torch imports

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, global_mean_pool
from pymatgen.core.structure import Structure
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from ase.io import read
from pymatgen.io.ase import AseAtomsAdaptor
import joblib
import logging
import time

# Configure device
device = torch.device("cpu")
torch.set_default_device(device)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ===================== 1. Configuration =====================
class Config:
    THERMO_DATA = {
        28: {'sgte_lse': 2.89, 'allen_en': 1.88, 'ionization_energy': 737.1, 'specific_heat': 26.1, 'atomic_mass': 58.693, 'metallic_radius': 1.24, 'symbol': 'Ni'},
        27: {'sgte_lse': -0.43, 'allen_en': 1.84, 'ionization_energy': 760.4, 'specific_heat': 24.8, 'atomic_mass': 58.933, 'metallic_radius': 1.25, 'symbol': 'Co'},
        24: {'sgte_lse': -2.85, 'allen_en': 1.65, 'ionization_energy': 652.9, 'specific_heat': 23.3, 'atomic_mass': 51.996, 'metallic_radius': 1.28, 'symbol': 'Cr'}
    }
    for z, data in THERMO_DATA.items():
        if 'symbol' not in data:
            if z == 28: data['symbol'] = 'Ni'
            elif z == 27: data['symbol'] = 'Co'
            elif z == 24: data['symbol'] = 'Cr'
            else: data['symbol'] = f'X{z}'
    N_FOLDS = 3
    EPOCHS = 300
    EARLY_STOP_PATIENCE = 75
    SFE_FILE = "temp_SFE.xlsx"
    LAMMPS_DATA_DIR = "New_results"
    LAMMPS_DATA_PREFIX = "NiCoCr_faulted_"
    PRECOMPUTED_DIR = "precomputed_graphs"
    SCALER_DIR = "scalers"
    OUTPUT_DIR = "results/"
    SF_Z_MIN = 15.0
    SF_Z_MAX = 19.0

config = Config()
os.makedirs(config.OUTPUT_DIR, exist_ok=True)
os.makedirs(config.PRECOMPUTED_DIR, exist_ok=True)
os.makedirs(config.SCALER_DIR, exist_ok=True)

# ===================== 2. Data Loading and Processing =====================
class ThermoDataLoader:
    def __init__(self, load_scalers=True):
        self.node_scaler = None
        self.edge_scaler = None
        if load_scalers:
            node_scaler_path = os.path.join(config.SCALER_DIR, "node_scaler.pkl")
            edge_scaler_path = os.path.join(config.SCALER_DIR, "edge_scaler.pkl")
            try:
                self.node_scaler = joblib.load(node_scaler_path)
                logging.info(f"Loaded node scaler from {node_scaler_path}")
            except FileNotFoundError:
                logging.error(f"Node scaler not found at {node_scaler_path}. Run preprocess_scalers() first.")
                raise
            except Exception as e:
                logging.error(f"Error loading node scaler: {e}", exc_info=True)
                raise
            if os.path.exists(edge_scaler_path):
                try:
                    self.edge_scaler = joblib.load(edge_scaler_path)
                    logging.info(f"Loaded edge scaler from {edge_scaler_path}")
                except Exception as e:
                    logging.error(f"Error loading edge scaler: {e}", exc_info=True)
                    logging.warning("Proceeding without edge scaler due to loading error.")
            else:
                logging.warning("Edge scaler file not found; edge features will not be scaled.")
        else:
            self.node_scaler = StandardScaler()
            self.edge_scaler = StandardScaler()

    def load_data(self, num_structures=800):
        logging.info(f"Loading precomputed graphs...")
        try:
            data_SF = pd.read_excel(config.SFE_FILE)
        except FileNotFoundError:
            logging.error(f"SFE file not found: {config.SFE_FILE}")
            raise
        num_available = len(data_SF)
        num_to_load = min(num_structures, num_available)
        logging.info(f"Found {num_available} samples in {config.SFE_FILE}, attempting to load up to {num_to_load}")

        graphs = []
        loaded_indices = []
        failed_load_count = 0
        for idx in range(1, num_to_load + 1):
            graph_path = os.path.join(config.PRECOMPUTED_DIR, f"graph_{idx}.pt")
            try:
                if os.path.exists(graph_path):
                    graph = torch.load(graph_path, map_location=device, weights_only=False)
                    if not hasattr(graph, 'x') or not hasattr(graph, 'edge_index') or not hasattr(graph, 'y'):
                        logging.warning(f"Graph {idx} from {graph_path} seems incomplete or corrupted. Skipping.")
                        failed_load_count += 1
                        continue
                    if graph.x.device != device: graph = graph.to(device)
                    graphs.append(graph)
                    loaded_indices.append(idx)
                else:
                    logging.debug(f"Precomputed graph {graph_path} not found. Skipping index {idx}.")
                    failed_load_count += 1
            except Exception as e:
                logging.error(f"Error loading graph {idx} from {graph_path}: {str(e)}", exc_info=True)
                failed_load_count += 1
                continue

        if not graphs:
            if failed_load_count == num_to_load and num_to_load > 0:
                error_msg = f"No precomputed graph files found in {config.PRECOMPUTED_DIR} (checked up to index {num_to_load}). Run graph preprocessing."
            else:
                error_msg = f"No valid graphs loaded. Loaded {len(graphs)}, Failed/Missing: {failed_load_count}. Check precomputed files and logs in {config.PRECOMPUTED_DIR}."
            raise ValueError(error_msg)

        logging.info(f"Successfully loaded {len(graphs)} graphs.")
        if loaded_indices:
            logging.info(f" Indices loaded: {min(loaded_indices)}...{max(loaded_indices)}")
        if failed_load_count > 0:
            logging.warning(f"Failed to load or missing {failed_load_count} graphs up to index {num_to_load}.")

        n_splits_actual = min(config.N_FOLDS, len(graphs))
        kf = KFold(n_splits=n_splits_actual, shuffle=True, random_state=42)
        if n_splits_actual < config.N_FOLDS:
            logging.warning(f"Reduced KFold splits from {config.N_FOLDS} to {n_splits_actual} because only {len(graphs)} graphs were loaded.")
        return graphs, kf

    def _create_graph(self, idx, sfe_value):
        lammps_file = os.path.join(config.LAMMPS_DATA_DIR, f"{config.LAMMPS_DATA_PREFIX}{idx}.data")
        logging.debug(f"CreateGraph [{idx}]: Reading {lammps_file}")
        try:
            ase_atoms = read(lammps_file, format="lammps-data", atom_style="atomic")
            full_structure = AseAtomsAdaptor.get_structure(ase_atoms)
            logging.debug(f"CreateGraph [{idx}]: Read {len(full_structure)} atoms.")
        except FileNotFoundError:
            logging.warning(f"CreateGraph [{idx}]: LAMMPS data file not found: {lammps_file}. Skipping.")
            return None
        except Exception as e:
            logging.error(f"CreateGraph [{idx}]: Error reading/converting {lammps_file}: {e}", exc_info=True)
            return None

        try:
            cart_coords = full_structure.cart_coords
            original_indices = np.arange(len(full_structure))
            sf_filter_mask = (cart_coords[:, 2] >= config.SF_Z_MIN) & (cart_coords[:, 2] <= config.SF_Z_MAX)
            sf_original_indices = original_indices[sf_filter_mask]
            num_filtered = len(sf_original_indices)
            if num_filtered == 0:
                logging.warning(f"CreateGraph [{idx}]: No atoms found in SF Z-region [{config.SF_Z_MIN}, {config.SF_Z_MAX}]. Skipping.")
                return None
            logging.debug(f"CreateGraph [{idx}]: Found {num_filtered} atoms in Z-region.")
            sort_key = np.argsort(sf_original_indices)
            sf_original_indices_sorted = sf_original_indices[sort_key]
            sf_species = [full_structure.species[i] for i in sf_original_indices_sorted]
            sf_frac_coords = [full_structure.frac_coords[i] for i in sf_original_indices_sorted]
            sf_structure = Structure(
                lattice=full_structure.lattice,
                species=sf_species,
                coords=sf_frac_coords,
                coords_are_cartesian=False
            )
            num_graph_nodes = len(sf_structure)
            logging.debug(f"CreateGraph [{idx}]: Subgraph created with {num_graph_nodes} nodes.")
            if abs(num_graph_nodes - 350) > 50:
                logging.warning(f"CreateGraph [{idx}]: Atoms in SF region ({num_graph_nodes}) differs significantly from ~350.")
            logging.debug(f"CreateGraph [{idx}]: Extracting node features...")
            node_features_raw = np.array([self._get_node_features(site) for site in sf_structure])
            if node_features_raw.ndim != 2 or node_features_raw.shape[1] != 7:
                logging.error(f"CreateGraph [{idx}]: Raw node features have incorrect shape {node_features_raw.shape}. Expected ({num_graph_nodes}, 7). Skipping graph.")
                return None
            node_features_processed = None
            if self.node_scaler and hasattr(self.node_scaler, 'mean_') and self.node_scaler.mean_ is not None:
                try:
                    atomic_numbers_col = node_features_raw[:, 0:1]
                    features_to_scale = node_features_raw[:, 1:]
                    if features_to_scale.shape[1] != len(self.node_scaler.mean_):
                        logging.error(f"CreateGraph [{idx}]: Mismatch between features to scale ({features_to_scale.shape[1]}) and scaler dimensions ({len(self.node_scaler.mean_)}). Skipping scaling.")
                        node_features_processed = node_features_raw
                    else:
                        scaled_features = self.node_scaler.transform(features_to_scale)
                        logging.debug(f"CreateGraph [{idx}]: Node features (cols 1-6) scaled.")
                        node_features_processed = np.hstack((atomic_numbers_col, scaled_features))
                except Exception as e:
                    logging.error(f"CreateGraph [{idx}]: Error applying node scaler: {e}. Using raw features.", exc_info=True)
                    node_features_processed = node_features_raw
            else:
                is_fitting = getattr(self, '_fitting_scalers', False)
                if not is_fitting:
                    logging.warning(f"CreateGraph [{idx}]: Node scaler not available/fitted. Using raw node features.")
                else:
                    logging.debug(f"CreateGraph [{idx}]: Node scaler not applied (fitting phase).")
                node_features_processed = node_features_raw
            logging.debug(f"CreateGraph [{idx}]: Extracting edge features...")
            edges, edge_features_array = self._get_edge_features(sf_structure)
            num_edges = len(edges)
            edge_features_scaled = None
            logging.debug(f"CreateGraph [{idx}]: Found {num_edges} edges within cutoff.")
            if edges:
                if edge_features_array.size > 0:
                    if edge_features_array.ndim != 2 or edge_features_array.shape[1] != 4:
                        logging.warning(f"CreateGraph [{idx}]: Raw edge features have unexpected shape {edge_features_array.shape}. Expected (*, 4). Using raw.")
                        edge_features_scaled = edge_features_array
                    elif self.edge_scaler and hasattr(self.edge_scaler, 'mean_') and self.edge_scaler.mean_ is not None:
                        try:
                            if edge_features_array.shape[1] != len(self.edge_scaler.mean_):
                                logging.error(f"CreateGraph [{idx}]: Edge feature/scaler dimension mismatch! Features: {edge_features_array.shape[1]}, Scaler expects: {len(self.edge_scaler.mean_)}. Using raw edge features.")
                                edge_features_scaled = edge_features_array
                            else:
                                edge_features_scaled = self.edge_scaler.transform(edge_features_array)
                                logging.debug(f"CreateGraph [{idx}]: Edge features scaled.")
                        except Exception as e:
                            logging.error(f"CreateGraph [{idx}]: Error applying edge scaler: {e}. Using raw edge features.", exc_info=True)
                            edge_features_scaled = edge_features_array
                    else:
                        is_fitting = getattr(self, '_fitting_scalers', False)
                        if not is_fitting:
                            logging.warning(f"CreateGraph [{idx}]: Edge scaler not available/fitted. Using raw edge features.")
                        else:
                            logging.debug(f"CreateGraph [{idx}]: Edge scaler not applied (fitting phase).")
                        edge_features_scaled = edge_features_array
                else:
                    logging.warning(f"CreateGraph [{idx}]: Edges found ({num_edges}), but edge feature array is empty.")
            logging.debug(f"CreateGraph [{idx}]: Calculating composition...")
            full_comp = full_structure.composition
            composition = np.array([full_comp.get_atomic_fraction(Z) for Z in [28, 27, 24]])
            logging.debug(f"CreateGraph [{idx}]: Creating tensors...")
            node_features_tensor = torch.tensor(node_features_processed, dtype=torch.float, device=device)
            if edges:
                edges_tensor = torch.tensor(edges, dtype=torch.long, device=device).t().contiguous()
                if num_graph_nodes > 0 and edges_tensor.max() >= num_graph_nodes:
                    logging.error(f"CreateGraph [{idx}]: Invalid edge index {edges_tensor.max()} >= num_nodes {num_graph_nodes}.")
                    return None
                edge_attr_tensor = torch.tensor(edge_features_scaled, dtype=torch.float, device=device) if edge_features_scaled is not None else None
            else:
                edges_tensor = torch.empty((2, 0), dtype=torch.long, device=device)
                edge_attr_tensor = None
        except Exception as e:
            logging.error(f"CreateGraph [{idx}]: Unhandled error during graph construction: {e}", exc_info=True)
            return None
        graph_data = Data(
            x=node_features_tensor,
            edge_index=edges_tensor,
            edge_attr=edge_attr_tensor,
            y=torch.tensor([sfe_value], dtype=torch.float, device=device),
            composition=torch.tensor(composition, dtype=torch.float, device=device).view(1, -1),
            structure_id=torch.tensor([idx], dtype=torch.long, device=device)
        )
        logging.debug(f"CreateGraph [{idx}]: Data object created successfully.")
        return graph_data

    def _get_node_features(self, site):
        Z = site.specie.Z
        try:
            return [float(Z), config.THERMO_DATA[Z]['sgte_lse'], config.THERMO_DATA[Z]['allen_en'],
                    config.THERMO_DATA[Z]['ionization_energy'], config.THERMO_DATA[Z]['specific_heat'],
                    config.THERMO_DATA[Z]['atomic_mass'], config.THERMO_DATA[Z]['metallic_radius']]
        except KeyError:
            logging.error(f"Atomic number {Z} not found in config.THERMO_DATA")
            raise

    def _get_edge_features(self, structure):
        edges, edge_features = [], []
        max_neighbors, cutoff = 12, 3.2
        try:
            all_neighbors = structure.get_all_neighbors(cutoff, include_index=True)
        except Exception as e:
            logging.warning(f"Pymatgen neighbor finding failed for structure: {e}. Returning no edges.")
            return [], np.array([])
        if not all_neighbors:
            logging.debug("No neighbors found within cutoff distance in the provided structure.")
            return [], np.array([])
        for i, neighbors_of_i in enumerate(all_neighbors):
            sorted_neighbors = sorted(neighbors_of_i, key=lambda x: x[1])[:max_neighbors]
            for neighbor_site, dist, j, image_offset in sorted_neighbors:
                sender_Z = int(structure[i].specie.Z)
                receiver_Z = int(structure[j].specie.Z)
                try:
                    allen_diff = float(config.THERMO_DATA[sender_Z]['allen_en'] - config.THERMO_DATA[receiver_Z]['allen_en'])
                    ion_avg = float((config.THERMO_DATA[sender_Z]['ionization_energy'] + config.THERMO_DATA[receiver_Z]['ionization_energy']) / 2)
                    sgte_diff = float(abs(config.THERMO_DATA[sender_Z]['sgte_lse'] - config.THERMO_DATA[receiver_Z]['sgte_lse']))
                    edge_feature = [float(dist), allen_diff, ion_avg, sgte_diff]
                except KeyError as e:
                    logging.error(f"Atomic number {sender_Z} or {receiver_Z} not in THERMO_DATA: {e}")
                    raise
                edge_features.append(edge_feature)
                edges.append((i, j))
        return edges, np.array(edge_features) if edge_features else np.array([])

# ===================== 3. GNN Model Architecture =====================
class ThermoGNN(nn.Module):
    def __init__(self, node_in=7, edge_in=4, comp_in=3, hidden_dim=128, dropout=0.2,
                 node_feature_indices=None, edge_feature_indices=None):
        super().__init__()
        self.node_feature_indices = node_feature_indices if node_feature_indices is not None else list(range(node_in))
        self.edge_feature_indices = edge_feature_indices if edge_feature_indices is not None else list(range(edge_in))
        node_in = len(self.node_feature_indices)
        edge_in = len(self.edge_feature_indices)
        self.node_encoder = nn.Linear(node_in, hidden_dim)
        self.edge_encoder = nn.Linear(edge_in, hidden_dim) if edge_in > 0 else None
        self.comp_head = nn.Linear(comp_in, hidden_dim)
        gat_edge_dim = hidden_dim if self.edge_encoder else None
        self.conv1 = GATConv(hidden_dim, hidden_dim, heads=4, dropout=dropout, edge_dim=gat_edge_dim, add_self_loops=False)
        self.bn1 = nn.BatchNorm1d(hidden_dim * 4)
        self.conv2 = GATConv(hidden_dim * 4, hidden_dim, heads=4, dropout=dropout, edge_dim=gat_edge_dim, add_self_loops=False)
        self.bn2 = nn.BatchNorm1d(hidden_dim * 4)
        self.conv3 = GATConv(hidden_dim * 4, hidden_dim, heads=1, concat=False, dropout=dropout, edge_dim=gat_edge_dim, add_self_loops=False)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        self.dropout_rate = dropout

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x = x[:, self.node_feature_indices]
        x = self.node_encoder(x)
        if self.edge_encoder and edge_attr is not None and edge_attr.numel() > 0 and len(self.edge_feature_indices) > 0:
            edge_attr = edge_attr[:, self.edge_feature_indices]
            edge_attr = self.edge_encoder(edge_attr)
        else:
            edge_attr = None
        x = self.bn1(self.conv1(x, edge_index, edge_attr))
        x = torch.relu(x)
        x = nn.functional.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.bn2(self.conv2(x, edge_index, edge_attr))
        x = torch.relu(x)
        x = nn.functional.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.bn3(self.conv3(x, edge_index, edge_attr))
        x = torch.relu(x)
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        x_pool = global_mean_pool(x, batch)
        comp_feat = self.comp_head(data.composition.to(x_pool.device))
        x_out = torch.cat([x_pool, comp_feat], dim=1)
        return self.predictor(x_out).squeeze(-1)
# ===================== 4. Training Utilities =====================
class EarlyStopper:
    def __init__(self, patience=75, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')
    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss - self.min_delta:
            self.min_validation_loss = validation_loss
            self.counter = 0
            return False
        else:
            self.counter += 1
            logging.debug(f"Early stopping counter: {self.counter}/{self.patience}")
            return self.counter >= self.patience

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    total_mae = 0
    num_batches = 0
    for batch_idx, batch in enumerate(loader):
        for param in model.parameters():
            param.requires_grad = True
        optimizer.zero_grad()
        try:
            pred = model(batch)
            target = batch.y.squeeze()
            if pred.shape != target.shape:
                logging.warning(f"Train shape mismatch: pred {pred.shape}, target {target.shape}.")
                if pred.ndim == 1 and target.ndim == 0:
                    target = target.unsqueeze(0)
                elif pred.ndim == 0 and target.ndim == 1:
                    pred = pred.unsqueeze(0)
            loss = criterion(pred, target)
            mae = torch.mean(torch.abs(pred - target))
            if torch.isnan(loss) or torch.isnan(mae):
                logging.warning(f"NaN loss/MAE in train batch {batch_idx}. Skipping.")
                continue
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_mae += mae.item()
            num_batches += 1
        except Exception as e:
            logging.error(f"Error train batch {batch_idx}: {e}", exc_info=True)
            continue
    avg_loss = total_loss / num_batches if num_batches > 0 else float('nan')
    avg_mae = total_mae / num_batches if num_batches > 0 else float('nan')
    return avg_loss, avg_mae

def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    total_mae = 0
    num_batches = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            try:
                pred = model(batch)
                target = batch.y.squeeze()
                if pred.shape != target.shape:
                    logging.warning(f"Val shape mismatch: pred {pred.shape}, target {target.shape}.")
                    if pred.ndim == 1 and target.ndim == 0:
                        target = target.unsqueeze(0)
                    elif pred.ndim == 0 and target.ndim == 1:
                        pred = pred.unsqueeze(0)
                loss = criterion(pred, target)
                mae = torch.mean(torch.abs(pred - target))
                if torch.isnan(loss) or torch.isnan(mae):
                    logging.warning(f"NaN loss/MAE in validation batch {batch_idx}.")
                    continue
                total_loss += loss.item()
                total_mae += mae.item()
                num_batches += 1
            except Exception as e:
                logging.error(f"Error val batch {batch_idx}: {e}", exc_info=True)
                continue
    avg_loss = total_loss / num_batches if num_batches > 0 else float('nan')
    avg_mae = total_mae / num_batches if num_batches > 0 else float('nan')
    return avg_loss, avg_mae

# ===================== 5. Evaluation & Visualization =====================
def evaluate(model, loader, device):
    model.eval()
    y_true, y_pred, ids = [], [], []
    with torch.no_grad():
        for batch in loader:
            try:
                pred = model(batch)
                y_true.extend(batch.y.cpu().numpy().flatten())
                y_pred.extend(pred.cpu().numpy().flatten())
                ids.extend(batch.structure_id.cpu().numpy().flatten() if hasattr(batch, 'structure_id') else [np.nan] * batch.num_graphs)
            except Exception as e:
                logging.error(f"Error eval batch: {e}", exc_info=True)
    y_true, y_pred, ids = np.array(y_true), np.array(y_pred), np.array(ids)
    valid_idx = ~np.isnan(y_pred)
    y_true, y_pred, ids = y_true[valid_idx], y_pred[valid_idx], ids[valid_idx]
    metrics = {
        'MAE': mean_absolute_error(y_true, y_pred) if len(y_true) else np.nan,
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)) if len(y_true) else np.nan,
        'R2': r2_score(y_true, y_pred) if len(y_true) else np.nan
    }
    results = pd.DataFrame({'Structure_ID': ids, 'True_SFE': y_true, 'Predicted_SFE': y_pred})
    return metrics, results

def plot_true_vs_pred(true, pred, fold, split, output_dir):
    valid_idx = ~np.isnan(pred) & ~np.isnan(true)
    true_valid, pred_valid = true[valid_idx], pred[valid_idx]
    if len(true_valid) == 0:
        logging.warning(f"Fold {fold} {split}: No valid points to plot.")
        return
    plt.figure(figsize=(8, 6))
    plt.scatter(true_valid, pred_valid, alpha=0.6, label=f'{split} data', color='blue' if split == 'Train' else 'red')
    min_val = min(np.min(true_valid), np.min(pred_valid))
    max_val = max(np.max(true_valid), np.max(pred_valid))
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', label='Ideal')
    plt.xlabel('True SFE (mJ/m²)')
    plt.ylabel('Predicted SFE (mJ/m²)')
    plt.title(f'{split} Set: Fold {fold} Predictions')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.tight_layout()
    filename = os.path.join(output_dir, f"fold_{fold}_{split.lower()}_pred_vs_true.png")
    plt.savefig(filename)
    plt.close()
    logging.info(f"Saved {split} prediction plot for fold {fold} at: {filename}")

def plot_loss_and_mae(train_losses, val_losses, train_maes, val_maes, fold, output_dir):
    if not train_losses or not val_losses or not train_maes or not val_maes:
        logging.warning(f"Fold {fold}: No metrics to plot for loss/MAE.")
        return
    
    epochs = range(1, len(train_losses) + 1)
    
    # Plot Loss
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train_losses, label='Train Loss', color='blue')
    plt.plot(epochs, val_losses, label='Validation Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Huber Loss')
    plt.title(f'Fold {fold}: Loss vs Epoch')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    loss_filename = os.path.join(output_dir, f"fold_{fold}_loss_vs_epoch.png")
    plt.savefig(loss_filename)
    plt.close()
    logging.info(f"Saved loss plot for fold {fold} at: {loss_filename}")
    
    # Plot MAE
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train_maes, label='Train MAE', color='blue')
    plt.plot(epochs, val_maes, label='Validation MAE', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Absolute Error (mJ/m²)')
    plt.title(f'Fold {fold}: MAE vs Epoch')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    mae_filename = os.path.join(output_dir, f"fold_{fold}_mae_vs_epoch.png")
    plt.savefig(mae_filename)
    plt.close()
    logging.info(f"Saved MAE plot for fold {fold} at: {mae_filename}")

# ===================== Preprocessing Functions =====================
def preprocess_graphs(sfe_file=config.SFE_FILE,
                      lammps_dir=config.LAMMPS_DATA_DIR,
                      lammps_prefix=config.LAMMPS_DATA_PREFIX,
                      output_dir=config.PRECOMPUTED_DIR,
                      num_structures=800,
                      start_from=1):
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"Starting graph preprocessing...")
    logging.info(f"Reading SFE data from: {sfe_file}")
    try:
        data_SF = pd.read_excel(sfe_file)
        energy_values = data_SF.iloc[:, -1].values
        logging.info(f"Loaded {len(energy_values)} SFE values.")
    except FileNotFoundError:
        logging.error(f"SFE file not found: {sfe_file}. Aborting graph preprocessing.")
        return
    except Exception as e:
        logging.error(f"Error reading SFE file {sfe_file}: {e}", exc_info=True)
        return
    num_available_sfe = len(energy_values)
    if num_structures > num_available_sfe:
        logging.warning(f"Requested {num_structures} structures, but only {num_available_sfe} SFE entries found. Processing up to {num_available_sfe}.")
        num_structures = num_available_sfe
    if start_from > num_structures:
        logging.warning(f"Start_from index ({start_from}) is > number of available structures ({num_structures}). No graphs need generation.")
        return
    if start_from < 1:
        logging.warning("Correcting start_from index to 1.")
        start_from = 1
    logging.info("Loading scalers...")
    try:
        loader = ThermoDataLoader(load_scalers=True)
        loader._fitting_scalers = False
        if loader.node_scaler is None:
            logging.error("Node scaler failed to load. Cannot proceed.")
            return
        if loader.edge_scaler is None:
            logging.warning("Edge scaler not loaded. Edge features will not be scaled.")
    except Exception as e:
        logging.error(f"Failed to prepare ThermoDataLoader with scalers: {e}", exc_info=True)
        return
    logging.info(f"Preprocessing graphs for indices {start_from} to {num_structures}...")
    logging.info(f"Reading LAMMPS data from: {lammps_dir} (prefix: {lammps_prefix})")
    logging.info(f"Saving precomputed graphs to: {output_dir}")
    graphs_created = 0
    graphs_skipped = 0
    graphs_failed = 0
    start_time_loop = time.time()
    for idx in range(start_from, num_structures + 1):
        graph_path = os.path.join(output_dir, f"graph_{idx}.pt")
        logging.info(f"--- Processing Index {idx}/{num_structures} ---")
        if os.path.exists(graph_path):
            logging.info(f"Graph {idx}: Already exists at {graph_path}. Skipping.")
            graphs_skipped += 1
            continue
        try:
            if idx - 1 < num_available_sfe:
                sfe_value = energy_values[idx - 1]
                if pd.isna(sfe_value):
                    logging.warning(f"Graph {idx}: SFE value is NaN. Skipping.")
                    graphs_failed += 1
                    continue
                logging.debug(f"Graph {idx}: Target SFE = {sfe_value:.4f}")
            else:
                logging.error(f"Graph {idx}: Index out of bounds for SFE values ({num_available_sfe}). Skipping.")
                graphs_failed += 1
                continue
            graph = loader._create_graph(idx, sfe_value)
            if graph is not None:
                num_nodes = graph.num_nodes if hasattr(graph, 'num_nodes') else graph.x.shape[0]
                num_edges_in_graph = graph.edge_index.shape[1] if graph.edge_index is not None else 0
                logging.info(f"Graph {idx}: Successfully created. Nodes={num_nodes}, Edges={num_edges_in_graph}.")
                if num_nodes == 0:
                    logging.warning(f"Graph {idx}: Created with 0 nodes. Skipping save.")
                    graphs_failed += 1
                    continue
                torch.save(graph, graph_path)
                logging.info(f"Graph {idx}: Saved to {graph_path}")
                graphs_created += 1
            else:
                logging.warning(f"Graph {idx}: Creation failed (returned None). Check previous logs.")
                graphs_failed += 1
        except Exception as e:
            logging.error(f"Graph {idx}: Unhandled error during processing: {str(e)}", exc_info=True)
            graphs_failed += 1
            if os.path.exists(graph_path):
                try:
                    os.remove(graph_path)
                    logging.info(f"Removed potentially corrupt file: {graph_path}")
                except OSError:
                    pass
            continue
    loop_duration = time.time() - start_time_loop
    logging.info(f"--- Finished graph preprocessing loop ({loop_duration:.2f} seconds) ---")
    logging.info(f"Summary: Range={start_from}-{num_structures}, Skipped Existing={graphs_skipped}, Created New={graphs_created}, Failed/Skipped={graphs_failed}.")

def preprocess_scalers(sfe_file=config.SFE_FILE,
                       lammps_dir=config.LAMMPS_DATA_DIR,
                       lammps_prefix=config.LAMMPS_DATA_PREFIX,
                       output_dir=config.SCALER_DIR,
                       num_structures=800):
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"Starting scaler preprocessing...")
    logging.info(f"Reading SFE data from {sfe_file} (for structure count)...")
    try:
        data_SF = pd.read_excel(sfe_file)
    except FileNotFoundError:
        logging.error(f"SFE file not found: {sfe_file}. Cannot determine structure count.")
        return
    except Exception as e:
        logging.error(f"Error reading SFE file {sfe_file}: {e}", exc_info=True)
        return
    num_available = len(data_SF)
    num_structures = min(num_structures, num_available)
    logging.info(f"Preparing to collect features for scaler fitting from up to {num_structures} structures.")
    logging.info(f"Reading LAMMPS data from: {lammps_dir} (prefix: {lammps_prefix})")
    logging.info(f"Saving scalers to: {output_dir}")
    temp_loader = ThermoDataLoader(load_scalers=False)
    temp_loader._fitting_scalers = True
    all_node_features_list = []
    all_edge_features_list = []
    structures_processed = 0
    structures_failed = 0
    total_nodes_collected = 0
    total_edges_collected = 0
    start_time_scaler_loop = time.time()
    for idx in range(1, num_structures + 1):
        logging.debug(f"Scaler: Processing index {idx}/{num_structures}...")
        try:
            lammps_file = os.path.join(lammps_dir, f"{lammps_prefix}{idx}.data")
            if not os.path.exists(lammps_file):
                logging.warning(f"Scaler: LAMMPS file {lammps_file} not found. Skipping index {idx}.")
                structures_failed += 1
                continue
            ase_atoms = read(lammps_file, format="lammps-data", atom_style="atomic")
            full_structure = AseAtomsAdaptor.get_structure(ase_atoms)
            logging.debug(f"Scaler [{idx}]: Read structure with {len(full_structure)} total atoms.")
            cart_coords = full_structure.cart_coords
            original_indices = np.arange(len(full_structure))
            sf_filter_mask = (cart_coords[:, 2] >= config.SF_Z_MIN) & (cart_coords[:, 2] <= config.SF_Z_MAX)
            sf_original_indices = original_indices[sf_filter_mask]
            num_filtered_nodes = len(sf_original_indices)
            if num_filtered_nodes == 0:
                logging.warning(f"Scaler [{idx}]: No atoms in SF region. Skipping.")
                structures_failed += 1
                continue
            logging.debug(f"Scaler [{idx}]: Found {num_filtered_nodes} nodes in Z-region.")
            nodes_in_struct = 0
            for original_idx in sf_original_indices:
                site = full_structure[original_idx]
                try:
                    raw_node_f = temp_loader._get_node_features(site)
                    if len(raw_node_f) == 7:
                        all_node_features_list.append(raw_node_f)
                        nodes_in_struct += 1
                    else:
                        logging.warning(f"Scaler [{idx}]: Node {original_idx} has incorrect number of features ({len(raw_node_f)}). Skipping.")
                except KeyError as e:
                    logging.error(f"Scaler [{idx}]: Skipping node (idx {original_idx}) due to KeyError {e} in THERMO_DATA.")
                    continue
                except Exception as inner_e:
                    logging.error(f"Scaler [{idx}]: Error getting features for node {original_idx}: {inner_e}")
                    continue
            total_nodes_collected += nodes_in_struct
            if num_filtered_nodes > 0:
                sort_key = np.argsort(sf_original_indices)
                sf_original_indices_sorted = sf_original_indices[sort_key]
                sf_species = [full_structure.species[i] for i in sf_original_indices_sorted]
                sf_frac_coords = [full_structure.frac_coords[i] for i in sf_original_indices_sorted]
                sf_structure = Structure(full_structure.lattice, sf_species, sf_frac_coords, coords_are_cartesian=False)
                _, raw_edge_features_array = temp_loader._get_edge_features(sf_structure)
                if raw_edge_features_array.size > 0:
                    if raw_edge_features_array.ndim == 2 and raw_edge_features_array.shape[1] == 4:
                        all_edge_features_list.append(raw_edge_features_array)
                        num_edges_in_struct = raw_edge_features_array.shape[0]
                        total_edges_collected += num_edges_in_struct
                        logging.debug(f"Scaler [{idx}]: Collected {num_edges_in_struct} edge features.")
                    else:
                        logging.warning(f"Scaler [{idx}]: Edge features array shape {raw_edge_features_array.shape} != (*, 4). Skipping.")
            structures_processed += 1
        except Exception as e:
            logging.error(f"Scaler: Unhandled error processing structure {idx}: {str(e)}", exc_info=True)
            structures_failed += 1
            continue
    scaler_loop_duration = time.time() - start_time_scaler_loop
    logging.info(f"Scaler feature collection loop finished ({scaler_loop_duration:.2f} seconds).")
    logging.info(f"Processed: {structures_processed}, Failed/Skipped: {structures_failed}")
    logging.info(f"Collected total: {total_nodes_collected} node features, {total_edges_collected} edge features.")
    if not all_node_features_list:
        logging.error("No node features collected. Cannot fit node scaler.")
    else:
        logging.info(f"Fitting node scaler on features 1-6 from {total_nodes_collected} nodes...")
        try:
            node_features_np = np.array(all_node_features_list)
            if node_features_np.ndim != 2 or node_features_np.shape[1] != 7:
                logging.error(f"Final node features array has unexpected shape {node_features_np.shape}. Expected (?, 7). Node scaler not saved.")
            else:
                features_to_scale = node_features_np[:, 1:]
                node_scaler = StandardScaler()
                node_scaler.fit(features_to_scale)
                node_scaler_path = os.path.join(output_dir, "node_scaler.pkl")
                joblib.dump(node_scaler, node_scaler_path)
                logging.info(f"Saved node scaler (fitted on features 1-6) to {node_scaler_path}")
                if hasattr(node_scaler, 'mean_'):
                    logging.debug(f"Node scaler fitted means (features 1-6): {node_scaler.mean_}")
                    logging.debug(f"Node scaler fitted scales (features 1-6): {node_scaler.scale_}")
        except Exception as e:
            logging.error(f"Error fitting or saving node scaler: {e}", exc_info=True)
    if not all_edge_features_list:
        logging.warning("No edge features collected across all structures. Edge scaler will not be created.")
    else:
        logging.info(f"Fitting edge scaler on {total_edges_collected} edge features...")
        try:
            all_edge_features_np = np.vstack(all_edge_features_list)
            if all_edge_features_np.shape[0] > 0:
                if all_edge_features_np.ndim == 2 and all_edge_features_np.shape[1] == 4:
                    edge_scaler = StandardScaler()
                    edge_scaler.fit(all_edge_features_np)
                    edge_scaler_path = os.path.join(output_dir, "edge_scaler.pkl")
                    joblib.dump(edge_scaler, edge_scaler_path)
                    logging.info(f"Saved edge scaler to {edge_scaler_path}")
                else:
                    logging.error(f"Final edge features array shape {all_edge_features_np.shape} != (?, 4). Edge scaler not saved.")
            else:
                logging.warning("Edge feature array is empty after stacking. Edge scaler not saved.")
        except ValueError as ve:
            logging.warning(f"ValueError during edge scaler fitting (likely empty list): {ve}. Edge scaler not saved.")
        except Exception as e:
            logging.error(f"Error fitting or saving edge scaler: {e}", exc_info=True)
    if hasattr(temp_loader, '_fitting_scalers'):
        delattr(temp_loader, '_fitting_scalers')

# ===================== 6. Main Execution =====================
def main():
    TOTAL_STRUCTURES = 800
    START_GRAPH_INDEX = 1
    logging.info(f"\n=== Starting Execution ===")
    logging.info(f"Using device: {device}")
    logging.info(f"Targeting {TOTAL_STRUCTURES} total structures.")

    # Workflow Control
    RUN_PREPROCESS_SCALERS = False
    RUN_PREPROCESS_GRAPHS = False
    RUN_TRAINING = True

    if RUN_PREPROCESS_SCALERS:
        logging.info("--- STAGE 1: Running Preprocessing Scalers ---")
        logging.warning("Ensure old scaler files are deleted if starting fresh.")
        preprocess_scalers(num_structures=TOTAL_STRUCTURES)
        logging.info("--- Scaler Preprocessing Complete ---")
        logging.info("Exiting after Stage 1. Set RUN_PREPROCESS_SCALERS=False and RUN_PREPROCESS_GRAPHS=True for Stage 2.")
        return

    if RUN_PREPROCESS_GRAPHS:
        logging.info(f"--- STAGE 2: Running Incremental Graph Preprocessing (Starting from {START_GRAPH_INDEX}) ---")
        logging.info("Ensure scalers exist and are compatible with SF region 15.0-19.0.")
        preprocess_graphs(num_structures=TOTAL_STRUCTURES, start_from=START_GRAPH_INDEX)
        logging.info("--- Graph Preprocessing Complete ---")
        logging.info("Exiting after Stage 2. Set RUN_PREPROCESS_GRAPHS=False and RUN_TRAINING=True for Stage 3.")
        return

    if RUN_TRAINING:
        logging.info("--- STAGE 3: Starting Training Phase ---")
        loader = ThermoDataLoader(load_scalers=True)
        try:
            graphs, kf = loader.load_data(num_structures=TOTAL_STRUCTURES)
        except Exception as e:
            logging.error(f"Failed to load data for training: {e}. Try RUN_PREPROCESS_SCALERS=True.", exc_info=True)
            return

        if not graphs:
            logging.error("No graphs were loaded for training. Exiting.")
            return

        # Define experiments with different node and edge feature combinations
        experiments = [
            {
                'name': 'baseline_allfeatures_800structures',
                'node_feature_indices': [0, 1, 2, 3, 4, 5, 6],
                'edge_feature_indices': [0, 1, 2, 3]
            },
            # {
            #     'name': 'edge:only_Z',
            #     'node_feature_indices': [0],
            #     'edge_feature_indices': [0, 1, 2, 3]
            # },
            # {
            #     'name': 'Z+sgte_lse+allen_en',
            #     'node_feature_indices': [0, 1, 2],
            #     'edge_feature_indices': [0, 1, 2, 3]
            # },
            # {
            #     'name': 'Z+sgte_lse+allen_en+IE+Specific_heat',
            #     'node_feature_indices': [0, 1, 2, 3, 4],
            #     'edge_feature_indices': [0, 1, 2, 3]
            # },
            # {
            #     'name': 'no_edge_features',
            #     'node_feature_indices': [0, 1, 2, 3, 4, 5, 6],
            #     'edge_feature_indices': []
            # },
            # {
            #     'name': 'dist_only',
            #     'node_feature_indices': [0, 1, 2, 3, 4, 5, 6],
            #     'edge_feature_indices': [0]
            # }
        ]

        # Hyperparameters
        best_params = {
            'lr': 0.0001,
            'hidden_dim': 128,
            'batch_size': 16,
            'dropout': 0.35,
            'epochs': 300,
            'early_stop_patience': 75,
            'weight_decay': 5e-4
        }
        logging.info(f"Using hyperparameters: {best_params}")

        for exp in experiments:
            logging.info(f"\n=== Running Experiment: {exp['name']} ===")
            config.OUTPUT_DIR = f"results/{exp['name']}"
            os.makedirs(config.OUTPUT_DIR, exist_ok=True)

            all_fold_results = []
            fold_metrics_summary = []
            fold_metrics_history = []
            n_splits = kf.get_n_splits()
            logging.info(f"Starting {n_splits}-Fold Cross Validation for {exp['name']}...")

            for fold, (train_idx, test_idx) in enumerate(kf.split(graphs)):
                logging.info(f"\n=== Fold {fold + 1}/{n_splits} ===")
                train_data = [graphs[i] for i in train_idx]
                test_data = [graphs[i] for i in test_idx]
                if not train_data or not test_data:
                    logging.warning(f"Fold {fold+1}: Not enough data for train/test split. Skipping.")
                    continue

                effective_train_batch_size = min(best_params['batch_size'], len(train_data))
                effective_test_batch_size = min(best_params['batch_size'], len(test_data))
                train_loader = DataLoader(train_data, batch_size=effective_train_batch_size, shuffle=True)
                test_loader = DataLoader(test_data, batch_size=effective_test_batch_size, shuffle=False)

                model = ThermoGNN(
                    hidden_dim=best_params['hidden_dim'],
                    dropout=best_params['dropout'],
                    node_feature_indices=exp['node_feature_indices'],
                    edge_feature_indices=exp['edge_feature_indices']
                ).to(device)
                optimizer = optim.AdamW(model.parameters(), lr=best_params['lr'], weight_decay=best_params['weight_decay'])
                criterion = nn.HuberLoss()
                stopper = EarlyStopper(patience=best_params['early_stop_patience'], min_delta=0.001)

                best_val_loss = float('inf')
                best_epoch = 0
                model_save_path = os.path.join(config.OUTPUT_DIR, f"best_model_fold{fold+1}.pth")
                train_losses, val_losses, train_maes, val_maes = [], [], [], []

                for epoch in range(best_params['epochs']):
                    start_time = time.time()
                    train_loss, train_mae = train_epoch(model, train_loader, optimizer, criterion, device)
                    val_loss, val_mae = validate(model, test_loader, criterion, device)
                    epoch_time = time.time() - start_time

                    if np.isnan(train_loss) or np.isnan(val_loss) or np.isnan(train_mae) or np.isnan(val_mae):
                        logging.warning(f"Epoch {epoch+1}: NaN metrics (Train Loss:{train_loss:.4f}, Val Loss:{val_loss:.4f}, "
                                      f"Train MAE:{train_mae:.4f}, Val MAE:{val_mae:.4f}). Stop fold.")
                        break

                    train_losses.append(train_loss)
                    val_losses.append(val_loss)
                    train_maes.append(train_mae)
                    val_maes.append(val_mae)

                    logging.info(f"Epoch {epoch+1}/{best_params['epochs']} | Time:{epoch_time:.2f}s | "
                                f"Train Loss:{train_loss:.4f} | Val Loss:{val_loss:.4f} | "
                                f"Train MAE:{train_mae:.4f} | Val MAE:{val_mae:.4f}")

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_epoch = epoch + 1
                        try:
                            torch.save(model.state_dict(), model_save_path)
                            logging.info(f"Saved new best model fold {fold+1} epoch {best_epoch} (Val Loss: {best_val_loss:.4f})")
                        except Exception as e:
                            logging.error(f"Error saving model fold {fold+1}: {e}")

                    if stopper.early_stop(val_loss):
                        logging.info(f"Early stop epoch {epoch+1} fold {fold+1}. Best Val Loss: {best_val_loss:.4f} at epoch {best_epoch}.")
                        break

                fold_metrics_history.append({
                    'fold': fold + 1,
                    'train_losses': train_losses,
                    'val_losses': val_losses,
                    'train_maes': train_maes,
                    'val_maes': val_maes
                })

                logging.info(f"Evaluating fold {fold + 1} (epoch {best_epoch})")
                try:
                    model.load_state_dict(torch.load(model_save_path, map_location=device))
                    metrics_test, results_test = evaluate(model, test_loader, device)
                    results_test['Fold'] = fold + 1
                    metrics_train, results_train = evaluate(model, train_loader, device)
                    results_train['Fold'] = fold + 1

                    all_fold_results.append(results_test)
                    metrics = {
                        'Fold': fold + 1,
                        'Train_MAE': metrics_train['MAE'],
                        'Train_RMSE': metrics_train['RMSE'],
                        'Train_R2': metrics_train['R2'],
                        'Test_MAE': metrics_test['MAE'],
                        'Test_RMSE': metrics_test['RMSE'],
                        'Test_R2': metrics_test['R2'],
                        'BestEpoch': best_epoch,
                        'BestValLoss': best_val_loss
                    }
                    fold_metrics_summary.append(metrics)
                    logging.info(f"Fold {fold + 1} Metrics - Train: MAE={metrics_train['MAE']:.4f}, R2={metrics_train['R2']:.4f} | "
                                f"Test: MAE={metrics_test['MAE']:.4f}, R2={metrics_test['R2']:.4f}")
                except FileNotFoundError:
                    logging.error(f"Model file {model_save_path} not found. Skipping eval fold {fold + 1}.")
                except Exception as e:
                    logging.error(f"Error eval fold {fold + 1}: {e}", exc_info=True)

            if not all_fold_results:
                logging.error(f"No folds completed successfully for experiment {exp['name']}.")
                continue

            final_results_df = pd.concat(all_fold_results, ignore_index=True)
            final_csv_path = os.path.join(config.OUTPUT_DIR, "all_predictions_cv.csv")
            final_results_df.to_csv(final_csv_path, index=False)
            logging.info(f"Saved test set predictions to: {final_csv_path}")

            metrics_df = pd.DataFrame(fold_metrics_summary)
            metric_cols = [col for col in metrics_df.columns if col not in ['Fold', 'BestEpoch', 'BestValLoss']]
            mean_row = metrics_df[metric_cols].mean().round(4)
            mean_row['Fold'] = 'Mean'
            mean_row['BestEpoch'] = ''
            mean_row['BestValLoss'] = ''
            metrics_df = pd.concat([metrics_df, pd.DataFrame([mean_row])], ignore_index=True)
            summary_csv_path = os.path.join(config.OUTPUT_DIR, "fold_metrics_summary.csv")
            metrics_df.to_csv(summary_csv_path, index=False)
            logging.info(f"Saved fold-wise metrics (with mean row) to: {summary_csv_path}")
            logging.info(f"\n=== Fold Metrics with Mean for {exp['name']} ===")
            logging.info(metrics_df.to_string(index=False))

            filtered_df = metrics_df[metrics_df['Fold'] != 'Mean']
            if not filtered_df.empty:
                best_fold_row = filtered_df.loc[filtered_df['Test_MAE'].idxmin()]
                best_fold = int(best_fold_row['Fold'])
                logging.info(f"\n=== Best Fold for {exp['name']}: Fold {best_fold} (Test MAE = {best_fold_row['Test_MAE']:.4f}, "
                            f"Test R2 = {best_fold_row['Test_R2']:.4f}) ===")

                logging.info(f"Generating train/test and loss/MAE plots for best fold ({best_fold})...")
                try:
                    model = ThermoGNN(
                        hidden_dim=best_params['hidden_dim'],
                        dropout=best_params['dropout'],
                        node_feature_indices=exp['node_feature_indices'],
                        edge_feature_indices=exp['edge_feature_indices']
                    ).to(device)
                    best_model_path = os.path.join(config.OUTPUT_DIR, f"best_model_fold{best_fold}.pth")
                    model.load_state_dict(torch.load(best_model_path, map_location=device))
                    train_loader = DataLoader([graphs[i] for i in train_idx], batch_size=best_params['batch_size'])
                    test_loader = DataLoader([graphs[i] for i in test_idx], batch_size=best_params['batch_size'])
                    train_metrics, train_results = evaluate(model, train_loader, device)
                    test_metrics, test_results = evaluate(model, test_loader, device)
                    plot_true_vs_pred(train_results['True_SFE'].values, train_results['Predicted_SFE'].values,
                                    best_fold, 'Train', config.OUTPUT_DIR)
                    plot_true_vs_pred(test_results['True_SFE'].values, test_results['Predicted_SFE'].values,
                                    best_fold, 'Test', config.OUTPUT_DIR)
                    best_fold_metrics = next(m for m in fold_metrics_history if m['fold'] == best_fold)
                    plot_loss_and_mae(
                        best_fold_metrics['train_losses'],
                        best_fold_metrics['val_losses'],
                        best_fold_metrics['train_maes'],
                        best_fold_metrics['val_maes'],
                        best_fold,
                        config.OUTPUT_DIR
                    )
                except FileNotFoundError:
                    logging.error(f"Best model file {best_model_path} not found. Skipping plots for fold {best_fold}.")
                except Exception as e:
                    logging.error(f"Error generating plots for fold {best_fold}: {e}", exc_info=True)
            else:
                logging.error(f"No valid folds for {exp['name']}. Cannot determine best fold or generate plots.")

        logging.info(f"\n=== Training and Evaluation Complete ===")
        logging.info(f"Results saved to {config.OUTPUT_DIR}")
    else:
        if not RUN_PREPROCESS_SCALERS and not RUN_PREPROCESS_GRAPHS:
            logging.warning("All run flags are False. Nothing to do.")

if __name__ == "__main__":
    try:
        import ase
        import torch_geometric
        import pymatgen
        import joblib
        import sklearn
        import matplotlib
        import pandas
        import openpyxl
    except ImportError as e:
        print(f"Error: Missing required library - {e}. See installation instructions.")
    else:
        main()