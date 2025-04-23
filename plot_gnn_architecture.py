import torch
from graphviz import Digraph

# Step 1: Specify the best model's details
MODEL_PATH = "results/baseline_allfeatures_1_GATLayer/best_model_fold2.pth"
EXPERIMENT = {
    'name': 'baseline_allfeatures',
    'node_feature_indices': [0, 1, 2, 3, 4, 5, 6],  # All 7 node features
    'edge_feature_indices': [0, 1, 2, 3]             # All 4 edge features
}
HIDDEN_DIM = 128
DROPOUT = 0.35
COMP_IN = 3

# Step 2: Verify model parameters and extract number of heads
try:
    state_dict = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
    # Confirm hidden_dim
    node_encoder_weight = state_dict['node_encoder.weight']
    hidden_dim = node_encoder_weight.shape[0]
    print(f"Confirmed hidden_dim: {hidden_dim}")

    # Extract number of heads from the GATConv layer (assuming key is 'conv1.lin.weight')
    # In PyTorch Geometric's GATConv, the weight matrix shape is (out_channels * heads, in_channels)
    gat1_weight = state_dict.get('conv1.lin.weight', None)
    if gat1_weight is not None:
        out_dim = gat1_weight.shape[0]  # out_channels * heads
        heads = out_dim // hidden_dim
        print(f"Number of heads in GATConv1: {heads}")
    else:
        print("GATConv1 weight not found in state_dict. Assuming heads=1 for single layer.")
        heads = 1
except FileNotFoundError:
    print(f"Model file {MODEL_PATH} not found. Using default hidden_dim={HIDDEN_DIM}, dropout={DROPOUT}, heads=1.")
    hidden_dim = HIDDEN_DIM
    heads = 1

# Step 3: Define the architecture for diagram
def create_gnn_diagram(node_in=7, edge_in=4, comp_in=COMP_IN, hidden_dim=hidden_dim, dropout=DROPOUT, heads=heads):
    dot = Digraph(comment='ThermoGNN Architecture - 1 GAT Layer', format='png')
    dot.attr(rankdir='TB', size='10,12', dpi='300')
    dot.attr('node', shape='box', style='filled', fillcolor='lightblue')
    dot.attr('edge', color='black')

    # Input nodes
    dot.node('node_input', f'Node Features\n({node_in})', fillcolor='lightgreen')
    dot.node('edge_input', f'Edge Features\n({edge_in})', fillcolor='lightgreen')
    dot.node('comp_input', f'Composition\n({comp_in})', fillcolor='lightgreen')

    # Encoder layers
    dot.node('node_enc', f'Node Encoder\nLinear({node_in}, {hidden_dim})')
    dot.node('edge_enc', f'Edge Encoder\nLinear({edge_in}, {hidden_dim})')
    dot.node('comp_head', f'Comp Head\nLinear({comp_in}, {hidden_dim})')

    # Single GATConv layer with dynamic heads
    gat1_out_dim = hidden_dim * heads
    dot.node('gat1', f'GATConv1\n({hidden_dim}→{hidden_dim}, heads={heads})\nBN({gat1_out_dim}), ReLU, Dropout({dropout})')

    # Pooling and concatenation
    dot.node('pool', f'Global Mean Pool\n({gat1_out_dim})')
    dot.node('concat', f'Concat\n({gat1_out_dim}+{hidden_dim}={gat1_out_dim + hidden_dim})')

    # Predictor MLP
    dot.node('mlp1', f'MLP1\nLinear({gat1_out_dim + hidden_dim}, {hidden_dim})\nReLU, Dropout({dropout})')
    dot.node('mlp2', f'MLP2\nLinear({hidden_dim}, {hidden_dim//2})\nReLU, Dropout({dropout})')
    dot.node('mlp3', f'MLP3\nLinear({hidden_dim//2}, 1)')

    # Output
    dot.node('output', f'SFE Output\n(1)', fillcolor='lightcoral')

    # Edges (data flow)
    dot.edge('node_input', 'node_enc')
    dot.edge('edge_input', 'edge_enc')
    dot.edge('comp_input', 'comp_head')
    dot.edge('node_enc', 'gat1', label='Node Emb')
    dot.edge('edge_enc', 'gat1', label='Edge Emb')
    dot.edge('gat1', 'pool')
    dot.edge('pool', 'concat')
    dot.edge('comp_head', 'concat')
    dot.edge('concat', 'mlp1')
    dot.edge('mlp1', 'mlp2')
    dot.edge('mlp2', 'mlp3')
    dot.edge('mlp3', 'output')

    # Save diagram
    output_file = 'gnn_architecture_1GATLayer_with_heads'
    dot.render(output_file, view=False, cleanup=True)
    print(f"Diagram saved as {output_file}.png")

# Step 4: Generate the diagram
create_gnn_diagram()





# import torch
# from graphviz import Digraph

# # Step 1: Specify the best model's details
# MODEL_PATH = "results/baseline_allfeatures_1_GATLayer/best_model_fold2.pth"  # Update with your best model's path
# EXPERIMENT = {
#     'name': 'baseline_allfeatures',
#     'node_feature_indices': [0, 1, 2, 3, 4, 5, 6],  # All 7 node features
#     'edge_feature_indices': [0, 1, 2, 3]             # All 4 edge features
# }
# HIDDEN_DIM = 128  # From set1_lr5e-4
# DROPOUT = 0.35    # From set1_lr5e-4
# COMP_IN = 3       # Composition input (Ni, Co, Cr fractions)

# # Step 2: Verify model parameters (optional, to confirm hidden_dim, dropout)
# try:
#     state_dict = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
#     # Extract hidden_dim from state_dict keys (e.g., node_encoder.weight shape)
#     node_encoder_weight = state_dict['node_encoder.weight']
#     hidden_dim = node_encoder_weight.shape[0]  # Output dimension
#     print(f"Confirmed hidden_dim: {hidden_dim}")
#     # Dropout is not in state_dict, rely on code definition
# except FileNotFoundError:
#     print(f"Model file {MODEL_PATH} not found. Using default hidden_dim={HIDDEN_DIM}, dropout={DROPOUT}.")

# # Step 3: Define the architecture for diagram
# def create_gnn_diagram(node_in=7, edge_in=4, comp_in=COMP_IN, hidden_dim=HIDDEN_DIM, dropout=DROPOUT):
#     dot = Digraph(comment='ThermoGNN Architecture', format='png')
#     dot.attr(rankdir='TB', size='10,12', dpi='300')  # Top-to-bottom, high-res
#     dot.attr('node', shape='box', style='filled', fillcolor='lightblue')
#     dot.attr('edge', color='black')

#     # Input nodes
#     dot.node('node_input', f'Node Features\n({node_in})', fillcolor='lightgreen')
#     dot.node('edge_input', f'Edge Features\n({edge_in})', fillcolor='lightgreen')
#     dot.node('comp_input', f'Composition\n({comp_in})', fillcolor='lightgreen')

#     # Encoder layers
#     dot.node('node_enc', f'Node Encoder\nLinear({node_in}, {hidden_dim})')
#     dot.node('edge_enc', f'Edge Encoder\nLinear({edge_in}, {hidden_dim})')
#     dot.node('comp_head', f'Comp Head\nLinear({comp_in}, {hidden_dim})')

#     # GATConv layers
#     dot.node('gat1', f'GATConv1\n({hidden_dim}→{hidden_dim}, heads=4)\nBN({hidden_dim*4}), ReLU, Dropout({dropout})')
#     dot.node('gat2', f'GATConv2\n({hidden_dim*4}→{hidden_dim}, heads=4)\nBN({hidden_dim*4}), ReLU, Dropout({dropout})')
#     dot.node('gat3', f'GATConv3\n({hidden_dim*4}→{hidden_dim}, heads=1)\nBN({hidden_dim}), ReLU')

#     # Pooling and concatenation
#     dot.node('pool', f'Global Mean Pool\n({hidden_dim})')
#     dot.node('concat', f'Concat\n({hidden_dim}+{hidden_dim}={hidden_dim*2})')

#     # Predictor MLP
#     dot.node('mlp1', f'MLP1\nLinear({hidden_dim*2}, {hidden_dim})\nReLU, Dropout({dropout})')
#     dot.node('mlp2', f'MLP2\nLinear({hidden_dim}, {hidden_dim//2})\nReLU, Dropout({dropout})')
#     dot.node('mlp3', f'MLP3\nLinear({hidden_dim//2}, 1)')

#     # Output
#     dot.node('output', f'SFE Output\n(1)', fillcolor='lightcoral')

#     # Edges (data flow)
#     dot.edge('node_input', 'node_enc')
#     dot.edge('edge_input', 'edge_enc')
#     dot.edge('comp_input', 'comp_head')
#     dot.edge('node_enc', 'gat1', label='Node Emb')
#     dot.edge('edge_enc', 'gat1', label='Edge Emb')
#     dot.edge('gat1', 'gat2')
#     dot.edge('gat2', 'gat3')
#     dot.edge('gat3', 'pool')
#     dot.edge('pool', 'concat')
#     dot.edge('comp_head', 'concat')
#     dot.edge('concat', 'mlp1')
#     dot.edge('mlp1', 'mlp2')
#     dot.edge('mlp2', 'mlp3')
#     dot.edge('mlp3', 'output')

#     # Save diagram
#     output_file = 'gnn_architecture_1GATLayer'
#     dot.render(output_file, view=False, cleanup=True)
#     print(f"Diagram saved as {output_file}.png")

# # Step 4: Generate the diagram
# create_gnn_diagram()