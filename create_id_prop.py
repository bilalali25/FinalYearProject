
import pandas as pd
import os

# Paths
excel_file = "/Users/bilalali2/Downloads/FYP/Ibrahim/temp_SFE.xlsx"
output_dir = "../data/nicocr-sfe"
os.makedirs(output_dir, exist_ok=True)

# Read SFE values from Excel
df = pd.read_excel(excel_file)
sfe_values = df["SFE"].tolist()

# Generate id_prop.csv
with open(os.path.join(output_dir, "id_prop.csv"), "w") as f:
    for i, sfe in enumerate(sfe_values, 1):
        cif_id = f"NiCoCr_faulted_{i}.cif"
        f.write(f"{cif_id},{sfe}\n")

print("Generated id_prop.csv")