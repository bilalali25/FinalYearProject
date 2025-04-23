
import os
from pymatgen.core import Structure
from pymatgen.io.lammps.data import LammpsData
from pymatgen.io.cif import CifWriter

# Directories
data_dir = "/Users/bilalali2/Downloads/FYP/Ibrahim/New_results"
output_dir = "../data/nicocr-sfe/cif_files"
os.makedirs(output_dir, exist_ok=True)

# Convert each .data file to CIF
for filename in os.listdir(data_dir):
    if filename.endswith(".data") and filename.startswith("NiCoCr_faulted_"):
        try:
            # Read LAMMPS .data file with explicit atom_style
            filepath = os.path.join(data_dir, filename)
            print(f"Processing {filepath}...")
            lammps_data = LammpsData.from_file(filepath, atom_style="atomic")
            structure = lammps_data.structure  # Convert to pymatgen Structure

            # Write to CIF
            cif_filename = os.path.join(output_dir, filename.replace(".data", ".cif"))
            CifWriter(structure).write_file(cif_filename)
            print(f"Converted {filename} to {cif_filename}")
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue
    else:
        print(f"Skipping {filename}: not a .data file or incorrect naming convention")