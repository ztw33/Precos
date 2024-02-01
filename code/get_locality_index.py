import os
import json

project = "java/geopackage-android"

project_dataset_dir = f"../project_dataset/{project}"
output_dir = f"output_dir/project_dataset/{project}/ext-subset_proj_data/all-types"
ext_dstore_size_file = f"output_dir/project_dataset/{project}/ext-subset_data/all-types/dstore_dir/dstore_size.txt"
proj_dstore_size_file = f"output_dir/project_dataset/{project}/proj/dstore_dir/dstore_size.txt"

with open(ext_dstore_size_file) as fin:
    ext_dstore_size = int(fin.read())
with open(proj_dstore_size_file) as fin:
    proj_dstore_size = int(fin.read())

dstore_size = {
    0: ext_dstore_size,
    1: proj_dstore_size
}
with open(f"{output_dir}/dstore_size.json", "w") as fout:
    json.dump(dstore_size, fout, indent=2)
locality_idx = [0]*dstore_size[0] + [1]*dstore_size[1]
with open(f"{output_dir}/locality_index.json", 'w') as fout:
    json.dump(locality_idx, fout, indent=2)

