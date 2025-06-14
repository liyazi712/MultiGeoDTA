import pandas as pd

pdbbind_dict = {}
pdb_id = []
resolution = []
release_year = []
affinity = []

with open('INDEX_general_PL_data.2021') as f:
    for line in f.readlines():
        if line[0] != '#':
            lines = line.strip().split('  ')
            pdb_id.append(lines[0])
            resolution.append(lines[1])
            release_year.append(lines[2])
            affinity.append(lines[3])
    pdbbind_dict = {'PDBID': pdb_id, 'Resolution': resolution, 'Release_year': release_year, 'Affinity': affinity}

    frame = pd.DataFrame(pdbbind_dict)
    frame.to_csv('./pdbbind_origin_data.csv')


