import pandas as pd
import numpy as np

data_path = './PDBBindv2016_robustness/last_train_2016.csv'
df = pd.read_csv(data_path)

missing_proportions = [0.2, 0.4, 0.6, 0.8]

for proportion in missing_proportions:
    num_to_drop = int(len(df) * proportion)
    indices_to_drop = np.random.choice(len(df), num_to_drop, replace=False)
    df_dropped = df.drop(indices_to_drop).reset_index(drop=True)
    new_data_path = f'./PDBBindv2016_robustness/last_train_2016_missing_{proportion}.csv'
    df_dropped.to_csv(new_data_path, index=False)