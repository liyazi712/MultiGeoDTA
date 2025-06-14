import pandas as pd
import torch

data_path = './PDBBindv2016_robustness/last_train_2016.csv'
df = pd.read_csv(data_path)

label_origin = df['label'].values
num = len(label_origin)

scale_factors = [0.2, 0.4, 0.6, 0.8, 1.0]

for scale in scale_factors:
    noise = scale * torch.randn(num).numpy()
    label_new = label_origin + noise
    df[f'label'] = label_new
    new_data_path = f'./PDBBindv2016_robustness/last_train_2016_noised_scale_{scale}.csv'
    df.to_csv(new_data_path, index=False)