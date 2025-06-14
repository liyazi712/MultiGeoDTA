import pandas as pd


df = pd.read_csv('./pdbbind_data_last.csv')
unique_sequences = set(df['Sequence'])
protein_unique_list = list(unique_sequences)
with open(f'protein_unique_list_2021.fasta', 'w') as fasta_file:
    for i, seq in enumerate(protein_unique_list):
        fasta_file.write(f'>protein_{i}\n')
        fasta_file.write(f'{seq}\n')



