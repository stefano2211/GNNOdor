import hydra
import pandas as pd 
import numpy as np 
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
import torch_geometric.nn as pyg_nn
import torch_geometric.data as pyg_data
import hydra
from rdkit import Chem
from omegaconf import DictConfig

def get_data(raw_path:str):
    data = pd.read_csv(raw_path)
    return data

def drop_description(data:pd.DataFrame, drop):
    dataframe = data.drop([drop],axis=1)
    return dataframe

def get_features(data:pd.DataFrame, features):
    train_smiles, test_smiles, train_labels, test_labels = train_test_split(data[features], data.iloc[:, 1:], test_size=0.2, random_state=42)
    return train_smiles, test_smiles, train_labels, test_labels

def balance_data(train_smiles, train_labels, test_smiles, test_labels, features):
    """
        Balances the data using RandomOverSampler.

        Args:
            train_smiles: Training SMILES strings.
            train_labels: Training labels.
            test_smiles: Test SMILES strings.
            test_labels: Test labels.

        Returns:
            Balanced training and test data.
    """

    # Apply RandomOverSampler to each label column individually
    train_smiles_balanced = train_smiles.copy()
    train_labels_balanced = train_labels.copy()
    for col in train_labels.columns:
        oversampler = RandomOverSampler(random_state=42)
        train_smiles_resampled, train_labels_resampled = oversampler.fit_resample(train_smiles.values.reshape(-1, 1), train_labels[col])
        train_smiles_balanced = pd.concat([train_smiles_balanced, pd.Series(train_smiles_resampled.flatten())])
        train_labels_balanced = pd.concat([train_labels_balanced, pd.Series(train_labels_resampled)])

    # Convert back to original format
    train_smiles_balanced = pd.DataFrame(train_smiles_balanced, columns=[features])[features]
    train_labels_balanced = pd.DataFrame(train_labels_balanced, columns=train_labels.columns)

    return train_smiles_balanced, train_labels_balanced, test_smiles, test_labels


def create_data_list(smiles_list, labels):
    data_list = []
    max_nodes = 0
    for i, smiles in enumerate(smiles_list):
        molecule = Chem.MolFromSmiles(smiles)
        atom_features = []
        for atom in molecule.GetAtoms():
            features = []
            features.append(atom.GetAtomicNum())
            features.append(atom.GetDegree())
            features.append(atom.GetTotalValence())
            atom_features.append(features)

        edge_index = []
        for bond in molecule.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_index.append([i, j])
            edge_index.append([j, i])
        edge_index = torch.tensor(edge_index, dtype=torch.long).T

        atom_features = np.array(atom_features, dtype=np.float32)


        max_nodes = max(max_nodes, len(atom_features))

        data = pyg_data.Data(x=torch.tensor(atom_features),
                             edge_index=edge_index,
                             y=torch.tensor(labels.iloc[i].values.astype(np.float32)))
        data_list.append(data)

    for data in data_list:
        num_nodes = data.x.size(0)
        if num_nodes < max_nodes:
            padding = torch.zeros((max_nodes - num_nodes, data.x.size(1)))
            data.x = torch.cat((data.x, padding), dim=0)

    return data_list




@hydra.main(config_path="../config", config_name="main", version_base="1.2")
def process_data(config: DictConfig):
    """Function to process the data"""

    print(f"Process data using {config.data.raw}")
    print(f"Columns used: {config.process.use_columns}")


if __name__ == "__main__":
    process_data()
