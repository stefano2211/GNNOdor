import hydra
import pandas as pd 
import numpy as np 
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
import torch_geometric.nn as pyg_nn
import torch_geometric.data as pyg_data
from hydra.utils import to_absolute_path as abspath
import hydra
import pickle
from torch.utils.data import Dataset
import os
from rdkit import Chem
from omegaconf import DictConfig

def get_data(raw_path:str):
    data = pd.read_csv(raw_path)
    return data

def drop_description(data:pd.DataFrame, drop:list):
    dataframe = data.drop(drop,axis=1)
    return dataframe

def balance_data(train_smiles, train_labels, test_smiles, test_labels):
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
    train_smiles_balanced = pd.DataFrame(train_smiles_balanced, columns=["nonStereoSMILES"])["nonStereoSMILES"]
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

class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

def create_dataset(data_loader):
    data_list = []
    for batch in data_loader:
        data_list.extend(batch)
    return MyDataset(data_list)

def serialize_dataset(dataset, filename, save_dir):
    filepath = os.path.join(save_dir, filename)
    with open(filepath, 'wb') as f:
        pickle.dump(dataset, f)

def deserialize_dataset(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


@hydra.main(config_path="../config", config_name="main", version_base="1.1")
def process_data(config: DictConfig):
    data = get_data(abspath(config.data.raw))
    dataframe = drop_description(data, config.column.drop)

    train_smiles, test_smiles, train_labels, test_labels = train_test_split(dataframe["nonStereoSMILES"], dataframe.iloc[:, 1:], test_size=0.2, random_state=42)

    train_smiles_balanced, train_labels_balanced, test_smiles_balanced, test_labels_balanced = balance_data(train_smiles, train_labels, test_smiles, test_labels)    

    train_data_list = create_data_list(train_smiles_balanced, train_labels_balanced)
    test_data_list = create_data_list(test_smiles_balanced, test_labels_balanced)

    train_data_loader = pyg_data.DataLoader(train_data_list, batch_size=32, shuffle=True)
    test_data_loader = pyg_data.DataLoader(test_data_list, batch_size=32, shuffle=False)
    max_nodes_train = max([data.x.size(0) for data in train_data_list])

    # Crear conjuntos de datos
    train_dataset = create_dataset(train_data_loader)
    test_dataset = create_dataset(test_data_loader)

    # Serializar conjuntos de datos
    serialize_dataset(train_dataset, 'train_dataset.pkl', abspath(config.data.processed))
    serialize_dataset(test_dataset, 'test_dataset.pkl', abspath(config.data.processed))


if __name__ == "__main__":
    process_data()
