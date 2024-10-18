import hydra
import pandas as pd 
import numpy as np 
import torch
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
import torch_geometric.data as pyg_data
from hydra.utils import to_absolute_path as abspath
import hydra
from rdkit import Chem
from omegaconf import DictConfig

def get_data(raw_path:str):
    """
    Loads data from a CSV file.

    Parameters:
    raw_path (str): Path to the CSV file containing the data.

    Returns:
    pd.DataFrame: DataFrame containing the loaded data.
    """
    data = pd.read_csv(raw_path)
    return data

def drop_description(data:pd.DataFrame, drop:list):
    """
    Removes specific columns from a DataFrame.

    Parameters:
    data (pd.DataFrame): DataFrame from which columns will be removed.
    drop (list): List of column names to remove.

    Returns:
    pd.DataFrame: DataFrame resulting from the column removal.
    """
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
    """
    Creates a data list for PyTorch Geometric from SMILES strings and labels.

    Parameters:
    smiles_list: List of SMILES strings.
    labels: DataFrame of labels.

    Returns:
    list: List of data in PyTorch Geometric format.
    """
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




@hydra.main(config_path="../config", config_name="main", version_base="1.1")
def process_data(config: DictConfig):
    """
    Processes data for training a machine learning model.

    This script performs the following tasks:
    1. Loads data from a CSV file.
    2. Removes unnecessary columns.
    3. Splits the data into training and testing sets.
    4. Balances the training data to avoid biases during training.
    5. Creates data lists for training and testing in PyTorch Geometric format.
    6. Creates data loaders for training and testing.

    Parameters:
    config (DictConfig): Configuration for the experiment.

    Returns:
    tuple: Data loaders for training and testing.
    """
    data = get_data(abspath(config.data.raw))
    dataframe = drop_description(data, config.column.drop)

    train_smiles, test_smiles, train_labels, test_labels = train_test_split(dataframe["nonStereoSMILES"], dataframe.iloc[:, 1:], test_size=0.2, random_state=42)

    train_smiles_balanced, train_labels_balanced, test_smiles_balanced, test_labels_balanced = balance_data(train_smiles, train_labels, test_smiles, test_labels)    

    train_data_list = create_data_list(train_smiles_balanced, train_labels_balanced)
    test_data_list = create_data_list(test_smiles_balanced, test_labels_balanced)

    train_data_loader = pyg_data.DataLoader(train_data_list, batch_size=32, shuffle=True)
    test_data_loader = pyg_data.DataLoader(test_data_list, batch_size=32, shuffle=False)
    max_nodes_train = max([data.x.size(0) for data in train_data_list])

    print("Proceso finalizado")
    print(max_nodes_train)

    return train_data_loader, test_data_loader
