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
from process import create_data_list



def evaluate_data(raw_data:str):
    data = pd.read_csv(raw_data)

    X = data["nonStereoSMILES"]
    y = data.drop(["nonStereoSMILES"], axis=1)

    data_list =  create_data_list(X, y)
    data_list_loader = pyg_data.DataLoader(data_list, batch_size=32, shuffle=True)
    max_nodes = max([data.x.size(0) for data in data_list_loader])

    return  data_list_loader


def load_model(model_path:str):
    model = torch.load(model_path)
    return model

def predict_odor(model, device, loader):
    model.eval()
    total_correct = 0
    total_labels = 0  # Keep track of the total number of labels
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            outputs = model(batch)

            # For multi-label, use a threshold to determine predicted labels
            predicted = (outputs > 0.5).float()  # Assuming 0.5 as the threshold

            # Iterate over graphs in the batch for comparison
            for i in range(batch.num_graphs):
                start = batch.ptr[i]
                end = batch.ptr[i + 1]
                graph_predicted = predicted[i]  # Predicted labels for the current graph

                # Extract the true labels for the current graph, ensuring correct shape
                graph_target = batch.y[start:end].view(-1)  # Reshape to match graph_predicted

                # Rellenar el tensor graph_target con ceros hasta que tenga el mismo tamaño que graph_predicted
                graph_target = torch.nn.functional.pad(graph_target, (0, graph_predicted.size(0) - graph_target.size(0)), value=0)

                # Count correct predictions for the current graph
                total_correct += (graph_predicted == graph_target).sum().item()
                total_labels += graph_target.size(0)  # Add the number of labels in this graph

    accuracy = total_correct / total_labels  # Calculate accuracy based on total labels
    return accuracy

@hydra.main(config_path="../config", config_name="main", version_base="1.2")
def evaluate_model(config:DictConfig):
    data_list = evaluate_data(abspath(config.data.raw))

    model = load_model(config.model_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    test_accuracy = predict_odor(model, device, data_list)
    print(f'Precisión en la prueba: {test_accuracy:.4f}')
