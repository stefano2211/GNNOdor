from hydra.utils import to_absolute_path as abspath
from fastapi import FastAPI
from rdkit import Chem
import torch
import hydra
from hydra import compose, initialize
from pydantic import BaseModel
import torch_geometric.data as pyg_data
import numpy as np
import pandas as pd
import uvicorn
from src.train_model import GNNModel
from src import process

app = FastAPI()

with initialize(version_base=None, config_path="../../config"):
    config = compose(config_name="main")
    MODEL_NAME = config.model_name
    PATH_MODEL = config.model_path

class Smile(BaseModel):
    smile:str

def load_model(model_path:str):
    torch.serialization.add_safe_globals([GNNModel])
    model = torch.load(model_path, map_location='cpu')
    return model

def predict_odor(smiles, max_nodes, model, device):
    """
    Predicts the odor type of a molecule given its SMILES string.

    Args:
        smiles: The SMILES string representing the molecule.
        max_nodes: The maximum number of nodes encountered in the training data. # Explain the new argument

    Returns:
        A list of predicted odor types.
    """

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


    data = pyg_data.Data(x=torch.tensor(atom_features),
                        edge_index=edge_index)


    num_nodes = data.x.size(0)
    if num_nodes < max_nodes:
        padding = torch.zeros((max_nodes - num_nodes, data.x.size(1)))
        data.x = torch.cat((data.x, padding), dim=0)


    model.eval()
    with torch.no_grad():
        output = model(data.to(device))
        predicted = (output).float()

    return predicted.view(-1).tolist()


def predict_odor_with_names(smiles, max_nodes, odor_names, model):
    """
    Predicts the odor type of a molecule given its SMILES string and returns the names of predicted odors.

    Args:
        smiles: The SMILES string representing the molecule.
        max_nodes: The maximum number of nodes encountered in the training data.
        odor_names: A list of odor names corresponding to the output indices.

    Returns:
        A list of predicted odor names.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    predicted_array = predict_odor(smiles, max_nodes, model, device)  # Asegúrate de que esto devuelve una lista
    predicted_odors = [odor_names[i] for i, value in enumerate(predicted_array) if value >= 0.5]  # Itera sobre todos los elementos y usa un umbral para seleccionar los olores predichos
    return predicted_odors

@app.post("/predict")
async def predict(smile: Smile):
    model = load_model(abspath(PATH_MODEL))
    smile_str = smile.smile  # Get the SMILES string from the request
    max_nodes = 92  # Assuming the maximum number of nodes is 100
    odor_names = ['alcoholic', 'aldehydic', 'alliaceous', 'almond', 'amber', 'animal',
        'anisic', 'apple', 'apricot', 'aromatic', 'balsamic', 'banana', 'beefy',
        'bergamot', 'berry', 'bitter', 'black currant', 'brandy', 'burnt',
        'buttery', 'cabbage', 'camphoreous', 'caramellic', 'cedar', 'celery',
        'chamomile', 'cheesy', 'cherry', 'chocolate', 'cinnamon', 'citrus', 'clean',
        'clove', 'cocoa', 'coconut', 'coffee', 'cognac', 'cooked', 'cooling',
        'cortex', 'coumarinic', 'creamy', 'cucumber', 'dairy', 'dry', 'earthy',
        'ethereal', 'fatty', 'fermented', 'fishy', 'floral', 'fresh', 'fruit skin',
        'fruity', 'garlic', 'gassy', 'geranium', 'grape', 'grapefruit', 'grassy',
        'green', 'hawthorn', 'hay', 'hazelnut', 'herbal', 'honey', 'hyacinth',
        'jasmin', 'juicy', 'ketonic', 'lactonic', 'lavender', 'leafy', 'leathery',
        'lemon', 'lily', 'malty', 'meaty', 'medicinal', 'melon', 'metallic',
        'milky', 'mint', 'muguet', 'mushroom', 'musk', 'musty', 'natural', 'nutty',
        'odorless', 'oily', 'onion', 'orange', 'orangeflower', 'orris', 'ozone',
        'peach', 'pear', 'phenolic', 'pine', 'pineapple', 'plum', 'popcorn',
        'potato', 'powdery', 'pungent', 'radish', 'raspberry', 'ripe', 'roasted',
        'rose', 'rummy', 'sandalwood', 'savory', 'sharp', 'smoky', 'soapy',
        'solvent', 'sour', 'spicy', 'strawberry', 'sulfurous', 'sweaty', 'sweet',
        'tea', 'terpenic', 'tobacco', 'tomato', 'tropical', 'vanilla', 'vegetable',
        'vetiver', 'violet', 'warm', 'waxy', 'weedy', 'winey', 'woody']  # Define the list of odor names
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    predicted_odors = predict_odor_with_names(smile_str, max_nodes, odor_names, model)
    return {"predicted_odors": predicted_odors}

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)