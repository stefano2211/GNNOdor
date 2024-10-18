import torch
import torch.nn as nn
import torch_geometric.nn as pyg_nn
from hydra.utils import to_absolute_path as abspath
import hydra
from omegaconf import DictConfig
from process import process_data


class GNNModel(nn.Module):
    """
    A Graph Neural Network (GNN) model for node classification.

    This model consists of two graph convolutional layers, followed by a global mean pooling layer and a fully connected layer.

    Attributes:
    conv1 (pyg_nn.GraphConv): The first graph convolutional layer.
    conv2 (pyg_nn.GraphConv): The second graph convolutional layer.
    pool (pyg_nn.global_mean_pool): The global mean pooling layer.
    fc1 (nn.Linear): The fully connected layer.
    """
    def __init__(self):
        super(GNNModel, self).__init__()
        self.conv1 = pyg_nn.GraphConv(3, 128)
        self.conv2 = pyg_nn.GraphConv(128, 128)
        self.pool = pyg_nn.global_mean_pool
        self.fc1 = nn.Linear(128, 138)

    def forward(self, data):
        """
        The forward pass of the model.

        Parameters:
        data (pyg_data.Data): The input data, containing node features, edge indices, and batch indices.

        Returns:
        torch.Tensor: The output of the model, representing the predicted node labels.
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
        x = self.pool(x, batch)
        x = self.fc1(x)
        return torch.sigmoid(x)

def train(model, device, loader, optimizer, criterion):
    """
    Trains the model for one epoch.

    Parameters:
    model (nn.Module): The model to train.
    device (torch.device): The device to use for training.
    loader (pyg_data.DataLoader): The data loader for the training data.
    optimizer (torch.optim.Optimizer): The optimizer to use for training.
    criterion (nn.Module): The loss function to use for training.

    Returns:
    float: The average loss for the epoch.
    """
    model.train()
    total_loss = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        outputs = model(batch)

        # Ensure outputs and target have the same shape and data type
        outputs = outputs.view(-1, 138).float()
        target = batch.y.float().view(-1, 138)

        # Check for NaN or inf in target and replace with 0
        target[torch.isnan(target)] = 0
        target[torch.isinf(target)] = 0

        # Clip to ensure values are between 0 and 1 (optional, since we already handled NaN and inf)
        outputs = torch.clamp(outputs, 0, 1)
        target = torch.clamp(target, 0, 1)

        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def test(model, device, loader):
    """
    Evaluates the model on the test data.

    Parameters:
    model (nn.Module): The model to evaluate.
    device (torch.device): The device to use for evaluation.
    loader (pyg_data.DataLoader): The data loader for the test data.

    Returns:
    float: The accuracy of the model on the test data.
    """
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
def train_model(config: DictConfig):
    """
    Trains a Graph Neural Network (GNN) model using the specified configuration.

    This function performs the following steps:
    1. Loads the training and test datasets using the `process_data` function.
    2. Initializes the GNN model, loss function, and optimizer.
    3. Moves the model to the appropriate device (GPU or CPU).
    4. Trains the model for a specified number of epochs, logging the training loss.
    5. Evaluates the model on the test dataset, logging the test accuracy.
    6. Saves the model's state dictionary to the specified path.

    Parameters:
    config (DictConfig): Configuration for the experiment, containing parameters such as
                         data paths, model paths, and training settings.

    Returns:
    None: This function does not return any value but performs training and saves the model.
    """

    train_dataset,  test_dataset = process_data(config)


    model = GNNModel()
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Entrenamiento del modelo
    for epoch in range(2):
        train_loss = train(model, device, train_dataset, optimizer, criterion)
        print(f'Epoch {epoch+1}, Pérdida de entrenamiento: {train_loss:.4f}')

    # Prueba del modelo
    test_accuracy = test(model, device, test_dataset)
    print(f'Precisión en la prueba: {test_accuracy:.4f}')


    state_dict = model.state_dict()
    torch.save(state_dict, abspath(config.model_path))

if __name__ == "__main__":
    train_model()
