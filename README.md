# OdorPrediction

https://github.com/stefano2211/GNNOdor/blob/main/GNN.jpeg

## Description
OdorPrediction is a project designed to predict odors using machine learning techniques. This project includes configuration management, automatic documentation generation, and tools for code review.

## Tools used in this project
* [hydra](https://hydra.cc/): Manage configuration files - [article](https://mathdatasimplified.com/stop-hard-coding-in-a-data-science-project-use-configuration-files-instead/)
* [pdoc](https://github.com/pdoc3/pdoc): Automatically create an API documentation for your project
* [pre-commit plugins](https://pre-commit.com/): Automate code reviewing formatting
* [PyTorch](https://pytorch.org/): Deep Learning framework
* [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/): Library built upon  PyTorch to easily write and train Graph Neural Networks (GNNs)
* [RDKit](https://www.rdkit.org/):Is open-source toolkit for cheminformatics


## Set up the environment


1. Create the virtual environment:
```bash
python3 -m venv venv
```
2. Activate the virtual environment:

- For Linux/MacOS:
```bash
source venv/bin/activate
```
- For Command Prompt:
```bash
.\venv\Scripts\activate
```
3. Install dependencies:
- To install all dependencies, run:
```bash
pip install -r requirements-dev.txt
```
- To install only production dependencies, run:
```bash
pip install -r requirements.txt
```
- To install a new package, run:
```bash
pip install <package-name>
```


## Auto-generate API documentation

To auto-generate API document for the project, run:

```bash
make docs
```

## Run API 

```bash
uvicorn app.src.api:app
```

## Contributions

Contributions are welcome. If you would like to contribute, please open an issue or a pull request.