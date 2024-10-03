import hydra
from evaluate_model import evaluate_model
from process import process_data
from train_model import train_model


@hydra.main(config_path='../config', config_name='main')
def main(config):
    train_model(config)
    evaluate_model(config)

 
if __name__ == "__main__":
    main()