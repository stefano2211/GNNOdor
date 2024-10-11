import hydra
from evaluate_model import evaluate_predict_odor
from process import process_data
from train_model import train_model


@hydra.main(config_path='../config', config_name='main')
def main(config):
    train_model(config)
    evaluate_predict_odor(config)

 
if __name__ == "__main__":
    main()