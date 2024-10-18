import hydra
from test_model import test_predict_odor
from train_model import train_model


@hydra.main(config_path='../config', config_name='main', version_base='1.2')
def main(config):
    train_model(config)
    test_predict_odor(config)

 
if __name__ == "__main__":
    main()