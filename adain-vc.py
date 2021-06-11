import fire

from preprocess import main as preprocess
from train import main as train
from inference import main as inference


if __name__ == '__main__':
    fire.Fire({
        'preprocess': preprocess,
        'train': train,
        'inference': inference,
    })
