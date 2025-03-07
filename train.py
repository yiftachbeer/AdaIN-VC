from tqdm.auto import trange
from pathlib import Path
import wandb
import yaml

import torch
import torch.nn as nn
from torch.utils.data import random_split

from data import InfiniteDataLoader, SpeakerDataset, infinite_iterator
from model import AdaINVC


def main(
    config_file: str,
    data_dir: str,
    save_dir: str,
    n_steps: int = int(1e6),
    save_steps: int = 5000,
    log_steps: int = 250,
    n_spks: int = 32,
    n_uttrs: int = 4,
):
    """
    Trains the model.

    Args:
    config_file: The config file for AdaIN-VC.
    data_dir: The directory of processed files given by preprocess.py.
    save_dir: The directory to save the model.
    n_steps: The number of steps for training.
    save_steps: To save the model every save steps.
    log_steps: To record training information every log steps.
    n_spks: The number of speakers in the batch.
    n_uttrs: The number of utterances for each speaker in the batch.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True

    # Load config
    config = yaml.load(open(config_file, "r"), Loader=yaml.FullLoader)

    # Prepare data
    data = SpeakerDataset(data_dir, segment=128, n_uttrs=n_uttrs)

    # split train/valid sets
    train_set, valid_set = random_split(
        data, [int(len(data) * 0.8), len(data) - int(len(data) * 0.8)]
    )
    print(f'Using speakers {[data.id2spk[idx] for idx in valid_set.indices]} for validation.')

    # construct loader
    train_loader = InfiniteDataLoader(
        train_set, batch_size=n_spks, shuffle=True, num_workers=8
    )
    valid_loader = InfiniteDataLoader(
        valid_set, batch_size=n_spks, shuffle=False, num_workers=8
    )

    # construct iterator
    train_iter = infinite_iterator(train_loader)
    valid_iter = infinite_iterator(valid_loader)

    with wandb.init(config=config, ) as run:
        config = wandb.config  # Standard for wandb, make sure we used everything as logged

        # Build model
        model = AdaINVC(config["Model"]).to(device)
        model = torch.jit.script(model)

        # Optimizer
        opt = torch.optim.Adam(
            model.parameters(),
            lr=config["Optimizer"]["lr"],
            betas=(config["Optimizer"]["beta1"], config["Optimizer"]["beta2"]),
            amsgrad=config["Optimizer"]["amsgrad"],
            weight_decay=config["Optimizer"]["weight_decay"],
        )

        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True)

        criterion = nn.L1Loss()
        pbar = trange(n_steps, ncols=0)
        valid_steps = 32

        for step in pbar:
            # get features
            org_mels = next(train_iter)
            org_mels = org_mels.flatten(0, 1)
            org_mels = org_mels.to(device)

            # reconstruction
            mu, log_sigma, emb, rec_mels = model(org_mels)

            # compute loss
            rec_loss = criterion(rec_mels, org_mels)
            kl_loss = 0.5 * (log_sigma.exp() + mu ** 2 - 1 - log_sigma).mean()
            rec_lambda = config["Lambda"]["rec"]
            kl_lambda = min(
                config["Lambda"]["kl"] * step / config["Lambda"]["kl_annealing"],
                config["Lambda"]["kl"],
            )
            loss = rec_lambda * rec_loss + kl_lambda * kl_loss

            # update parameters
            opt.zero_grad()
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
            opt.step()

            # save model and optimizer
            if (step + 1) % save_steps == 0:
                model_path = save_path / f'model-{step + 1}.ckpt'
                model.cpu()
                model.save(model_path)
                model.to(device)
                opt_path = save_path / f'opt-{step + 1}.ckpt'
                torch.save(opt.state_dict(), opt_path)

            if (step + 1) % log_steps == 0:
                # validation
                model.eval()
                valid_loss = 0
                for _ in range(valid_steps):
                    org_mels = next(valid_iter)
                    org_mels = org_mels.flatten(0, 1)
                    org_mels = org_mels.to(device)
                    mu, log_sigma, emb, rec_mels = model(org_mels)
                    loss = criterion(rec_mels, org_mels)
                    valid_loss += loss.item()
                valid_loss /= valid_steps
                wandb.log({'validation_rec_loss': valid_loss}, step + 1)
                model.train()

                wandb.log({'training_rec_loss': rec_loss,
                           'training_kl_loss': kl_loss,
                           'training_grad_norm': grad_norm,
                           'lambda_kl': kl_lambda},
                          step + 1)

            # update tqdm bar
            pbar.set_postfix({"rec_loss": rec_loss.item(), "kl_loss": kl_loss.item()})

    model_path = save_path / f'adain_vc.pt'
    model.cpu()
    model.save(model_path)
