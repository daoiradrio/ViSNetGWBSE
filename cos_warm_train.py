import multiprocessing
multiprocessing.set_start_method("fork")

import os
import torch
import shutil

from omegaconf import OmegaConf
from ViSNetGW.model.visnet import create_model
from ViSNetGW.data.gwset import GWSet
from ViSNetGW.data.omol25 import OMol25
from torch.utils.data import DataLoader
from math import ceil
from tqdm import tqdm



torch.manual_seed(42)



def main():
    cfg = OmegaConf.load(os.path.join(os.getcwd(), "config.yaml"))

    if cfg.data.dataset == "gwset":
        data_module = GWSet(**cfg.data)
    elif cfg.data.dataset == "omol25":
        data_module = OMol25(**cfg.data)
    else:
        print("Invalid dataset.")
        return

    train_dataloader = DataLoader(
        data_module.train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=4 if cfg.training.device == "cuda" else 0,
        persistent_workers=True if cfg.training.device == "cuda" else False,
        pin_memory=True if cfg.training.device == "cuda" else False,
        collate_fn=data_module.collate_fn
    )
    val_dataloader = DataLoader(
        data_module.val_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=4 if cfg.training.device == "cuda" else 0,
        persistent_workers=True if cfg.training.device == "cuda" else False,
        pin_memory=True if cfg.training.device == "cuda" else False,
        collate_fn=data_module.collate_fn
    )

    model = create_model(cfg)
    if cfg.model.load_model:
        state_dict = torch.load(cfg.model.model_path, map_location=torch.device("cpu"))
        model.load_state_dict(state_dict)
    if cfg.model.transfer_learn:
        model.representation_model.requires_grad_(False)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model.to(torch.device(cfg.training.device))

    print()
    print(f"Number of trainable parameters: {num_params}")
    print()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training.lr_max,
        weight_decay=cfg.training.weight_decay
    )
    
    #lr_warmup = torch.optim.lr_scheduler.LinearLR(
    #    optimizer,
    #    start_factor=cfg.training.lr_start/cfg.training.lr_max,
    #    total_iters=cfg.training.num_warmup_epochs
    #)
    lr_warm_cos = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer=optimizer,
        T_0=10,
        eta_min=cfg.training.eta_min
    )
    '''
    lr_decay = torch.optim.lr_scheduler.MultiplicativeLR(
        optimizer=optimizer,
        lr_lambda=lambda epoch: 0.5
    )
    lr_scheduler = torch.optim.lr_scheduler.ChainedScheduler(
        schedulers=[lr_warm_cos, lr_decay]
    )
    '''

    mse_fn = torch.nn.MSELoss()
    mae_fn = torch.nn.L1Loss()

    chkpt_dir = os.path.join(os.getcwd(), cfg.training.log_dir)
    if not os.path.exists(chkpt_dir):
        os.makedirs(chkpt_dir)
    
    shutil.copyfile(
        src=os.path.join(os.getcwd(), "config.yaml"),
        dst=os.path.join(os.getcwd(), cfg.training.log_dir, "config.yaml")
    )

    with open(os.path.join(chkpt_dir, "metrics.csv"), "w") as f:
        print("epoch,train_mse,train_mae,val_mse,val_mae,lr", file=f)

    min_val_mae = 1000000
    num_train_batches = ceil(cfg.data.num_train / cfg.training.batch_size)
    num_val_batches = ceil(cfg.data.num_val / cfg.training.batch_size)
    for epoch in range(cfg.training.num_epochs):
        train_mse = 0
        train_mae = 0
        val_mse = 0
        val_mae = 0
        model.train()
        for i, (data, E) in tqdm(enumerate(train_dataloader), total=num_train_batches, leave=False):
            data["z"] = data["z"].to(torch.device(cfg.training.device))
            data["pos"] = data["pos"].to(torch.device(cfg.training.device))
            data["batch"] = data["batch"].to(torch.device(cfg.training.device))
            E = E.to(torch.device(cfg.training.device))
            optimizer.zero_grad()
            E_pred, _ = model(data)
            mae = mae_fn(E_pred, E)
            mse = mse_fn(E_pred, E)
            mse.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=100.0)
            train_mse += (mse.detach() - train_mse) / (i + 1)
            train_mae += (mae.detach() - train_mae) / (i + 1)
            optimizer.step()
        with torch.no_grad():
            model.eval()
            for i, (data, E) in tqdm(enumerate(val_dataloader), total=num_val_batches, leave=False):
                data["z"] = data["z"].to(torch.device(cfg.training.device))
                data["pos"] = data["pos"].to(torch.device(cfg.training.device))
                data["batch"] = data["batch"].to(torch.device(cfg.training.device))
                E = E.to(torch.device(cfg.training.device))
                E_pred, _ = model(data)
                mae = mae_fn(E_pred, E)
                mse = mse_fn(E_pred, E)
                val_mse += (mse - val_mse) / (i + 1)
                val_mae += (mae - val_mae) / (i + 1)
        #if epoch < cfg.training.num_warmup_epochs:
        #    lr_warmup.step()
        #else:
        #    lr_decay.step()
        lr_warm_cos.step()
        if val_mae < min_val_mae:
            torch.save(model.state_dict(), os.path.join(chkpt_dir, f"best_model.ckpt"))
            min_val_mae = val_mae
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), os.path.join(chkpt_dir, f"model_{epoch + 1}_epochs.ckpt"))
        print(f"Epoch {epoch + 1:3d}: {train_mse:10.5f} {train_mae:10.5f} {val_mse:10.5f} {val_mae:10.5f} {optimizer.param_groups[0]['lr']:15.8f}")
        with open(os.path.join(chkpt_dir, "metrics.csv"), "a") as f:
            print(f"{epoch+1},{train_mse:.8f},{train_mae:.8f},{val_mse:.8f},{val_mae:.8f},{optimizer.param_groups[0]['lr']:.8f}", file=f)



if __name__ == "__main__":
    main()
