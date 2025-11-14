#!/usr/bin/env python3
import argparse
import os
import time
import yaml
import random
import numpy as np
import torch
import torchvision
import csv

from tqdm import tqdm
from difflogic import LogicLayer, GroupSum, PackBitsTensor


torch.set_num_threads(1)

BITS_TO_TORCH_FLOATING_POINT_TYPE = {
    16: torch.float16,
    32: torch.float32,
    64: torch.float64
}

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def load_n(loader, n):
    i = 0
    while i < n:
        for x in loader:
            yield x
            i += 1
            if i == n:
                break


def train_step(model, x, y, loss_fn, optimizer):
    out = model(x)
    loss = loss_fn(out, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


def eval_mode(model, loader, mode):
    orig = model.training
    with torch.no_grad():
        model.train(mode=mode)
        accs = []
        for x, y in loader:
            pred = model(x.to('cuda').round()).argmax(-1)
            accs.append((pred == y.to('cuda')).float().mean().item())
        res = float(np.mean(accs))
    model.train(mode=orig)
    return res


def packbits_eval(model, loader):
    orig = model.training
    with torch.no_grad():
        model.eval()
        accs = []
        for x, y in loader:
            x_pb = PackBitsTensor(x.to('cuda').reshape(x.shape[0], -1).round().bool())
            pred = model(x_pb).argmax(-1)
            accs.append((pred == y.to('cuda')).float().mean().item())
        res = float(np.mean(accs))
    model.train(mode=orig)
    return res

def save_checkpoint(model, optimizer, iteration, cfg, ckpt_dir="checkpoints"):
    os.makedirs(ckpt_dir, exist_ok=True)
    path = os.path.join(
        ckpt_dir,
        f"{cfg['experiment_name']}_iter{iteration}.pt"
    )
    torch.save({
        'model_state':    model.state_dict(),
        'optim_state':    optimizer.state_dict(),
        'iteration':      iteration,
        'config':         cfg,
    }, path)
    print(f"[CHECKPOINT] saved to {path}")



def run_experiment(config, log_csv_path):
    torch.set_num_threads(1)
    # Hyperparams
    batch_size        = config["batch_size"]
    num_iterations    = config["num_iterations"]
    training_bit_count= config["training_bit_count"]

    # Seeding
    seed = config.get("seed", None)
    if seed is not None:
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.cuda.manual_seed_all(seed)
        print(f"[INFO] Using fixed seed: {seed}")

    # Dataset setup
    dataset_name = config.get("dataset", "CIFAR10").upper()
    if dataset_name == "CIFAR10":
        binarize = lambda x: torch.cat([(x > (i+1)/32).float() for i in range(31)], dim=0)
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Lambda(binarize)
        ])
        train_set = torchvision.datasets.CIFAR10('./data', train=True, download=True, transform=transforms)
        test_set  = torchvision.datasets.CIFAR10('./data', train=False, download=True, transform=transforms)
        in_dim    = 3 * 32 * 32 * 31
    elif dataset_name == "CIFAR100":
        binarize = lambda x: torch.cat([(x > (i+1)/32).float() for i in range(31)], dim=0)
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Lambda(binarize)
        ])
        train_set = torchvision.datasets.CIFAR100('./data', train=True, download=True, transform=transforms)
        test_set  = torchvision.datasets.CIFAR100('./data', train=False, download=True, transform=transforms)
        in_dim    = 3 * 32 * 32 * 31
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    # DataLoaders
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                               shuffle=True, pin_memory=True,
                                               drop_last=True, num_workers=4)
    test_loader  = torch.utils.data.DataLoader(test_set,  batch_size=batch_size,
                                               shuffle=False, pin_memory=True,
                                               drop_last=False, num_workers=4)

    # Model
    llkw = dict(grad_factor=config["grad_factor"], connections='unique')
    layers = [torch.nn.Flatten(),
              LogicLayer(in_dim=in_dim, out_dim=config["num_neurons"], stochastic=config["stochastic"],
                         gumbel_tau=config.get("gumbel_tau",1.0), init = config["init"], **llkw)]
    for _ in range(config["num_layers"] - 1):
        layers.append(
            LogicLayer(in_dim=config["num_neurons"], out_dim=config["num_neurons"],
                       stochastic=config["stochastic"], gumbel_tau=config.get("gumbel_tau",1.0), init = config["init"], **llkw)
        )
    layers.append(GroupSum(k=100, tau=config["tau"]))
    model = torch.nn.Sequential(*layers).to('cuda')

    loss_fn   = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

    csv_file = open(log_csv_path, 'w', newline='')
    logger = csv.writer(csv_file)
    # Write header
    logger.writerow([
        'iteration',
        'train_acc_eval_mode',
        'train_acc_train_mode',
        'test_acc_eval_mode',
        'test_acc_train_mode',
        'test_acc_packbits',
        'duration_sec'
    ])

    start_time = time.time()
    # Training loop
    for i, (x, y) in tqdm(enumerate(load_n(train_loader, num_iterations)), total=num_iterations):
        x = x.to(BITS_TO_TORCH_FLOATING_POINT_TYPE[training_bit_count]).to('cuda')
        y = y.to('cuda')

        loss = train_step(model, x, y, loss_fn, optimizer)
        
        # Evaluation & best-model saving every 2000 iters
        if (i + 1) % 2000 == 0:
            metrics = {
                'train_acc_eval_mode': eval_mode(model, train_loader, mode=False),
                'train_acc_train_mode': eval_mode(model, train_loader, mode=True),
                'test_acc_eval_mode': eval_mode(model, test_loader, mode=False),
                'test_acc_train_mode': eval_mode(model, test_loader, mode=True),
                'test_acc_packbits': packbits_eval(model, test_loader),
                'iteration': i + 1,
                'duration_sec': time.time() - start_time
            }
            # Print to console
            print(metrics)
            # Append to CSV
            logger.writerow([
                metrics['iteration'],
                metrics['train_acc_eval_mode'],
                metrics['train_acc_train_mode'],
                metrics['test_acc_eval_mode'],
                metrics['test_acc_train_mode'],
                metrics['test_acc_packbits'],
                metrics['duration_sec']
            ])
            csv_file.flush()

        if (i + 1) % 100_000 == 0:
            save_checkpoint(model, optimizer, i + 1, config)

    csv_file.close()
            

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Run binary-logic experiments")
    p.add_argument("--config", type=str, required=True,
                   help="Path to YAML config")
    p.add_argument("--log", type=str, default=None,
                   help="Where to write metric logs (CSV). If omitted, uses experiment_name from the config.")
    args = p.parse_args()

    cfg = load_config(args.config)

    # If no --log provided, derive from experiment_name
    if args.log:
        log_path = args.log
    else:
        name = cfg.get("experiment_name", "metrics")
        # sanitize name if needed, e.g. replace spaces/slashes:
        safe_name = "".join(c if c.isalnum() or c in ('-','_') else '_' for c in name)
        log_path = os.path.join("logs", f"{safe_name}.csv")

    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    run_experiment(cfg, log_path)
