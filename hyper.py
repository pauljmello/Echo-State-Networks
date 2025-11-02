import csv
import random
from pathlib import Path

from main import train_esn, evaluate_esn
from utils import prepare_data

search_space = {
    'n_res': (100, 1000, 'int'),
    'rho': (0.5, 1.05, 'float'),
    'density': (0.01, 0.5, 'float'),
    'input_scale': (0.01, 5.0, 'float'),
    'leak_rate': (0.01, 0.99, 'float'),
    'reg': (0.00001, 0.5, 'float'),
    'washout': (50, 750, 'int'),
    'train_len': (2000, 25000, 'int'),
    'test_len': (2000, 7500, 'int'),
}


def sample_config(device):
    cfg = {'device': device}
    for param, (min_val, max_val, dtype) in search_space.items():
        if dtype == 'int':
            cfg[param] = random.randint(min_val, max_val)
        else:
            cfg[param] = random.uniform(min_val, max_val)
    return cfg


def count_existing(csv_path):
    if not csv_path.exists():
        return 0
    with open(csv_path, 'r') as f:
        return sum(1 for _ in csv.DictReader(f))


def run_trial(cfg):
    train_in, train_out, test_in, test_out = prepare_data(cfg)
    esn = train_esn(cfg, train_in, train_out)
    _, nrmse = evaluate_esn(esn, test_in, test_out)
    return float(nrmse.cpu().item())


def run_averaged_trial(cfg, n_runs=3):
    nrmse_values = []
    for _ in range(n_runs):
        esn_seed = random.randint(1, 100000)
        cfg['seed'] = esn_seed
        nrmse = run_trial(cfg)
        nrmse_values.append(nrmse)
    return sum(nrmse_values) / len(nrmse_values)


def format_value(val):
    if isinstance(val, float):
        return f'{val:.8f}'
    return val


def save_result(csv_path, trial, cfg, nrmse):
    header = not csv_path.exists()

    with open(csv_path, 'a', newline='') as f:
        fields = ['trial', 'nrmse'] + list(search_space.keys())
        writer = csv.DictWriter(f, fieldnames=fields)

        if header:
            writer.writeheader()

        row = {'trial': trial, 'nrmse': f'{nrmse:.8f}'}
        for k in search_space.keys():
            row[k] = format_value(cfg[k])

        writer.writerow(row)


def hyperparameter_sweep(n_trials, results_file, device):
    csv_path = Path(results_file)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    existing = count_existing(csv_path)
    start = existing + 1
    end = existing + n_trials

    if existing > 0:
        print(f"\nResuming from trial {start} ({existing} existing)\n")
    else:
        print()

    best_nrmse = float('inf')

    for trial in range(start, end + 1):
        cfg = sample_config(device)
        nrmse = run_averaged_trial(cfg)
        save_result(csv_path, trial, cfg, nrmse)

        status = f"Trial {trial}: NRMSE {nrmse:.6f}"
        if nrmse < best_nrmse:
            best_nrmse = nrmse
            status += " (New Best)"
        print(status)

    print(f"\nBest NRMSE: {best_nrmse:.6f}")
    print(f"Results: {csv_path}\n")


if __name__ == "__main__":
    hyperparameter_sweep(
        n_trials=1000,
        results_file='results/hyper_results.csv',
        device='cuda'
    )
