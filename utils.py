import torch


def mackey_glass(n, tau=17, beta=0.2, gamma=0.1, p=10, device='cpu'):
    """dx/dt = β*x(t-τ)/(1+x(t-τ)^p) - γ*x(t)"""
    x = torch.zeros(n, device=device)
    x[0] = 1.2

    const = beta * 1.2 / (1 + 1.2 ** p)

    for t in range(1, min(tau, n)):
        x[t] = x[t - 1] + const - gamma * x[t - 1]

    for t in range(tau, n):
        x_tau = x[t - tau]
        x[t] = x[t - 1] + beta * x_tau / (1 + x_tau ** p) - gamma * x[t - 1]

    return x


def nrmse(pred, target):
    """Normalized RMSE"""
    return torch.sqrt(torch.mean((pred - target) ** 2)) / torch.std(target)


def prepare_data(cfg):
    """Generate Mackey-Glass data"""
    total = cfg['train_len'] + cfg['test_len'] + cfg['washout'] + 1
    data = mackey_glass(total, device=cfg['device'])

    train_in = data[:cfg['train_len']].unsqueeze(-1)
    train_out = data[1:cfg['train_len'] + 1].unsqueeze(-1)
    test_in = data[cfg['train_len']:cfg['train_len'] + cfg['test_len']].unsqueeze(-1)
    test_out = data[cfg['train_len'] + 1:cfg['train_len'] + cfg['test_len'] + 1].unsqueeze(-1)

    return train_in, train_out, test_in, test_out


def print_cfg(cfg):
    dens = "dense" if cfg.get("density") is None else f"{cfg['density']:.3g}"
    print(
        f"ESN [{cfg.get('device', 'cpu')}]:\n"
        f"res={cfg['n_res']}, rho={cfg['rho']:.3g}, dens={dens}\n"
        f"in={cfg['input_scale']:.3g}, leak={cfg['leak_rate']:.3g}, reg={cfg['reg']:.3g}\n"
        f"wash={cfg['washout']}, train={cfg['train_len']}, test={cfg['test_len']}\n"
        f"seed={cfg['seed']}\n"
    )
