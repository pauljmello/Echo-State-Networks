from esn import ESN
from utils import nrmse, prepare_data, print_cfg
from visualize import generate_all_visualizations


def config():
    """General parameter sweep tests on Mackey-glass yielded the following configurations:"""
    return {                    # | Suggested   | Best Found   | Top 10% Range     |
        'n_res': 550,           # | 550         | 413          | 300-800           |
        'rho': 0.92,            # | 0.92        | 0.9274       | 0.70-1.04         |
        'density': 0.17,        # | 0.17        | 0.2091       | 0.01-0.30         |
        'input_scale': 3.2,     # | 3.2         | 3.5479       | 1.23-4.92         |
        'leak_rate': 0.79,      # | 0.79        | 0.8789       | 0.51-0.98         |
        'reg': 0.19,            # | 0.19        | 0.1260       | 0.05-0.30         |
        'washout': 250,         # | 250         | 397          | 50-494            |
        'train_len': 12000,     # | 12000       | 10727        | 5307-19823        |
        'test_len': 3400,       # | 3400        | 3450         | 1176-4974         |
        'seed': 1001,
        'device': 'cuda'
    }


def train_esn(cfg, train_in, train_out):
    esn = ESN(n_in=1, n_res=cfg['n_res'], n_out=1, spectral_radius=cfg['rho'], density=cfg['density'], input_scale=cfg['input_scale'], leak_rate=cfg['leak_rate'], reg=cfg['reg'], seed=cfg['seed'], device=cfg['device'])
    esn.fit(train_in, train_out, washout=cfg['washout'])
    return esn


def evaluate_esn(esn, test_in, test_out):
    pred = esn.predict(test_in)
    error = nrmse(pred, test_out)
    return pred, error


def main():
    cfg = config()
    print_cfg(cfg)

    train_in, train_out, test_in, test_out = prepare_data(cfg)
    esn = train_esn(cfg, train_in, train_out)
    pred, error = evaluate_esn(esn, test_in, test_out)

    print(f"NRMSE: {error:.6f} (expected: 0.03-0.05)")

    generate_all_visualizations(esn, pred, test_out, error, test_in, cfg)


if __name__ == "__main__":
    main()
