import torch

from esn import ESN
from utils import nrmse, prepare_data


def log_test(test_name, passed, details=""):
    if passed:
        status = "PASS"
    else:
        status = "FAIL"
    print(f"{status}: {test_name}")
    if details:
        print(f"    {details}")
    print()


def test_reservoir_weights():
    print("TEST 1: Reservoir Weight Initialization")

    n_res = 200
    rho = 0.9
    density = 0.1

    esn = ESN(n_in=1, n_res=n_res, n_out=1, spectral_radius=rho, density=density, seed=42, device='cpu')

    W = esn.W

    # Test: sparsity matches density
    n_nonzero = torch.count_nonzero(W).item()
    expected_nonzero = int(n_res * n_res * density)
    sparsity_error = abs(n_nonzero - expected_nonzero)
    test_density = sparsity_error <= 1
    log_test("Reservoir sparsity (density)", test_density, f"nonzero = {n_nonzero}, expected = {expected_nonzero}")

    # Test: spectral radius ~= rho
    eigs = torch.linalg.eigvals(W)
    actual_rho = torch.max(torch.abs(eigs)).item()
    rho_error = abs(actual_rho - rho)
    test_rho = rho_error < 0.01
    log_test("Spectral radius rho", test_rho, f"actual = {actual_rho:.6f}, target = {rho}, error = {rho_error:.6f}")

    # Test: non-zero values in valid range
    W_nonzero = W[W != 0]
    min_val = W_nonzero.min().item()
    max_val = W_nonzero.max().item()
    test_values = (min_val >= -rho - 0.1) and (max_val <= rho + 0.1)
    log_test("Non-zero weight values in valid range", test_values, f"min = {min_val:.4f}, max = {max_val:.4f}")

    return test_density and test_rho and test_values


def test_forward_dynamics():
    print("TEST 2: Forward Dynamics (Reservoir Update)")

    n_in, n_res = 2, 50
    leak_rate = 0.8
    esn = ESN(n_in=n_in, n_res=n_res, n_out=1, leak_rate=leak_rate, seed=42, device='cpu')

    u = torch.randn(n_in)
    x = torch.randn(n_res)

    x_next = esn.forward(u, x)

    # Manual computation: x(t+1) = (1-α)x(t) + α*tanh(W_in*u + W*x + b)
    pre_activation = esn.W_in @ u + esn.W @ x + esn.bias
    activation = torch.tanh(pre_activation)
    x_next_manual = (1 - leak_rate) * x + leak_rate * activation

    # Test: forward matches manual computation
    test_forward = torch.allclose(x_next, x_next_manual, atol=1e-6)
    max_diff = torch.max(torch.abs(x_next - x_next_manual)).item()
    log_test("Forward dynamics equation", test_forward, f"Max difference = {max_diff:.2e}")

    # Test: output shape
    test_shape = x_next.shape == (n_res,)
    log_test("Forward output shape", test_shape, f"shape = {x_next.shape}")

    # Test: leak_rate = 1.0 (full update)
    esn_full = ESN(n_in=n_in, n_res=n_res, n_out=1, leak_rate=1.0, seed=42, device='cpu')
    x_full = esn_full.forward(u, x)
    pre_activation_full = esn_full.W_in @ u + esn_full.W @ x + esn_full.bias
    x_expected_full = torch.tanh(pre_activation_full)
    test_full_leak = torch.allclose(x_full, x_expected_full, atol=1e-6)
    log_test("Leak rate = 1.0 (full update)", test_full_leak)

    # Test: leak_rate = 0.0 (no update)
    esn_zero = ESN(n_in=n_in, n_res=n_res, n_out=1, leak_rate=0.0, seed=42, device='cpu')
    x_zero = esn_zero.forward(u, x)
    test_zero_leak = torch.allclose(x_zero, x, atol=1e-6)
    log_test("Leak rate = 0.0 (no update)", test_zero_leak)

    return test_forward and test_shape and test_full_leak and test_zero_leak


def test_run_method():
    print("TEST 3: Run Method (State Collection)")

    n_in, n_res = 1, 50
    T = 200
    washout = 50

    esn = ESN(n_in=n_in, n_res=n_res, n_out=1, seed=42, device='cpu')
    inputs = torch.randn(T, n_in)
    states, final_state = esn.run(inputs, washout=washout)

    # Test: correct number of states collected (T - washout)
    test_count = len(states) == (T - washout)
    log_test("State collection count", test_count, f"collected = {len(states)}, expected = {T - washout}")

    # Test: state shape
    test_shape = states.shape == (T - washout, n_res)
    log_test("Collected states shape", test_shape, f"shape = {states.shape}")

    # Test: final state shape
    test_final_shape = final_state.shape == (n_res,)
    log_test("Final state shape", test_final_shape, f"shape = {final_state.shape}")

    # Test: manual verification of state changes
    x_manual = torch.zeros(n_res)
    for t in range(washout + 1):
        x_manual = esn.forward(inputs[t], x_manual)
    x_after_washout = x_manual.clone()

    # First collected state should match state after washout procedure
    test_washout = torch.allclose(states[0], x_after_washout, atol=1e-5)
    log_test("State after washout", test_washout)

    # Test: custom initial state
    x0 = torch.randn(n_res)
    states_x0, final_x0 = esn.run(inputs, washout=0, x0=x0)
    x_manual_init = x0.clone()
    x_manual_init = esn.forward(inputs[0], x_manual_init)
    test_x0 = torch.allclose(states_x0[0], x_manual_init, atol=1e-5)
    log_test("Custom initial state", test_x0)

    return test_count and test_shape and test_final_shape and test_washout and test_x0


def test_ridge_regression():
    print("TEST 4: Ridge Regression (Training)")

    n_in, n_res, n_out = 1, 50, 1
    T = 200
    washout = 50
    reg = 1e-8

    esn = ESN(n_in=n_in, n_res=n_res, n_out=n_out, reg=reg, seed=42, device='cpu')

    inputs = torch.randn(T, n_in)
    targets = torch.randn(T, n_out)
    esn.fit(inputs, targets, washout=washout)

    # Test: W_out exists after training
    test_exists = esn.W_out is not None
    log_test("W_out created after fit", test_exists)

    # Test: W_out shape (includes bias term)
    expected_shape = (n_res + 1, n_out)
    test_shape = esn.W_out.shape == expected_shape
    log_test("W_out shape", test_shape, f"shape = {esn.W_out.shape}, expected = {expected_shape}")

    # Ridge regression
    X, _ = esn.run(inputs, washout=washout)
    X_bias = torch.cat([torch.ones(len(X), 1), X], dim=1)
    Y = targets[washout:T]

    # W_out = (X^T*X + lambda*I)^-1 * X^T*Y
    XTX = X_bias.T @ X_bias
    XTX_reg = XTX + reg * torch.eye(X_bias.shape[1])
    XTY = X_bias.T @ Y
    W_out_manual = torch.linalg.solve(XTX_reg, XTY)

    # Test: ridge regression
    test_ridge = torch.allclose(esn.W_out, W_out_manual, atol=1e-5)
    max_diff = torch.max(torch.abs(esn.W_out - W_out_manual)).item()
    log_test("Ridge regression equation", test_ridge, f"Max difference = {max_diff:.2e}")

    # Test: regularization effect
    esn_high_reg = ESN(n_in=n_in, n_res=n_res, n_out=n_out, reg=1e-2, seed=42, device='cpu')
    esn_high_reg.fit(inputs, targets, washout=washout)

    # Higher regularization should lead to smaller weights
    norm_low = torch.norm(esn.W_out).item()
    norm_high = torch.norm(esn_high_reg.W_out).item()
    test_reg = norm_high < norm_low
    log_test("Regularization effect (higher lambda -> smaller weights (||W_out||))", test_reg, f"lambda=1e-8: ||W|| = {norm_low:.4f}, lambda=1e-2: ||W|| = {norm_high:.4f}")

    return test_exists and test_shape and test_ridge and test_reg


def test_prediction():
    print("TEST 5: Prediction")

    n_in, n_res, n_out = 1, 50, 1
    T_train, T_test = 200, 100
    washout = 50

    esn = ESN(n_in=n_in, n_res=n_res, n_out=n_out, seed=42, device='cpu')

    train_inputs = torch.randn(T_train, n_in)
    train_targets = torch.randn(T_train, n_out)
    esn.fit(train_inputs, train_targets, washout=washout)

    # Predict
    test_inputs = torch.randn(T_test, n_in)
    predictions = esn.predict(test_inputs)

    # Test: prediction shape
    test_shape = predictions.shape == (T_test, n_out)
    log_test("Prediction shape", test_shape, f"shape = {predictions.shape}, expected = ({T_test}, {n_out})")

    # Manual computation: y(t) = W_out * [1, x(t)]
    X, _ = esn.run(test_inputs, washout=0)
    X_bias = torch.cat([torch.ones(len(X), 1), X], dim=1)
    pred_manual = X_bias @ esn.W_out

    # Test: prediction equation
    test_pred = torch.allclose(predictions, pred_manual, atol=1e-5)
    max_diff = torch.max(torch.abs(predictions - pred_manual)).item()
    log_test("Prediction equation", test_pred, f"Max difference = {max_diff:.2e}")

    # Test: custom initial state
    x0 = torch.randn(n_res)
    pred_x0 = esn.predict(test_inputs, x0=x0)
    X_x0, _ = esn.run(test_inputs, washout=0, x0=x0)
    X_x0_bias = torch.cat([torch.ones(len(X_x0), 1), X_x0], dim=1)
    pred_x0_manual = X_x0_bias @ esn.W_out
    test_x0 = torch.allclose(pred_x0, pred_x0_manual, atol=1e-5)
    log_test("Prediction with custom initial state", test_x0)

    return test_shape and test_pred and test_x0


def test_end_to_end_mackey_glass():
    print("TEST 6: End-to-End Mackey-Glass Prediction")

    cfg = {
        'n_res': 500,
        'rho': 0.92,
        'density': 0.20,
        'input_scale': 3.0,
        'leak_rate': 0.80,
        'reg': 0.10,
        'washout': 300,
        'train_len': 5000,
        'test_len': 1000,
        'seed': 1001,
        'device': 'cuda'
    }

    train_in, train_out, test_in, test_out = prepare_data(cfg)
    esn = ESN(n_in=1, n_res=cfg['n_res'], n_out=1, spectral_radius=cfg['rho'], density=cfg['density'], input_scale=cfg['input_scale'], leak_rate=cfg['leak_rate'], reg=cfg['reg'], seed=cfg['seed'], device=cfg['device'])
    esn.fit(train_in, train_out, washout=cfg['washout'])

    # Predict
    pred = esn.predict(test_in)
    error = nrmse(pred, test_out)

    # Test: NRMSE
    test_performance = error < 0.25
    log_test("Prediction accuracy (NRMSE < 0.25)", test_performance, f"NRMSE = {error:.6f}")

    # Test: predictions (not NaN, not exploding)
    test_valid = torch.isfinite(pred).all().item()
    log_test("Predictions without (NaN/Inf)", test_valid)

    # Test: predictions (reasonable range)
    min_pred = pred.min().item()
    max_pred = pred.max().item()
    test_pred_range = (min_pred > -1) and (max_pred < 3)
    log_test("Predictions range (-1 < x < 3)", test_pred_range, f"min = {min_pred:.4f}, max = {max_pred:.4f}")

    # Test: correlation with target
    pred_flat = pred.flatten()
    target_flat = test_out.flatten()
    correlation = torch.corrcoef(torch.stack([pred_flat, target_flat]))[0, 1].item()
    test_corr = correlation > 0.9
    log_test("High correlation with target (r > 0.9)", test_corr, f"correlation = {correlation:.4f}")

    print(f"Final Performance Summary:")
    print(f"\tNRMSE: {error:.6f}")
    print(f"\tCorrelation: {correlation:.4f}")

    return test_performance and test_valid and test_pred_range and test_corr


def run_all_tests():
    print("\n")

    print("Testing Echo State Networks")

    print("\n")

    results = {}

    results['reservoir_weights'] = test_reservoir_weights()
    results['forward_dynamics'] = test_forward_dynamics()
    results['run_method'] = test_run_method()
    results['ridge_regression'] = test_ridge_regression()
    results['prediction'] = test_prediction()
    results['end_to_end'] = test_end_to_end_mackey_glass()

    print("\nSummary of Tests:")

    total = len(results)
    passed = sum(results.values())
    failed = total - passed

    print(f"\nTotal Tests: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Success Rate: {100 * passed / total:.1f}%\n")

    if failed > 0:
        print("Failed Tests:")
        for name, result in results.items():
            if not result:
                print(f"\t[X] {name}")
    else:
        print("All tests passed")

    return passed == total


if __name__ == "__main__":
    all_passed = run_all_tests()
