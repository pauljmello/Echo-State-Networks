import torch


class ESN(torch.nn.Module):
    """Echo State Network (Jaeger 2001, Lukoševičius 2012)"""
    def __init__(self, n_in, n_res, n_out, spectral_radius=1.0, density=None, input_scale=1.0, leak_rate=0.3, reg=1e-8, seed=None, device='cpu'):
        super().__init__()
        self.n_in = n_in
        self.n_res = n_res
        self.n_out = n_out
        self.leak_rate = leak_rate
        self.reg = reg
        self.device = torch.device(device)

        if seed is not None:
            torch.manual_seed(seed)

        self.W_in = self._init_input_weights(n_res, n_in, input_scale)
        self.W = self._init_reservoir_weights(n_res, density, spectral_radius)
        self.bias = (torch.rand(n_res, device=self.device) * 2 - 1) * input_scale
        self.W_out = None

    def _init_input_weights(self, n_res, n_in, scale):
        """W_in ~ Uniform(-scale, scale)"""
        return (torch.rand(n_res, n_in, device=self.device) * 2 - 1) * scale

    def _init_reservoir_weights(self, n_res, density, rho):
        """W with spectral radius"""
        if density is None:
            W = torch.rand(n_res, n_res, device=self.device) * 2 - 1
        else:
            n_conn = int(n_res * n_res * density)
            W = torch.zeros(n_res, n_res, device=self.device)
            idx = torch.randperm(n_res * n_res, device=self.device)[:n_conn]
            W.view(-1)[idx] = torch.rand(n_conn, device=self.device) * 2 - 1

        eigs = torch.linalg.eigvals(W.cpu())
        radius = torch.max(torch.abs(eigs)).item()

        if radius > 0:
            return W * (rho / radius)
        else:
            return W

    def forward(self, u, x):
        """x(t+1) = (1-a)x(t) + a*tanh(W_in*u + W*x + b)"""
        return (1 - self.leak_rate) * x + self.leak_rate * torch.tanh(self.W_in @ u + self.W @ x + self.bias)

    def run(self, inputs, washout=0, x0=None):
        """Collect states"""
        states = []

        if x0 is None:
            x = torch.zeros(self.n_res, device=self.device)
        else:
            x = x0.clone()

        for t in range(len(inputs)):
            x = self.forward(inputs[t], x)
            if t >= washout:
                states.append(x)
        return torch.stack(states), x

    def fit(self, inputs, targets, washout=0):
        """Ridge regression: W_out = (X^T*X + lambda*I)^-1 * X^T*Y"""
        X, _ = self.run(inputs, washout)
        X = torch.cat([torch.ones(len(X), 1, device=self.device), X], dim=1)
        Y = targets[washout:len(inputs)]
        self.W_out = torch.linalg.solve(X.T @ X + self.reg * torch.eye(X.shape[1], device=self.device), X.T @ Y)

    def predict(self, inputs, x0=None):
        """y(t) = W_out * [1, x(t)]"""
        X, _ = self.run(inputs, 0, x0)
        X = torch.cat([torch.ones(len(X), 1, device=self.device), X], dim=1)
        return X @ self.W_out
