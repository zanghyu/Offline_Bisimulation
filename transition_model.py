
import random
import torch
import torch.nn as nn
import numpy as np


def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        # delta-orthogonal init from https://arxiv.org/pdf/1806.05393.pdf
        assert m.weight.size(2) == m.weight.size(3)
        m.weight.data.fill_(0.0)
        m.bias.data.fill_(0.0)
        mid = m.weight.size(2) // 2
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)


class DeterministicTransitionModel(nn.Module):

    def __init__(self, encoder_feature_dim, action_shape, layer_width, device):
        super().__init__()
        self.fc = nn. Linear(encoder_feature_dim +
                             action_shape, layer_width)
        self.ln = nn.LayerNorm(layer_width)
        self.fc_mu = nn.Linear(layer_width, encoder_feature_dim)
        self.device = device
        print("Deterministic transition model chosen.")
        self.apply(weight_init)

    def forward(self, x):
        x = self.fc(x)
        x = self.ln(x)
        x = torch.relu(x)

        mu = self.fc_mu(x)
        sigma = None
        return mu, sigma

    def sample_prediction(self, x):
        mu, sigma = self(x)
        return mu


class ProbabilisticTransitionModel(nn.Module):

    def __init__(self, encoder_feature_dim, action_shape, layer_width, device, announce=True, max_sigma=1e1, min_sigma=1e-4):
        super().__init__()
        self.fc = nn. Linear(encoder_feature_dim +
                             action_shape, layer_width)
        self.ln = nn.LayerNorm(layer_width)
        self.fc_mu = nn.Linear(layer_width, encoder_feature_dim)
        self.fc_sigma = nn.Linear(layer_width, encoder_feature_dim)

        self.max_sigma = max_sigma
        self.min_sigma = min_sigma

        self.device = device
        assert(self.max_sigma >= self.min_sigma)
        if announce:
            print("Probabilistic transition model chosen.")
        self.apply(weight_init)

    def forward(self, x):
        x = self.fc(x)
        x = self.ln(x)
        x = torch.relu(x)

        mu = self.fc_mu(x)
        sigma = torch.sigmoid(self.fc_sigma(x))  # range (0, 1.)
        # scaled range (min_sigma, max_sigma)
        sigma = self.min_sigma + (self.max_sigma - self.min_sigma) * sigma
        return mu, sigma

    def sample_prediction(self, x):
        mu, sigma = self(x)
        eps = torch.randn_like(sigma)
        return mu + sigma * eps


class ProbabilisticTransitionModel2(nn.Module):
    def __init__(self, encoder_feature_dim, action_shape, layer_width, device, announce=True, max_sigma=1e1, min_sigma=1e-4):
        super().__init__()
        self.fc = nn. Linear(encoder_feature_dim +
                             action_shape, layer_width)

        self.log_std_min = np.log(min_sigma)
        self.log_std_max = np.log(max_sigma)

        self.trunk = nn.Sequential(
            nn.Linear(encoder_feature_dim +
                      action_shape, layer_width), nn.ReLU(),
            nn.Linear(layer_width, layer_width), nn.ReLU(),
            nn.Linear(layer_width, 2 * encoder_feature_dim)
        )
        # self.ln = nn.LayerNorm(layer_width)
        # self.fc_mu = nn.Linear(layer_width, encoder_feature_dim)
        # self.fc_sigma = nn.Linear(layer_width, encoder_feature_dim)
        # self.max_sigma = max_sigma
        # self.min_sigma = min_sigma
        assert(self.log_std_max >= self.log_std_min)
        self.device = device
        if announce:
            print("Probabilistic transition model chosen.")
        self.apply(weight_init)

    def forward(self, x):

        mu, log_std = self.trunk(x).chunk(2, dim=-1)

        # constrain log_std inside [log_std_min, log_std_max]
        log_std = torch.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (
            self.log_std_max - self.log_std_min
        ) * (log_std + 1)

        sigma = log_std.exp()
        return mu, sigma

    def sample_prediction(self, x):
        mu, sigma = self(x)
        eps = torch.randn_like(sigma)
        return mu + sigma * eps


class EnsembleOfProbabilisticTransitionModels(object):

    def __init__(self, encoder_feature_dim, action_shape, layer_width, device, ensemble_size=5):
        self.models = [ProbabilisticTransitionModel2(encoder_feature_dim, action_shape, layer_width, device, announce=False)
                       for _ in range(ensemble_size)]
        self.device = device
        print("Ensemble of probabilistic transition models chosen.")

    def __call__(self, x):
        mu_sigma_list = [model.forward(x) for model in self.models]
        mus, sigmas = zip(*mu_sigma_list)
        mus, sigmas = torch.stack(mus), torch.stack(sigmas)
        return mus, sigmas

    def sample_prediction(self, x):
        model = random.choice(self.models)
        return model.sample_prediction(x)

    def to(self, device):
        for model in self.models:
            model.to(device)
        return self

    def parameters(self):
        list_of_parameters = [list(model.parameters())
                              for model in self.models]
        parameters = [p for ps in list_of_parameters for p in ps]
        return parameters









def sample_vMF(mu, kappa):
    """Generate num_samples N-dimensional samples from von Mises Fisher
    distribution around center mu \in R^N with concentration kappa.
    """
    dim = mu.shape[-1]
    mu = mu.numpy()
    # sample offset from center (on sphere) with spread kappa
    w = _sample_weight(kappa, dim)

    # sample a point v on the unit sphere that's orthogonal to mu
    v = _sample_orthonormal_to(mu)

    # compute new point
    result = v * np.sqrt(1. - w ** 2) + w * mu

    return result


def _sample_weight(kappa, dim):
    """Rejection sampling scheme for sampling distance from center on
    surface of the sphere.
    """
    dim = dim - 1  # since S^{n-1}
    b = dim / (np.sqrt(4. * kappa ** 2 + dim ** 2) + 2 * kappa)
    x = (1. - b) / (1. + b)
    c = kappa * x + dim * np.log(1 - x ** 2)

    while True:
        z = np.random.beta(dim / 2., dim / 2.)
        w = (1. - (1. + b) * z) / (1. - (1. - b) * z)
        u = np.random.uniform(low=0, high=1)
        if kappa * w + dim * np.log(1. - x * w) - c >= np.log(u):
            return w


def _sample_orthonormal_to(mu):
    """Sample point on sphere orthogonal to mu."""
    v = np.random.randn(mu.shape[0])
    proj_mu_v = mu * np.dot(mu, v) / np.linalg.norm(mu)
    orthto = v - proj_mu_v
    return orthto / np.linalg.norm(orthto)

class VMFProbabilisticTransitionModel(nn.Module):
    def __init__(self, encoder_feature_dim, action_shape, layer_width, device, announce=True):
        super().__init__()

        self.trunk = nn.Sequential(
            nn.Linear(encoder_feature_dim + action_shape, layer_width), nn.ReLU(),
            nn.Linear(layer_width, layer_width), nn.ReLU(),
            nn.Linear(layer_width, encoder_feature_dim)
        )

        if announce:
            print("Probabilistic transition model chosen.")
        self.apply(weight_init)

    def forward(self, x):

        mu = self.trunk(x)
        mu = nn.functional.normalize(mu, p=2)
        return mu

    def sample_prediction(self, x):
        mu = torch.Tensor.cpu(self(x))
        result = torch.ones_like(mu)
        for i in range(128):
            mu[i] = nn.functional.normalize(mu[i], p=2, dim=0)
            t = sample_vMF(mu[i], kappa=30)
            result[i] = torch.from_numpy(t)
        return result.to(device="cuda")





# class VMFProbabilisticTransitionModel(nn.Module):
#     def __init__(self, encoder_feature_dim, action_shape, layer_width, device, announce=True):
#         super().__init__()

#         self.kappa = 30
#         self.norm_eps = 1
#         self.norm_max = 10
#         self.normclip = torch.nn.Hardtanh(0, 10 - 1)

#         self.trunk = nn.Sequential(
#             nn.Linear(encoder_feature_dim + action_shape, layer_width), nn.ReLU(),
#             nn.Linear(layer_width, layer_width), nn.ReLU(),
#             nn.Linear(layer_width, encoder_feature_dim)
#         )
#         self.device = device

#         if announce:
#             print("Probabilistic transition model chosen.")
#         self.apply(weight_init)

#     def forward(self, x):

#         mu = self.trunk(x)
#         mu = nn.functional.normalize(mu, p=2)
#         return mu

#     def sample_prediction(self, x):
#         """vMF sampler in pytorch.
#         http://stats.stackexchange.com/questions/156729/sampling-from-von-mises-fisher-distribution-in-python
#         Args:
#             mu (Tensor): of shape (batch_size, 2*word_dim)
#             kappa (Float): controls dispersion. kappa of zero is no dispersion.
#         """
#         mu = self(x)
#         batch_size, id_dim = mu.size()
#         result_list = []
#         for i in range(batch_size):
#             munorm = mu[i].norm(p=2).expand(id_dim)
#             munoise = self.add_norm_noise(munorm, self.norm_eps)
#             if float(mu[i].norm().data.cpu().numpy()) > 1e-10:
#                 # sample offset from center (on sphere) with spread kappa
#                 w = self._sample_weight(self.kappa, id_dim)
#                 wtorch = torch.autograd.Variable(w * torch.ones(id_dim)).to(self.device)

#                 # sample a point v on the unit sphere that's orthogonal to mu
#                 v = self._sample_orthonormal_to(mu[i] / munorm, id_dim)

#                 # compute new point
#                 scale_factr = torch.sqrt(torch.autograd.Variable(torch.ones(id_dim)).to(self.device) - torch.pow(wtorch, 2))
#                 orth_term = v * scale_factr
#                 muscale = mu[i] * wtorch / munorm
#                 sampled_vec = (orth_term + muscale) * munoise
#             else:
#                 rand_draw = Variable(torch.randn(id_dim))
#                 rand_draw = rand_draw / torch.norm(rand_draw, p=2).expand(id_dim)
#                 rand_norms = (torch.rand(1) * self.norm_eps).expand(id_dim)
#                 sampled_vec = rand_draw * Variable(rand_norms)  # mu[i]
#             result_list.append(sampled_vec)

#         return torch.stack(result_list, 0)

#     def _sample_orthonormal_to(self, mu, dim):
#         """Sample point on sphere orthogonal to mu.
#         """
#         v = torch.autograd.Variable(torch.randn(dim)).to(self.device)
#         rescale_value = mu.dot(v) / mu.norm()
#         proj_mu_v = mu * rescale_value.expand(dim)
#         ortho = v - proj_mu_v
#         ortho_norm = torch.norm(ortho)
#         return ortho / ortho_norm.expand_as(ortho)

#     def add_norm_noise(self, munorm, eps):
#         """
#         KL loss is - log(maxvalue/eps)
#         cut at maxvalue-eps, and add [0,eps] noise.
#         """
#         trand = torch.rand(1).expand(munorm.size()) * eps
#         return (self.normclip(munorm) + torch.autograd.Variable(trand).to(self.device))

#     def _sample_weight(self, kappa, dim):
#         """Rejection sampling scheme for sampling distance from center on
#         surface of the sphere.
#         """
#         dim = dim - 1  # since S^{n-1}
#         b = dim / (np.sqrt(4. * kappa ** 2 + dim ** 2) + 2 * kappa)  # b= 1/(sqrt(4.* kdiv**2 + 1) + 2 * kdiv)
#         x = (1. - b) / (1. + b)
#         c = kappa * x + dim * np.log(1 - x ** 2)  # dim * (kdiv *x + np.log(1-x**2))

#         while True:
#             z = np.random.beta(dim / 2., dim / 2.)  # concentrates towards 0.5 as d-> inf
#             w = (1. - (1. + b) * z) / (1. - (1. - b) * z)
#             u = np.random.uniform(low=0, high=1)
#             if kappa * w + dim * np.log(1. - x * w) - c >= np.log(
#                     u):  # thresh is dim *(kdiv * (w-x) + log(1-x*w) -log(1-x**2))
#                 return w










# class VMFProbabilisticTransitionModel(nn.Module):
#     def __init__(self, encoder_feature_dim, action_shape, layer_width, device, kappa=20):
#         super().__init__()

#         self.trunk = nn.Sequential(
#             nn.Linear(encoder_feature_dim + action_shape, layer_width), nn.ReLU(),
#             nn.Linear(layer_width, layer_width), nn.ReLU(),
#             nn.Linear(layer_width, encoder_feature_dim)
#         )
#         self.apply(weight_init)
#         self.device = device
#         self.encoder_feature_dim = encoder_feature_dim
#         # 预先计算一批w
#         epsilon = 1e-7
#         self.x = x = np.arange(-1 + epsilon, 1, epsilon)
#         y = kappa * x + np.log(1 - x**2) * (self.encoder_feature_dim - 3) / 2
#         y = np.cumsum(np.exp(y - y.max()))
#         self.y = y / y[-1]
#         self.W = torch.Tensor(np.interp(np.random.random(10**6), self.y, self.x)).to(self.device)

#     def forward(self, x):
#         mu = self.trunk(x)
#         mu = nn.functional.normalize(mu, dim=1, p=2)
#         return mu

#     def sample_prediction(self, x):
#         mu = self(x)
#         return self.sampling(mu)

#     def sampling(self, mu, kappa=20):
#         """vMF分布重参数操作
#         """
#         # 实时采样w
#         idx = torch.randint(0, 10**6, mu[:, :1].shape).to(self.device)
#         w = self.W[idx]
#         # 实时采样z
#         eps = torch.randn(mu.shape).to(self.device)
#         nu = eps - torch.sum(eps * mu, dim=1, keepdims=True) * mu
#         nu = nn.functional.normalize(nu, dim=1, p=2)
#         return w * mu + (1 - w**2)**0.5 * nu












_AVAILABLE_TRANSITION_MODELS = {'': DeterministicTransitionModel,
                                'deterministic': DeterministicTransitionModel,
                                'probabilistic': ProbabilisticTransitionModel2,
                                'ensemble': EnsembleOfProbabilisticTransitionModels,
                                'vmf': VMFProbabilisticTransitionModel}


def make_transition_model(transition_model_type, encoder_feature_dim, action_shape, device, layer_width=512):
    assert transition_model_type in _AVAILABLE_TRANSITION_MODELS
    return _AVAILABLE_TRANSITION_MODELS[transition_model_type](
            encoder_feature_dim, action_shape, layer_width, device
    )
