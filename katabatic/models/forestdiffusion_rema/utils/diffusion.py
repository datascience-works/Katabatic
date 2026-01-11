import numpy as np
import abc
import functools

class SDE(abc.ABC):
    """SDE abstract class. Functions are designed for a mini-batch of inputs."""

    def __init__(self, N):
        super().__init__()
        self.N = N

    @property
    @abc.abstractmethod
    def T(self):
        pass

    @abc.abstractmethod
    def sde(self, x, t):
        pass

    @abc.abstractmethod
    def marginal_prob(self, x, t):
        pass

    @abc.abstractmethod
    def prior_sampling(self, shape):
        pass

    def reverse(self, score_fn, probability_flow=False):
        N = self.N
        T = self.T
        sde_fn = self.sde

        class RSDE(self.__class__):
            def __init__(self):
                self.N = N
                self.probability_flow = probability_flow

            @property
            def T(self):
                return T

            def sde(self, x, t):
                drift, diffusion = sde_fn(x, t)
                score = score_fn(y=x, t=t)
                drift = drift - (diffusion ** 2) * (score * (0.5 if self.probability_flow else 1.0))
                diffusion = np.zeros_like(diffusion) if self.probability_flow else diffusion
                return drift, diffusion

        return RSDE()

class VPSDE(SDE):
    def __init__(self, beta_min=0.1, beta_max=20, N=1000):
        super().__init__(N)
        self.beta_0 = beta_min
        self.beta_1 = beta_max
        self.N = N
        self.discrete_betas = np.linspace(beta_min / N, beta_max / N, N)
        self.alphas = 1.0 - self.discrete_betas
        self.alphas_cumprod = np.cumprod(self.alphas, axis=0)

    @property
    def T(self):
        return 1.0

    def sde(self, x, t):
        beta_t = self.beta_0 + t * (self.beta_1 - self.beta_0)
        drift = -0.5 * beta_t * x
        diffusion = np.sqrt(beta_t)
        return drift, diffusion

    def marginal_prob(self, x, t):
        log_mean_coeff = -0.25 * t**2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
        mean = np.exp(log_mean_coeff) * x
        std = np.sqrt(1.0 - np.exp(2.0 * log_mean_coeff))
        return mean, std

    def marginal_prob_coef(self, x, t):
        log_mean_coeff = -0.25 * t**2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
        mean = np.exp(log_mean_coeff)
        std = np.sqrt(1.0 - np.exp(2.0 * log_mean_coeff))
        return mean, std

    def prior_sampling(self, shape):
        return np.random.normal(size=shape)

class Predictor(abc.ABC):
    def __init__(self, sde, score_fn):
        super().__init__()
        self.sde = sde
        self.rsde = sde.reverse(score_fn)
        self.score_fn = score_fn

    @abc.abstractmethod
    def update_fn(self, x, t):
        pass

class EulerMaruyamaPredictor(Predictor):
    def __init__(self, sde, score_fn):
        super().__init__(sde, score_fn)

    def update_fn(self, x, t, h):
        my_sde = self.rsde.sde
        z = self.sde.prior_sampling(x.shape)
        drift, diffusion = my_sde(x, t)
        x_mean = x - drift * h
        x = x_mean + diffusion * np.sqrt(h) * z
        return x, x_mean

def shared_predictor_update_fn(x, t, h=None, sde=None, score_fn=None):
    predictor_obj = EulerMaruyamaPredictor(sde, score_fn)
    return predictor_obj.update_fn(x, t, h)

def get_pc_sampler(score_fn, sde, denoise=True, eps=1e-3, repaint=False):
    predictor_update_fn = functools.partial(shared_predictor_update_fn, sde=sde, score_fn=score_fn)

    def pc_sampler(prior, r=5, j=5):
        x = prior
        timesteps = np.linspace(sde.T, eps, sde.N)
        h = timesteps - np.append(timesteps, 0)[1:]
        N = sde.N - 1

        for i in range(N):
            x, _ = predictor_update_fn(x, timesteps[i], h[i])

        if denoise:
            _, std = sde.marginal_prob(x, eps)
            x = x + (std**2) * score_fn(y=x, t=eps)

        return x

    # repaint version not needed unless you use it
    return pc_sampler
