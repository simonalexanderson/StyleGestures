import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.linalg
from . import thops

def nan_throw(tensor, name="tensor"):
        stop = False
        if ((tensor!=tensor).any()):
            print(name + " has nans")
            stop = True
        if (torch.isinf(tensor).any()):
            print(name + " has infs")
            stop = True
        if stop:
            print(name + ": " + str(tensor))
            #raise ValueError(name + ' contains nans of infs')

class _ActNorm(nn.Module):
    """
    Activation Normalization
    Initialize the bias and scale with a given minibatch,
    so that the output per-channel have zero mean and unit variance for that.

    After initialization, `bias` and `logs` will be trained as parameters.
    """

    def __init__(self, num_features, scale=1.):
        super().__init__()
        # register mean and scale
        size = [1, num_features, 1]
        self.register_parameter("bias", nn.Parameter(torch.zeros(*size)))
        self.register_parameter("logs", nn.Parameter(torch.zeros(*size)))
        self.num_features = num_features
        self.scale = float(scale)
        self.inited = False

    def _check_input_dim(self, input):
        return NotImplemented

    def initialize_parameters(self, input):
        self._check_input_dim(input)
        if not self.training:
            return
        assert input.device == self.bias.device
        with torch.no_grad():
            bias = thops.mean(input.clone(), dim=[0, 2], keepdim=True) * -1.0
            vars = thops.mean((input.clone() + bias) ** 2, dim=[0, 2], keepdim=True)
            logs = torch.log(self.scale/(torch.sqrt(vars)+1e-6))
            self.bias.data.copy_(bias.data)
            self.logs.data.copy_(logs.data)
            self.inited = True

    def _center(self, input, reverse=False):
        if not reverse:
            return input + self.bias
        else:
            return input - self.bias

    def _scale(self, input, logdet=None, reverse=False):
        logs = self.logs
        if not reverse:
            input = input * torch.exp(logs)
        else:
            input = input * torch.exp(-logs)
        if logdet is not None:
            """
            logs is log_std of `mean of channels`
            so we need to multiply timesteps
            """
            dlogdet = thops.sum(logs) * thops.timesteps(input)
            if reverse:
                dlogdet *= -1
            logdet = logdet + dlogdet
        return input, logdet

    def forward(self, input, logdet=None, reverse=False):
        if not self.inited:
            self.initialize_parameters(input)
        self._check_input_dim(input)
        # no need to permute dims as old version
        if not reverse:
            # center and scale
            input = self._center(input, reverse)
            input, logdet = self._scale(input, logdet, reverse)
        else:
            # scale and center
            input, logdet = self._scale(input, logdet, reverse)
            input = self._center(input, reverse)
        return input, logdet


class ActNorm2d(_ActNorm):
    def __init__(self, num_features, scale=1.):
        super().__init__(num_features, scale)

    def _check_input_dim(self, input):
        assert len(input.size()) == 3
        assert input.size(1) == self.num_features, (
            "[ActNorm]: input should be in shape as `BCT`,"
            " channels should be {} rather than {}".format(
                self.num_features, input.size()))


class LinearZeros(nn.Linear):
    def __init__(self, in_channels, out_channels, logscale_factor=3):
        super().__init__(in_channels, out_channels)
        self.logscale_factor = logscale_factor
        # set logs parameter
        self.register_parameter("logs", nn.Parameter(torch.zeros(out_channels)))
        # init
        self.weight.data.zero_()
        self.bias.data.zero_()

    def forward(self, input):
        output = super().forward(input)
        return output * torch.exp(self.logs * self.logscale_factor)


class Conv2d(nn.Conv2d):
    pad_dict = {
        "same": lambda kernel, stride: [((k - 1) * s + 1) // 2 for k, s in zip(kernel, stride)],
        "valid": lambda kernel, stride: [0 for _ in kernel]
    }

    @staticmethod
    def get_padding(padding, kernel_size, stride):
        # make paddding
        if isinstance(padding, str):
            if isinstance(kernel_size, int):
                kernel_size = [kernel_size, kernel_size]
            if isinstance(stride, int):
                stride = [stride, stride]
            padding = padding.lower()
            try:
                padding = Conv2d.pad_dict[padding](kernel_size, stride)
            except KeyError:
                raise ValueError("{} is not supported".format(padding))
        return padding

    def __init__(self, in_channels, out_channels,
                 kernel_size=[3, 3], stride=[1, 1],
                 padding="same", do_actnorm=True, weight_std=0.05):
        padding = Conv2d.get_padding(padding, kernel_size, stride)
        super().__init__(in_channels, out_channels, kernel_size, stride,
                         padding, bias=(not do_actnorm))
        # init weight with std
        self.weight.data.normal_(mean=0.0, std=weight_std)
        if not do_actnorm:
            self.bias.data.zero_()
        else:
            self.actnorm = ActNorm2d(out_channels)
        self.do_actnorm = do_actnorm

    def forward(self, input):
        x = super().forward(input)
        if self.do_actnorm:
            x, _ = self.actnorm(x)
        return x


class Conv2dZeros(nn.Conv2d):
    def __init__(self, in_channels, out_channels,
                 kernel_size=[3, 3], stride=[1, 1],
                 padding="same", logscale_factor=3):
        padding = Conv2d.get_padding(padding, kernel_size, stride)
        super().__init__(in_channels, out_channels, kernel_size, stride, padding)
        # logscale_factor
        self.logscale_factor = logscale_factor
        self.register_parameter("logs", nn.Parameter(torch.zeros(out_channels, 1, 1)))
        # init
        self.weight.data.zero_()
        self.bias.data.zero_()

    def forward(self, input):
        output = super().forward(input)
        return output * torch.exp(self.logs * self.logscale_factor)

class LinearNormInit(nn.Linear):
    def __init__(self, in_channels, out_channels, weight_std=0.05):
        super().__init__(in_channels, out_channels)
        # init
        self.weight.data.normal_(mean=0.0, std=weight_std)
        self.bias.data.zero_()
        
class LinearZeroInit(nn.Linear):
    def __init__(self, in_channels, out_channels):
        super().__init__(in_channels, out_channels)
        # init
        self.weight.data.zero_()
        self.bias.data.zero_()

class Permute2d(nn.Module):
    def __init__(self, num_channels, shuffle):
        super().__init__()
        self.num_channels = num_channels
        print(num_channels)
        self.indices = np.arange(self.num_channels - 1, -1,-1).astype(np.long)
        self.indices_inverse = np.zeros((self.num_channels), dtype=np.long)
        print(self.indices_inverse.shape)
        for i in range(self.num_channels):
            self.indices_inverse[self.indices[i]] = i
        if shuffle:
            self.reset_indices()

    def reset_indices(self):
        np.random.shuffle(self.indices)
        for i in range(self.num_channels):
            self.indices_inverse[self.indices[i]] = i

    def forward(self, input, reverse=False):
        assert len(input.size()) == 3
        if not reverse:
            return input[:, self.indices, :]
        else:
            return input[:, self.indices_inverse, :]


class InvertibleConv1x1(nn.Module):
    def __init__(self, num_channels, LU_decomposed=False):
        super().__init__()
        w_shape = [num_channels, num_channels]
        w_init = np.linalg.qr(np.random.randn(*w_shape))[0].astype(np.float32)
        if not LU_decomposed:
            # Sample a random orthogonal matrix:
            self.register_parameter("weight", nn.Parameter(torch.Tensor(w_init)))
        else:
            np_p, np_l, np_u = scipy.linalg.lu(w_init)
            np_s = np.diag(np_u)
            np_sign_s = np.sign(np_s)
            np_log_s = np.log(np.abs(np_s))
            np_u = np.triu(np_u, k=1)
            l_mask = np.tril(np.ones(w_shape, dtype=np.float32), -1)
            eye = np.eye(*w_shape, dtype=np.float32)

            #self.p = torch.Tensor(np_p.astype(np.float32))
            #self.sign_s = torch.Tensor(np_sign_s.astype(np.float32))
            self.register_buffer('p', torch.Tensor(np_p.astype(np.float32)))
            self.register_buffer('sign_s', torch.Tensor(np_sign_s.astype(np.float32)))
            self.l = nn.Parameter(torch.Tensor(np_l.astype(np.float32)))
            self.log_s = nn.Parameter(torch.Tensor(np_log_s.astype(np.float32)))
            self.u = nn.Parameter(torch.Tensor(np_u.astype(np.float32)))
            self.l_mask = torch.Tensor(l_mask)
            self.eye = torch.Tensor(eye)
        self.w_shape = w_shape
        self.LU = LU_decomposed

    def get_weight(self, input, reverse):
        w_shape = self.w_shape
        if not self.LU:
            timesteps = thops.timesteps(input)
            dlogdet = torch.slogdet(self.weight)[1] * timesteps
            if not reverse:
                weight = self.weight.view(w_shape[0], w_shape[1], 1)
            else:
                weight = torch.inverse(self.weight.double()).float()\
                              .view(w_shape[0], w_shape[1], 1)
            return weight, dlogdet
        else:
            self.p = self.p.to(input.device)
            self.sign_s = self.sign_s.to(input.device)
            self.l_mask = self.l_mask.to(input.device)
            self.eye = self.eye.to(input.device)
            l = self.l * self.l_mask + self.eye
            u = self.u * self.l_mask.transpose(0, 1).contiguous() + torch.diag(self.sign_s * torch.exp(self.log_s))
            dlogdet = thops.sum(self.log_s) * thops.timesteps(input)
            if not reverse:
                w = torch.matmul(self.p, torch.matmul(l, u))
            else:
                l = torch.inverse(l.double()).float()
                u = torch.inverse(u.double()).float()
                w = torch.matmul(u, torch.matmul(l, self.p.inverse()))
            return w.view(w_shape[0], w_shape[1], 1), dlogdet

    def forward(self, input, logdet=None, reverse=False):
        """
        log-det = log|abs(|W|)| * timesteps
        """
        weight, dlogdet = self.get_weight(input, reverse)
        nan_throw(weight, "weight")
        nan_throw(dlogdet, "dlogdet")
        
        if not reverse:
            z = F.conv1d(input, weight)
            if logdet is not None:
                logdet = logdet + dlogdet
            return z, logdet
        else:
            nan_throw(input, "InConv input")
            z = F.conv1d(input, weight)
            nan_throw(z, "InConv z")
            nan_throw(logdet, "InConv logdet")
            if logdet is not None:
                logdet = logdet - dlogdet
            return z, logdet

# Here we define our model as a class
class LSTM(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim=1, num_layers=2, dropout=0.0):
        super(LSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Define the LSTM layer
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers, batch_first=True)

        # Define the output layer
        self.linear = LinearZeroInit(self.hidden_dim, output_dim)

        # do_init
        self.do_init = True

    def init_hidden(self):
        # This is what we'll initialise our hidden state as
        self.do_init = True

    def forward(self, input):
        # Forward pass through LSTM layer
        # shape of lstm_out: [batch_size, input_size, hidden_dim]
        # shape of self.hidden: (a, b), where a and b both 
        # have shape (batch_size, num_layers, hidden_dim).
        if self.do_init:
            lstm_out, self.hidden = self.lstm(input)
            self.do_init = False
        else:
            lstm_out, self.hidden = self.lstm(input, self.hidden)
        
        #self.hidden = hidden[0].to(input.device), hidden[1].to(input.device)
        
        # Final layer 
        y_pred = self.linear(lstm_out)
        return y_pred

# Here we define our model as a class
class GRU(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim=1, num_layers=2, dropout=0.0):
        super(GRU, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Define the LSTM layer
        self.gru = nn.GRU(self.input_dim, self.hidden_dim, self.num_layers, batch_first=True)

        # Define the output layer
        self.linear = LinearZeroInit(self.hidden_dim, output_dim)

        # do_init
        self.do_init = True

    def init_hidden(self):
        # This is what we'll initialise our hidden state as
        self.do_init = True

    def forward(self, input):
        # Forward pass through LSTM layer
        # shape of lstm_out: [batch_size, input_size, hidden_dim]
        # shape of self.hidden: (a, b), where a and b both 
        # have shape (batch_size, num_layers, hidden_dim).
        if self.do_init:
            gru_out, self.hidden = self.gru(input)
            self.do_init = False
        else:
            gru_out, self.hidden = self.gru(input, self.hidden)
        
        #self.hidden = hidden[0].to(input.device), hidden[1].to(input.device)
        
        # Final layer 
        y_pred = self.linear(gru_out)
        return y_pred

class GaussianDiag:
    Log2PI = float(np.log(2 * np.pi))

    def likelihood(self,x):
        """
        lnL = -1/2 * { ln|Var| + ((X - Mu)^T)(Var^-1)(X - Mu) + kln(2*PI) }
              k = 1 (Independent)
              Var = logs ** 2
        """
        return -0.5 * (((x) ** 2) + GaussianDiag.Log2PI)

    def logp(self,x):
        likelihood = self.likelihood(x)
        return thops.sum(likelihood, dim=[1, 2])

    def sample(self,z_shape, eps_std=None, device=None):
        eps_std = eps_std or 1
        eps = torch.normal(mean=torch.zeros(z_shape),
                           std=torch.ones(z_shape) * eps_std)
        eps = eps.to(device)
        return eps

class StudentT:

    def __init__(self, df, d):
        self.df=df
        self.d=d
        self.norm_const = scipy.special.loggamma(0.5*(df+d))-scipy.special.loggamma(0.5*df)-0.5*d*np.log(np.pi*df)

    def logp(self,x):
        '''
        Multivariate t-student density:
        output:
            the sum density of the given element
        '''
        #df=100
        #d=x.shape[1]
        #norm_const = scipy.special.loggamma(0.5*(df+d))-scipy.special.loggamma(0.5*df)-0.5*d*np.log(np.pi*df)
        #import pdb; pdb.set_trace()        
        x_norms = thops.sum(((x) ** 2), dim=[1])
        likelihood = self.norm_const-0.5*(self.df+self.d)*torch.log(1+(1/self.df)*x_norms)
        return thops.sum(likelihood, dim=[1])

    def sample(self,z_shape, eps_std=None, device=None):
        '''generate random variables of multivariate t distribution
        Parameters
        ----------
        m : array_like
            mean of random variable, length determines dimension of random variable
        S : array_like
            square array of covariance  matrix
        df : int or float
            degrees of freedom
        n : int
            number of observations, return random array will be (n, len(m))
        Returns
        -------
        rvs : ndarray, (n, len(m))
            each row is an independent draw of a multivariate t distributed
            random variable
        '''
        #df=100
        #import pdb; pdb.set_trace()
        x_shape = torch.Size((z_shape[0], 1, z_shape[2]))
        x = np.random.chisquare(self.df, x_shape)/self.df
        x = np.tile(x, (1,z_shape[1],1))
        x = torch.Tensor(x.astype(np.float32))
        z = torch.normal(mean=torch.zeros(z_shape),std=torch.ones(z_shape) * eps_std)
        
        return (z/torch.sqrt(x)).to(device)

class Split2d(nn.Module):
    def __init__(self, num_channels, distribution):
        super().__init__()
        print("Split2d num_channels:" + str(num_channels))

        self.num_channels = num_channels
        self.conv = Conv2dZeros(num_channels // 2, num_channels)

    def split2d_prior(self, z):
        h = self.conv(z)
        return thops.split_feature(h, "cross")

    def forward(self, input, cond, logdet=0., reverse=False, eps_std=None):
        if not reverse:
            #print("forward Split2d input:" + str(input.shape))
            z1, z2 = thops.split_feature(input, "split")
            #mean, logs = self.split2d_prior(z1)
            logdet = self.distribution.logp(z2) + logdet
            return z1, cond, logdet
        else:
            z1 = input
            #print("reverse Split2d z1.shape:" + str(z1.shape))
            #mean, logs = self.split2d_prior(z1)
            z2_shape = list(z1.shape)
            z2_shape[1] = self.num_channels-z1.shape[1]
            z2 = self.distribution.sample(z2_shape, eps_std, device=input.device)
            z = thops.cat_feature(z1, z2)
            return z, cond, logdet

def squeeze2d(input, factor=2):
    assert factor >= 1 and isinstance(factor, int)
    if factor == 1:
        return input
    size = input.size()
    B = size[0]
    C = size[1]
    H = size[2]
    W = size[3]
    assert H % factor == 0 , "{}".format((H, W))
    x = input.view(B, C, H // factor, factor, W, 1)
    x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
    x = x.view(B, C * factor, H // factor, W)
    return x


def unsqueeze2d(input, factor=2):
    assert factor >= 1 and isinstance(factor, int)
    #factor2 = factor ** 2
    if factor == 1:
        return input
    size = input.size()
    B = size[0]
    C = size[1]
    H = size[2]
    W = size[3]
    assert C % (factor) == 0, "{}".format(C)
    x = input.view(B, C // factor, factor, 1, H, W)
    x = x.permute(0, 1, 4, 2, 5, 3).contiguous()
    x = x.view(B, C // (factor), H * factor, W)
    return x


class SqueezeLayer(nn.Module):
    def __init__(self, factor):
        super().__init__()
        self.factor = factor

    def forward(self, input, cond = None, logdet=None, reverse=False):
        if not reverse:
            output = squeeze2d(input, self.factor)
            cond_out = squeeze2d(cond, self.factor)
            return output, cond_out, logdet
        else:
            output = unsqueeze2d(input, self.factor)
            cond_output = unsqueeze2d(cond, self.factor)
            return output, cond_output, logdet

    def squeeze_cond(self, cond):
        cond_out = squeeze2d(cond, self.factor)
        return cond_out
