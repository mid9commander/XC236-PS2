import numpy as np
import torch
import os
script_directory = os.path.dirname(os.path.abspath(__file__))

if 'solution' in script_directory:
    from solution import utils as ut
    from solution.models import nns
else:
    from submission import utils as ut
    from submission.models import nns

from torch import nn

class GMVAE(nn.Module):
    def __init__(self, nn='v1', z_dim=2, k=500, name='gmvae'):
        super().__init__()
        self.name = name
        self.k = k
        self.z_dim = z_dim
        nn = getattr(nns, nn)
        self.enc = nn.Encoder(self.z_dim) # from ./nns/v1.py
        self.dec = nn.Decoder(self.z_dim) # from ./nns/v1.py

        # Mixture of Gaussians prior
        self.z_pre = torch.nn.Parameter(torch.randn(1, 2 * self.k, self.z_dim)
                                        / np.sqrt(self.k * self.z_dim))
        # Uniform weighting
        self.pi = torch.nn.Parameter(torch.ones(k) / k, requires_grad=False)

    def negative_elbo_bound(self, x):
        """
        Computes the Evidence Lower Bound, KL and, Reconstruction costs

        Args:
            x: tensor: (batch, dim): Observations

        Returns:
            nelbo: tensor: (): Negative evidence lower bound
            kl: tensor: (): ELBO KL divergence to prior
            rec: tensor: (): ELBO Reconstruction term
        """
        ################################################################################
        # TODO: Modify/complete the code here
        # Compute negative Evidence Lower Bound and its KL and Rec decomposition
        #
        # To help you start, we have computed the mixture of Gaussians prior
        # prior = (m_mixture, v_mixture) for you, where
        # m_mixture and v_mixture each have shape (1, self.k, self.z_dim)
        #
        # Note that nelbo = kl + rec
        #
        # Outputs should all be tensor scalars
        #
        # Return:
        #   nelbo, kl_z, rec
        ################################################################################
        # We provide the learnable prior for you. Familiarize yourself with
        # this object by checking its shape.
        m_prior, logvar_prior = ut.gaussian_parameters(self.z_pre, dim=1)
        m, logvar = self.enc(x)
        z = ut.sample_gaussian(m, logvar)

        x_recon_logits = self.dec(z)

        rec = -ut.log_bernoulli_with_logits(x, x_recon_logits)

        log_normal = ut.log_normal(z, m, logvar)

        batch_size = x.shape[0]
        mix = m_prior.shape[1]

        m_prior = m_prior.expand(batch_size, mix, -1)
        logvar_prior = logvar_prior.expand(batch_size, mix, -1)

        log_normal_mixture = ut.log_normal_mixture(z, m_prior, logvar_prior) 
        #log_normal_mixture = ut.log_normal_mixture(z, m, logvar) # this is wrong

        kl = log_normal - log_normal_mixture
        nelbo = rec + kl
        nelbo = torch.mean(nelbo)
        kl = torch.mean(kl)
        rec = torch.mean(rec)
        return nelbo, kl, rec
        ################################################################################
        # End of code modification
        ################################################################################
        raise NotImplementedError

    def negative_iwae_bound(self, x, iw):
        """
        Computes the Importance Weighted Autoencoder Bound
        Additionally, we also compute the ELBO KL and reconstruction terms

        Args:
            x: tensor: (batch, dim): Observations
            iw: int: (): Number of importance weighted samples

        Returns:
            niwae: tensor: (): Negative IWAE bound
            kl: tensor: (): ELBO KL divergence to prior
            rec: tensor: (): ELBO Reconstruction term
        """
        ################################################################################
        # TODO: Modify/complete the code here
        # Compute niwae (negative IWAE) with iw importance samples, and the KL
        # and Rec decomposition of the Evidence Lower Bound
        #
        # Outputs should all be tensor scalars
        #
        # Return:
        #   niwae, kl, rec
        ################################################################################
        # We provide the learnable prior for you. Familiarize yourself with
        # this object by checking its shape.
        m_prior, logvar_prior = ut.gaussian_parameters(self.z_pre, dim=1)
        batch_size = x.shape[0]
        data_dimension = x.shape[1]

        m, logvar = self.enc(x)

        samples = []
        for _ in range(iw):
            z_single = ut.sample_gaussian(m, logvar)
            samples.append(z_single)
        z = torch.stack(samples, dim=1) # batch, iw, dim
        
        x_recon_logits = self.dec(z)

        x_expanded = x.unsqueeze(1).expand(batch_size, iw, data_dimension) 

        # Flatten both for log_bernoulli_with_logits function
        x_flat = x_expanded.reshape(batch_size * iw, data_dimension)
        logits_flat = x_recon_logits.reshape(batch_size * iw, data_dimension)

        # Compute log p(x|z)        
        log_p_x_given_z_flat = ut.log_bernoulli_with_logits(x_flat, logits_flat)
        log_p_x_given_z = log_p_x_given_z_flat.view(batch_size, iw)

        k = m_prior.shape[1]
        m_prior_exp = m_prior.expand(batch_size, k, -1)            
        logvar_prior_exp = logvar_prior.expand(batch_size, k, -1)  


        log_pz_all = ut.log_normal_mixture(z.unsqueeze(2), m_prior_exp.unsqueeze(1), torch.exp(logvar_prior_exp.unsqueeze(1)))
        # log_pz = ut.log_normal(z, m_prior_reshape, torch.exp(logvar_prior_reshape))
        log_pz = ut.log_sum_exp(log_pz_all, dim=-1) - np.log(self.k)
        del log_pz_all, m_prior_exp, logvar_prior_exp

        m_enc_exp = m.unsqueeze(1).expand(batch_size, iw, -1)       
        logvar_enc_exp = logvar.unsqueeze(1).expand(batch_size, iw, -1)

        log_q_z_given_x = ut.log_normal_mixture(
            z, 
            m_enc_exp, 
            logvar_enc_exp
        )        
        del m_enc_exp, logvar_enc_exp

        # print("log_p_x_given_z mean:", log_p_x_given_z.mean().item())
        # print("log_pz mean:", log_pz.mean().item())
        # print("log_q_z_given_x mean:", log_q_z_given_x.mean().item())        

        log_w = log_p_x_given_z + log_pz - log_q_z_given_x
        del log_p_x_given_z, log_pz, log_q_z_given_x

        iwae_bound = ut.log_mean_exp(log_w, dim=1)  
       

        niwae = -torch.mean(iwae_bound)
        del log_w, iwae_bound
        # rec = -torch.mean(log_p_x_given_z) 
        # kl  =  torch.mean(log_q_z_given_x - log_pz)

        rec = -torch.mean(x_recon_logits)  # Alternatively, use -mean(log_p_x_given_z) if computed per sample.
        kl = torch.tensor(0.0, device=x.device)  # If not

        print("niwae", niwae, "kl", kl, "rec", rec)
        return niwae, kl, rec

        ### START CODE HERE ###
        ### END CODE HERE ###
        ################################################################################
        # End of code modification
        ################################################################################
        raise NotImplementedError

    def loss(self, x):
        nelbo, kl, rec = self.negative_elbo_bound(x)
        loss = nelbo

        summaries = dict((
            ('train/loss', nelbo),
            ('gen/elbo', -nelbo),
            ('gen/kl_z', kl),
            ('gen/rec', rec),
        ))

        return loss, summaries

    def sample_sigmoid(self, batch):
        z = self.sample_z(batch)
        return self.compute_sigmoid_given(z)

    def compute_sigmoid_given(self, z):
        logits = self.dec(z)
        return torch.sigmoid(logits)

    def sample_z(self, batch):
        m, v = ut.gaussian_parameters(self.z_pre.squeeze(0), dim=0)
        idx = torch.distributions.categorical.Categorical(self.pi).sample((batch,))
        m, v = m[idx], v[idx]
        return ut.sample_gaussian(m, v)

    def sample_x(self, batch):
        z = self.sample_z(batch)
        return self.sample_x_given(z)

    def sample_x_given(self, z):
        return torch.bernoulli(self.compute_sigmoid_given(z))
