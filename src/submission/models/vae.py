import torch
from torch import nn
import os
script_directory = os.path.dirname(os.path.abspath(__file__))

if 'solution' in script_directory:
    from solution import utils as ut
    from solution.models import nns
else:
    from submission import utils as ut
    from submission.models import nns

class VAE(nn.Module):
    def __init__(self, nn='v1', name='vae', z_dim=2):
        super().__init__()
        self.name = name
        self.z_dim = z_dim
        nn = getattr(nns, nn)
        self.enc = nn.Encoder(self.z_dim) # from ./nns/v1.py
        self.dec = nn.Decoder(self.z_dim) # from ./nns/v1.py

        # Set prior as fixed parameter attached to Module
        self.z_prior_m = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self.z_prior_v = torch.nn.Parameter(torch.ones(1), requires_grad=False)
        self.z_prior = (self.z_prior_m, self.z_prior_v)

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
        # Note that nelbo = kl + rec
        #
        # Outputs should all be tensor scalars
        #
        # Return:
        #   nelbo, kl, rec
        ################################################################################
        ### START CODE HERE ###
        m, logvar = self.enc(x)
        z = ut.sample_gaussian(m, logvar)

        x_recon_logits = self.dec(z)

        # Then rec is the negative log-likelihood, summed over features (dim=1)
        rec = -ut.log_bernoulli_with_logits(x, x_recon_logits)

        kl = ut.kl_normal(m, logvar, self.z_prior[0], self.z_prior[1])
        nelbo = rec + kl
        nelbo = torch.mean(nelbo)
        kl = torch.mean(kl)
        rec = torch.mean(rec)
        return nelbo, kl, rec
        ### END CODE HERE ###
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
        #
        # HINT: The summation over m may seem to prevent us from 
        # splitting the ELBO into the KL and reconstruction terms, but instead consider 
        # calculating log_normal w.r.t prior and q
        ################################################################################
        if iw == 10:
            print("Using 10 importance samples")
            
        batch_size = x.shape[0]
        data_dimension = x.shape[1]

        # Encode x to get latent distribution parameters (mean, log variance)
        m, logvar = self.enc(x)

        # Sample multiple z's using reparameterization trick
        samples = []
        for _ in range(iw):
            z_single = ut.sample_gaussian(m, logvar)
            samples.append(z_single)
        z = torch.stack(samples, dim=1) # batch, iw, dim
        z_decoded = self.dec(z)

        # Compute log probabilities
        x_expanded = x.unsqueeze(1).expand(-1, iw, -1)

        # Flatten both for log_bernoulli_with_logits function
        x_flat = x_expanded.reshape(batch_size * iw, data_dimension)
        logits_flat = z_decoded.reshape(batch_size * iw, data_dimension)
        # Compute log p(x|z)
        log_p_x_given_z_flat = ut.log_bernoulli_with_logits(x_flat, logits_flat)
        log_px_z = log_p_x_given_z_flat.view(batch_size, iw)  # reshape back to (batch, iw)
        log_pz = ut.log_normal(z, self.z_prior_m, self.z_prior_v)

        m_expanded = m.unsqueeze(1).expand_as(z)
        logvar_expanded = logvar.unsqueeze(1).expand_as(z)
        log_qz_x = ut.log_normal(z, m_expanded, logvar_expanded)

        log_w = log_px_z + log_pz - log_qz_x  # compute importance weights

        iwae_bound = ut.log_mean_exp(log_w, dim=1) 

        # Compute negative IWAE bound (minimization objective)
        niwae = -torch.mean(iwae_bound)
        kl = torch.mean(log_qz_x - log_pz) # Compute ELBO decomposition (for iw=1 case)
        rec = torch.mean(-log_px_z) # Compute ELBO decomposition (for iw=1 case)

        return niwae, kl, rec
        ################################################################################
        # End of code modification
        ################################################################################


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
        return ut.sample_gaussian(
            self.z_prior[0].expand(batch, self.z_dim),
            self.z_prior[1].expand(batch, self.z_dim))

    def sample_x(self, batch):
        z = self.sample_z(batch)
        return self.sample_x_given(z)

    def sample_x_given(self, z):
        return torch.bernoulli(self.compute_sigmoid_given(z))
