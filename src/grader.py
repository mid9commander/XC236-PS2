#!/usr/bin/env python3
import inspect
import unittest
import argparse
from graderUtil import graded, CourseTestRunner, GradedTestCase
from subprocess import Popen
from subprocess import DEVNULL, STDOUT, check_call
import torch
import numpy as np
import pickle

# import student submission
import submission

#########
# TESTS #
#########

ref_sample_gaussian = torch.Tensor([[ 3.1794934273, -0.5144982934, -4.8044347763,  0.9323326349,
         -1.5944868326],
        [-3.0817422867,  1.2388616800,  2.1787362099, -1.7788546085,
         -0.8472986221],
        [-1.7190116644,  0.2775234580, -1.4334810972,  2.1897780895,
         -1.8190975189]])

ref_log_normal = torch.Tensor([-6.9819622040, -7.6092824936, -4.8691501617])

ref_log_normal_mixture = torch.Tensor([-4.8879451752, -7.7989110947, -8.3332138062])

ref_vae_niwae = torch.Tensor([544.3607177734])

device = "cpu"

def max_error(a, b):
    a = a.to(device)
    error = np.max(np.abs(np.array(a - b)))
    return error

def is_different(a, b, tol):
    a = a.to(device)
    error = np.max(np.abs(np.array(a - b)))
    return error > tol

class Test_1a(GradedTestCase):
    def setUp(self):
        self.tol = 1e-5

    @graded()
    def test_0(self):
        """1a-0-basic: sample_gaussian"""
        torch.manual_seed(0)
        m = torch.randn(3, 5)
        v = torch.exp(torch.randn_like(m))
        torch.manual_seed(0)
        z_s = submission.utils.sample_gaussian(m, v)
        self.assertTrue(not is_different(z_s, ref_sample_gaussian, self.tol),
                        f"Max absolute error {max_error(z_s, ref_sample_gaussian)} > {self.tol}")


class Test_1b(GradedTestCase):
    def setUp(self):
        self.tol = 1e-5
        self.sol_VAE = self.run_with_solution_if_possible(submission, lambda sub_or_sol:sub_or_sol.models.vae.VAE)
    
    @torch.no_grad()
    @graded()
    def test_0(self):
        """1b-0-basic: check VAE negative_elbo_bound output shapes"""
        torch.manual_seed(0)
        x = torch.randint(0, 2, (97, 784)).float()

        vae = submission.models.vae.VAE(z_dim=10)
        nelbo, kl, rec = vae.negative_elbo_bound(x)
        self.assertTrue(nelbo.shape == torch.Size([]), "NELBO unexpected tensor dimensions")
        self.assertTrue(kl.shape == torch.Size([]), "KL unexpected tensor dimensions")
        self.assertTrue(rec.shape == torch.Size([]), "Rec unexpected tensor dimensions")

    ### BEGIN_HIDE ###
        ### END_HIDE ###

class Test_1c(GradedTestCase):
    def setUp(self):
        self.nelbo = 100
        self.nelbo_tol = 2
        self.kl = 19.5
        self.kl_tol = 2
        self.rec = 82
        self.rec_tol = 2
    
    @graded()
    def test_0(self):
        """1c-0-basic: check VAE reported nelbo, kl, and rec values"""
        with open('./submission/VAE.pkl', 'rb') as f:
            metrics = pickle.load(f)
        nelbo, kl, rec = metrics.values()
        self.assertTrue(not is_different(nelbo, self.nelbo, self.nelbo_tol), f"Max absolute error {max_error(nelbo, self.nelbo)} > {self.nelbo_tol} for NELBO")
        self.assertTrue(not is_different(kl, self.kl, self.kl_tol), f"Max absolute error {max_error(kl, self.kl)} > {self.kl_tol} for KL")
        self.assertTrue(not is_different(rec, self.rec, self.rec_tol), f"Max absolute error {max_error(rec, self.rec)} > {self.rec_tol} for Rec")

class Test_2a(GradedTestCase):
    def setUp(self):
        self.tol = 1e-5
        self.sol_sample_gaussian = self.run_with_solution_if_possible(submission, lambda sub_or_sol:sub_or_sol.utils.sample_gaussian)
        self.sol_GMVAE = self.run_with_solution_if_possible(submission, lambda sub_or_sol:sub_or_sol.models.gmvae.GMVAE)
    
    @graded()
    def test_0(self):
        """2a-0-basic: log_normal"""
        torch.manual_seed(0)
        m = torch.randn(3, 5)
        v = torch.exp(torch.randn_like(m))
        z = self.sol_sample_gaussian(m, v)

        v_s = submission.utils.log_normal(z, m, v)

        self.assertTrue(not torch.isnan(v_s).any(), "NaN values detected in the log_normal")
        self.assertTrue(not is_different(v_s, ref_log_normal, self.tol), f"Max absolute error {max_error(v_s, ref_log_normal)} > {self.tol}")
    
    @graded()
    def test_1(self):
        """2a-1-basic: log_normal_mixture"""
        torch.manual_seed(0)
        m = torch.randn(3, 4, 5)
        v = torch.exp(torch.randn_like(m))
        z = self.sol_sample_gaussian(m[:, 0], v[:, 0])

        v_s = submission.utils.log_normal_mixture(z, m, v)

        self.assertTrue(not torch.isnan(v_s).any(), "NaN values detected in the log_normal_mixture")
        self.assertTrue(not is_different(v_s, ref_log_normal_mixture, self.tol), f"Max absolute error {max_error(v_s, ref_log_normal_mixture)} > {self.tol}")

    ### BEGIN_HIDE ###
    ### END_HIDE ###

class Test_2b(GradedTestCase):
    def setUp(self):
        self.nelbo = 97
        self.nelbo_tol = 3
        self.kl = 17.5
        self.kl_tol = 1
        self.rec = 80
        self.rec_tol = 2
    
    @graded()
    def test_0(self):
        """2b-0-basic: check GMVAE reported nelbo, kl, and rec values"""
        with open('./submission/GMVAE.pkl', 'rb') as f:
            metrics = pickle.load(f)
        nelbo, kl, rec = metrics.values()
        self.assertTrue(not is_different(nelbo, self.nelbo, self.nelbo_tol), f"Max absolute error {max_error(nelbo, self.nelbo)} > {self.nelbo_tol} for NELBO")
        self.assertTrue(not is_different(kl, self.kl, self.kl_tol), f"Max absolute error {max_error(kl, self.kl)} > {self.kl_tol} for KL")
        self.assertTrue(not is_different(rec, self.rec, self.rec_tol), f"Max absolute error {max_error(rec, self.rec)} > {self.rec_tol} for Rec")


class Test_3b(GradedTestCase):
    def setUp(self):
        self.tol = 0.15
        
    @torch.no_grad()
    @graded()
    def test_0(self):
        """3b-0-basic: negative_iwae_bound for VAE"""
        torch.manual_seed(0)
        x = torch.randint(0, 2, (97, 784)).float()

        vae = submission.models.vae.VAE(z_dim=10)

        torch.manual_seed(0)
        niwae_pred, _, _ = vae.negative_iwae_bound(x, iw=10)

        self.assertTrue(
            not is_different(niwae_pred, ref_vae_niwae, self.tol),
            f"Max absolute error in NIWAE-10 {max_error(niwae_pred, ref_vae_niwae)} > {self.tol}"
        )

class Test_3c(GradedTestCase):
    def setUp(self):
        self.model = "VAE"
        self.niwae_1 = 99
        self.niwae_1_tol = 2.5
        self.niwae_10 = 97
        self.niwae_10_tol = 2.5
        self.niwae_100 = 96
        self.niwae_100_tol = 2.5
        self.niwae_1000 = 95
        self.niwae_1000_tol = 2.5

    @graded()
    def test_0(self):
        """3c-0-basic: NIWAE-1 for VAE"""
        with open(f'./submission/{self.model}_iwae_1.pkl', 'rb') as f:
            metrics = pickle.load(f)
        niwae = metrics['NIWAE']
        self.assertTrue(not is_different(niwae, self.niwae_1, self.niwae_1_tol), f"Max absolute error {max_error(niwae, self.niwae_1)} > {self.niwae_1_tol} for NIWAE-1")
    
    @graded()
    def test_1(self):
        """3c-1-basic: NIWAE-10 for VAE"""
        with open(f'./submission/{self.model}_iwae_10.pkl', 'rb') as f:
            metrics = pickle.load(f)
        niwae = metrics['NIWAE']
        self.assertTrue(not is_different(niwae, self.niwae_10, self.niwae_10_tol), f"Max absolute error {max_error(niwae, self.niwae_10)} > {self.niwae_10_tol} for NIWAE-10")
    
    @graded()
    def test_2(self):
        """3c-2-basic: NIWAE-100 for VAE"""
        with open(f'./submission/{self.model}_iwae_100.pkl', 'rb') as f:
            metrics = pickle.load(f)
        niwae = metrics['NIWAE']
        self.assertTrue(not is_different(niwae, self.niwae_100, self.niwae_100_tol), f"Max absolute error {max_error(niwae, self.niwae_100)} > {self.niwae_100_tol} for NIWAE-100")
    
    @graded()
    def test_3(self):
        """3c-3-basic: NIWAE-1000 for VAE"""
        with open(f'./submission/{self.model}_iwae_1000.pkl', 'rb') as f:
            metrics = pickle.load(f)
        niwae = metrics['NIWAE']
        self.assertTrue(not is_different(niwae, self.niwae_1000, self.niwae_1000_tol), f"Max absolute error {max_error(niwae, self.niwae_1000)} > {self.niwae_1000_tol} for NIWAE-1000")

class Test_3d(GradedTestCase):
    def setUp(self):
        self.model = 'GMVAE'
        self.niwae_1 = 97
        self.niwae_1_tol = 2.5
        self.niwae_10 = 94
        self.niwae_10_tol = 2.6
        self.niwae_100 = 93
        self.niwae_100_tol = 2.5
        self.niwae_1000 = 93
        self.niwae_1000_tol = 2.5

        self.tol = 0.15
        self.sol_GMVAE = self.run_with_solution_if_possible(submission, lambda sub_or_sol:sub_or_sol.models.gmvae.GMVAE)
    
    @graded()
    def test_0(self):
        """3d-0-basic: NIWAE-1 for GMVAE"""
        with open(f'./submission/{self.model}_iwae_1.pkl', 'rb') as f:
            metrics = pickle.load(f)
        niwae = metrics['NIWAE']
        self.assertTrue(not is_different(niwae, self.niwae_1, self.niwae_1_tol), f"Max absolute error {max_error(niwae, self.niwae_1)} > {self.niwae_1_tol} for NIWAE-1")
    
    @graded()
    def test_1(self):
        """3d-1-basic: NIWAE-10 for GMVAE"""
        with open(f'./submission/{self.model}_iwae_10.pkl', 'rb') as f:
            metrics = pickle.load(f)
        niwae = metrics['NIWAE']
        self.assertTrue(not is_different(niwae, self.niwae_10, self.niwae_10_tol), f"Max absolute error {max_error(niwae, self.niwae_10)} > {self.niwae_10_tol} for NIWAE-10")
    
    @graded()
    def test_2(self):
        """3d-2-basic: NIWAE-100 for GMVAE"""
        with open(f'./submission/{self.model}_iwae_100.pkl', 'rb') as f:
            metrics = pickle.load(f)
        niwae = metrics['NIWAE']
        self.assertTrue(not is_different(niwae, self.niwae_100, self.niwae_100_tol), f"Max absolute error {max_error(niwae, self.niwae_100)} > {self.niwae_100_tol} for NIWAE-100")
    
    @graded()
    def test_3(self):
        """3d-3-basic: NIWAE-1000 for GMVAE"""
        with open(f'./submission/{self.model}_iwae_1000.pkl', 'rb') as f:
            metrics = pickle.load(f)
        niwae = metrics['NIWAE']
        self.assertTrue(not is_different(niwae, self.niwae_1000, self.niwae_1000_tol), f"Max absolute error {max_error(niwae, self.niwae_1000)} > {self.niwae_1000_tol} for NIWAE-1000")

    ### BEGIN_HIDE ###
    ### END_HIDE ###

class Test_4a(GradedTestCase):
    def setUp(self):
        self.accuracy = 0.735
        self.accuracy_tol = 0.025
    
    @graded()
    def test_0(self):
        """4a-0-basic: check classifier accuracy for SSVAE with gw=0"""
        with open('./submission/SSVAE_0.pkl', 'rb') as f:
            metrics = pickle.load(f)
        accuracy =metrics['accuracy']
        self.assertTrue(not is_different(accuracy, self.accuracy, self.accuracy_tol), f"Max absolute error {max_error(accuracy, self.accuracy)} > {self.accuracy_tol} for accuracy with gw=0")

class Test_4b(GradedTestCase):
    def setUp(self):
        self.tol = 0.003
        self.sol_SSVAE = self.run_with_solution_if_possible(submission, lambda sub_or_sol:sub_or_sol.models.ssvae.SSVAE)
    
    @torch.no_grad()
    @graded()
    def test_0(self):
        """4b-0-basic: SSVAE negative_elbo_bound output shapes"""
        torch.manual_seed(0)
        x = torch.randint(0, 2, (97, 784)).float()

        ssvae = submission.models.ssvae.SSVAE(gen_weight=1, class_weight=100)
        torch.manual_seed(0)
        nelbo, klz, kly, rec = ssvae.negative_elbo_bound(x)
        self.assertTrue(nelbo.shape == torch.Size([]), "NELBO unexpected tensor dimensions")
        self.assertTrue(klz.shape == torch.Size([]), "KLz unexpected tensor dimensions")
        self.assertTrue(kly.shape == torch.Size([]), "KLy unexpected tensor dimensions")
        self.assertTrue(rec.shape == torch.Size([]), "Rec unexpected tensor dimensions")

    ### BEGIN_HIDE ###
    ### END_HIDE ###

class Test_4c(GradedTestCase):
    def setUp(self):
        self.accuracy = 0.935
        self.accuracy_tol = 0.025
    
    @graded()
    def test_0(self):
        """4c-0-basic: check classifier accuracy for SSVAE with gw=1"""
        with open('./submission/SSVAE_1.pkl', 'rb') as f:
            metrics = pickle.load(f)
        accuracy = metrics['accuracy']
        self.assertTrue(not is_different(accuracy, self.accuracy, self.accuracy_tol), f"Max absolute error {max_error(accuracy, self.accuracy)} > {self.accuracy_tol} for accuracy with gw=1")

class Test_5b(GradedTestCase):
    def setUp(self):
        self.tol = 1e-3
        self.sol_FSVAE = self.run_with_solution_if_possible(submission, lambda sub_or_sol:sub_or_sol.models.fsvae.FSVAE)

    ### BEGIN_HIDE ###
    ### END_HIDE ###

        
def getTestCaseForTestID(test_id):
    question, part, _ = test_id.split("-")
    g = globals().copy()
    for name, obj in g.items():
        if inspect.isclass(obj) and name == ("Test_" + question):
            return obj("test_" + part)

if __name__ == "__main__":
    # Parse for a specific test
    parser = argparse.ArgumentParser()
    parser.add_argument("test_case", nargs="?", default="all")
    test_id = parser.parse_args().test_case

    assignment = unittest.TestSuite()
    if test_id != "all":
        assignment.addTest(getTestCaseForTestID(test_id))
    else:
        assignment.addTests(
            unittest.defaultTestLoader.discover(".", pattern="grader.py")
        )
    CourseTestRunner().run(assignment)