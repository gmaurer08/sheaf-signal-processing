import numpy as np
import networkx as nx
from sklearn.neighbors import NearestNeighbors
from vdm import VDM
from tqdm import tqdm
import cvxpy as cp
from sklearn.linear_model import OrthogonalMatchingPursuit

SEED = 6111983
np.random.seed(SEED)

class Wavelet:
    def __init__(self,L):
        # Laplacian and its eigendecomposition
        self.L = L
        self.eigvals = None
        self.eigvecs = None

        # kernel parameters (default values)
        self.k = 1
        self.t = 1
        self.p = 1
    
    # Kernel parameter modification
    def set_kernel_parameters(self,k,t,p):
        if k < 0:
            raise ValueError("k must be non-negative")
        if t < 0:
            raise ValueError("t must be non-negative")
        if p < 0:
            raise ValueError("p must be non-negative")
        self.k = k
        self.t = t
        self.p = p
    
    # Parametric kernel function g
    def g(self,x):
        '''
        Function that computes the parametric kernel g
        The kernel is of the form g(x) = x^k * exp(-t * x^p)
        satisfying the conditions g(0)=0 and g(x)->0 as x->+inf to serve as a bandpass filter
        '''
        return x**self.k * np.exp(-self.t * x**self.p)

    def get_eig_laplacian(self):
        '''
        Function that computes the eigendecomposition of the laplacian 
        and returns a tuple with sorted eigenvalues and eigenvectors
        L = laplacian
        '''
        eigvals, eigvecs = np.linalg.eig(self.L)
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:,idx]
        self.eigvals = eigvals
        self.eigvecs = eigvecs
        return eigvals, eigvecs
    
    # Function that ensures the eigendecomposition is computed
    def _ensure_eig_laplacian(self):
        if self.eigvals is None or self.eigvecs is None:
            self.get_eig_laplacian(self.L)

    # Function that applies the wavelet operator to a node or edge
    def wavelet(self,m,scale,shift):
        '''
        Function that applies a wavelet operator with input scale and shift parameters to a node or edge m
        Inputs:
        m = node or edge
        scale = scale parameter
        shift = shift parameter
        '''
        self._ensure_eig_laplacian()
        eigvals = self.eigvals
        eigvecs = self.eigvecs
        weights = np.multiply(self.g(scale*eigvals), eigvecs[shift])
        return np.dot(eigvecs[m], weights)
    
    # Function that returns the wavelet coefficient of a signal
    def wavelet_coef(self,f,scale,shift):
        '''
        Function that computes the wavelet coefficient of a signal f
        f = signal
        scale = scale parameter
        shift = shift parameter
        '''
        self._ensure_eig_laplacian()
        eigvals = self.eigvals
        eigvecs = self.eigvecs
        fourier_transform = eigvecs.T @ f
        weights = np.multiply(self.g(scale*eigvals), eigvecs[shift])
        return np.dot(fourier_transform, weights)
    
    # Function that builds an array of atoms for a given set of scales and shifts (wavelet dictionary)
    # The resulting dictionary will have size L.shape[0] x num_scales*num_shifts
    def make_dictionary(self, scales, shifts, normalize=False):
        '''
        Function that creates a dictionary of wavelets for different scales and shifts
        Inputs:
        scales = list of scales
        shifts = list of shifts
        normalize = boolean
        Returns:
        wavelet_dict = dictionary of wavelets
        '''
        self._ensure_eig_laplacian()
        num_scales = len(scales)
        num_shifts = len(shifts)
        # Initialize the wavelet dictionary
        wavelet_dict = np.zeros((self.L.shape[0],num_scales*num_shifts))
        for i, scale in enumerate(scales):
            # Compute the spectral scaling coefficients, applying the kernel function
            spectral_scaling = self.g(self.eigvals * scale)
            wavelet_dict[:,i*num_shifts:(i+1)*num_shifts] = self.eigvecs @ np.diag(spectral_scaling) @ self.eigvecs.T
        # Optional normalization
        if normalize:
            col_norms = np.linalg.norm(wavelet_dict, axis=0)
            wavelet_dict = wavelet_dict/col_norms
        return wavelet_dict
    
    # Function that builds an dictionary for a given list of (scale, shift) tuples
    def make_dictionary_from_tuples(self, scale_shift_tuples, normalize=False):
        '''
        Function that creates a dictionary of wavelets for different scales and shifts
        Inputs:
        scale_shift_tuples = list of (scale, shift) tuples
        normalize = boolean
        Returns:
        wavelet_dict = dictionary of wavelets
        '''
        self._ensure_eig_laplacian()
        # Initialize the wavelet dictionary
        wavelet_dict = np.zeros((self.L.shape[0],len(scale_shift_tuples)))
        # Iterate over the (scale, shift) tuples and compute the wavelets
        for idx, scale_shift_tuple in enumerate(scale_shift_tuples):
            scale, shift = scale_shift_tuple
            spectral_scaling = self.g(self.eigvals * scale)
            weights = np.multiply(spectral_scaling, self.eigvecs[shift])
            wavelet_dict[:,idx] = self.eigvecs @ weights
        # Optional normalization
        if normalize:
            col_norms = np.linalg.norm(wavelet_dict, axis=0)
            wavelet_dict = wavelet_dict/col_norms
        return wavelet_dict
    
    def sparse_signal(self,f,wav_dict,method='OMP'):
        '''
        Function that computes the sparsest representation of a signal f
        f = signal
        wav_dict = dictionary of wavelets, optionally already implemented
        method = 'OMP' (orthogonal matching pursuit) or 'CVXPY', with default 'OMP'
        Returns:
        sparse representation of f 
        '''
        # Solve LASSO, basis pursuit problem with convex optimization using CVXPY
        # (convex optimization, takes longer but is more precise)
        if method=='CVXPY':
            num_atoms = wav_dict.shape[1]
            x = cp.Variable(num_atoms)
            problem = cp.Problem(cp.Minimize(cp.norm1(x)),[wav_dict@ x == f])
            problem.solve(solver=cp.SCS, verbose=False)
            sparse_signal = x.value
            return sparse_signal
        # Solve LASSO, basis pursuit problem with convex optimization using OMP (Orthogonal Matching Pursuit)
        if method=='OMP':
            omp = OrthogonalMatchingPursuit()
            omp.fit(wav_dict,f)
            sparse_signal = omp.coef_
            return sparse_signal
        else:
            raise ValueError("Method must be 'OMP' or 'CVXPY'")