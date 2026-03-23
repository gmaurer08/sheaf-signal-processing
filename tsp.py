import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.neighbors import NearestNeighbors
from vdm import VDM
import cvxpy as cp
from sklearn.linear_model import OrthogonalMatchingPursuit
from wavelet import Wavelet
from builder import CochainSample # builder.py file provided by project supervisor
import pandas as pd
from tqdm import tqdm
import seaborn as sns
from collections import defaultdict
from utils import fibonacci_sphere, geodetic_to_ecef, project_to_tangent
import warnings
warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)


# Topological Signal Processing Class
class TSP(VDM):
    def __init__(self, data, eps, eps_pca, k, laplacian_code='Connection Normalized', gamma=0.95):
        super().__init__(data, eps, eps_pca, k, gamma)

        # dictionary that maps laplacian name codes to the VDM class methods that compute them
        self.create_laplacian = {
            'Connection': self.connection_laplacian,
            'Connection Normalized': self.connection_laplacian_norm,
            'Trivial': self.trivial_laplacian,
            'Trivial Normalized': self.trivial_laplacian_norm,
            'Sheaf': self.sheaf_laplacian
        }

        # verify that the laplacian_code value is valid
        if laplacian_code not in self.create_laplacian.keys():
            raise ValueError("laplacian must be in ['Connection', 'Connection Normalized', 'Trivial', 'Trivial Normalized', 'Sheaf']")

        self.laplacian_code = laplacian_code # string in ['Connection', 'Connection Normalized', 'Trivial', 'Trivial Normalized', 'Sheaf']

        self.laplacian = None # variable that will store the chosen laplacian
        self.wav = None # object that will store a wavelet object for sparse sheaf signal processing

    ################################################################
    # Plotting + Helper Functions
    ################################################################

    # Function that plots the points in R^3
    def plot_points(self,title='Point Cloud in R^3'):
        x = self.data
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(x[:,0], x[:,1], x[:,2], marker='o', s=10, c='darkblue')
        plt.title(title)
        plt.show()
    
    # Function that plots the graph
    def plot_graph(self,title="Graph of the Point Cloud in R^3"):
        self._ensure_graph()
        G = self.graph
        x = self.data
        # Plot the points
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(x[:,0], x[:,1], x[:,2], marker='o', s=2, c='navy')
        # Plot the edges
        for edge in G.edges():
            x1, y1, z1 = x[edge[0]]
            x2, y2, z2 = x[edge[1]]
            ax.plot([x1, x2], [y1, y2], [z1, z2], c='darkslateblue', linewidth=0.5, alpha=0.5)
        plt.title(title)
        plt.show()

    ################################################################
    # Laplacians and dictionaries
    ################################################################
    
    # Function that builds the laplacian specified by the self.laplacian_code variable 
    def build_laplacian(self):
        self.laplacian = self.create_laplacian[self.laplacian_code]()
        return self.laplacian
    
    # Function that ensures the laplacian is built
    def _ensure_laplacian(self):
        if self.laplacian is None:
            self.build_laplacian()

    # Function that creates a wavelet object
    def create_wav_object(self):
        self._ensure_laplacian()
        self.wav = Wavelet(self.laplacian)
        return self.wav
    
    # Function that ensures the wavelet object is created
    def _ensure_wav_object(self):
        if self.wav is None:
            self.create_wav_object()

    # Function that computes a wavelet dictionary with all shifts and scales in the list scales
    def create_dictionary(self, scales, normalize=False):
        self._ensure_wav_object()
        wav = self.wav
        return wav.make_dictionary(scales, normalize=normalize)

    ################################################################
    # Signal Compression
    ################################################################


    # Function that generates signals from linear combinations of the wavelet dictionary
    def generate_random_lc_signals(self, dictionary, num_signals=100, SEED=42, interval=[-10,10]):
        '''
        Function that generates signals from linear combinations of the wavelet dictionary
        Inputs:
        dictionary = wavelet dictionary
        num_signals = number of signals to generate
        SEED = random seed
        Returns:
        X = signals
        '''
        self._ensure_wav_object()
        X = np.zeros((dictionary.shape[0], num_signals))
        rng = np.random.default_rng(SEED)
        for i in range(num_signals):
            combination = dictionary @ (rng.random(size=dictionary.shape[1]) * (interval[1] - interval[0]) + interval[0])
            X[:, i] = combination
        return X


    # Function that generates that generates vector fields with kraichnan and samples from them
    def generate_kraichnan_signals(self, num_signals=500, M=100, Sigma=None, len_scale=10, SEED=42):
        '''
        Function that generates vector fields with kraichnan and samples from them
        Inputs:
        num_signals = number of signals to generate
        M = number of Monte Carlo samples in kraichnan
        n = number of waves in kraichnan
        SEED = random seed
        Returns:
        X = samples
        covariance = empirical covariance of X
        X_GT = ground truth
        '''

        # Monte Carlo samples
        M = M # number of samples
        N = self.data.shape[0] # number of nodes
        X = np.zeros((2*N, M)) # initialize matrix to store samples of the vector field
        rng = np.random.default_rng(SEED) # random number generator

        self._ensure_wav_object() # ensures that the wavelet object and therefore the laplacian are built
        self.wav._ensure_eig_laplacian() # ensures that laplacian eigendecomposition is built
        eigvals = self.wav.eigvals
        eigvecs = self.wav.eigvecs

        def kraichnan_r3(eigvals, eigvecs, alpha):
            '''
            Function that computes a vector field for R^3
            laplacian = string that specifies the laplacian to compute the vector field for,
                        possible values: 'Connection Normalized', 'Trivial', 'Trivial Normalized', 'Sheaf'
            alpha = vector of length N with coefficients sampled from a normal distribution
            This function assumes that laplacian eigendecompositions have been computed on the outside
            and are now stored in self.laplacian_eigs
            '''
            U = eigvecs @ np.multiply(self.kernel(eigvals), alpha) 
            return U

        self._ensure_orthonormal_bases()
        O = self.orthonormal_bases

        for m in range(M):
            # Compute normal random vectors k_i and random scalars z_i
            U = kraichnan_r3(eigvals, eigvecs, rng.normal(size=self.laplacian.shape[0])) # Field sample
            X[:,m] = U
            #for i in range(N):
                # X[2*i : 2*i+2, m] =  U[2*i:2*i+2]  #  store the field samples in X
                # before it was O[i].T @ U[2*i:2*i+2] (projection onto orthonormal basis)
                # I was using sine and cosine waves instead of the eigendecomposition
                # but now the projection raises errors
        
        # Function that computes the empirical covariance
        def empirical_covariance(X):
            X_mean = np.mean(X, axis=1, keepdims=True) # 2N x 1
            X_centered = X - X_mean # 2N x M
            cov = (X_centered @ X_centered.T) / (X.shape[1] - 1)
            return cov

        # Compute the empirical covariance
        cov = empirical_covariance(X)

        # Create a CochainSample object
        sample = CochainSample(X=X,
                            covariance=cov,
                            X_GT=X,
                            points=self.data,
                            local_bases=O,
                            V=N
                            )
        # Saplming signals
        sampled = sample.random_tangent_bundle_signals(Sigma=None,len_scale=len_scale, M=num_signals, seed=SEED)
        # Get the results
        X = sampled.X
        covariance = sampled.covariance
        X_GT = sampled.X_GT

        return X, covariance, X_GT

    # Function that computes sparse signal representations using OMP or CVXPY
    def sparsify_signals(self, X, dictionary, method = 'OMP'):
        '''
        Function that sparsifies a matrix of signals X 
        inputs:
        X = signal matrix
        dictionary = wavelet matrix
        method = string, 'OMP' or 'CVXPY'
        returns:
        sparse_signals = array with sparse signals
        '''
        self._ensure_wav_object()
        sparse_signals = np.zeros((dictionary.shape[1], X.shape[1])) # number of atoms x number of signals 
        for k in range(X.shape[1]):
            sparse_signals[:,k] = self.wav.sparse_signal(X[:,k], dictionary, method=method)
        return sparse_signals

    # Function that computes the percentage of nonzero coefficient in the sparse signal representations
    def compute_sparsity(self, sparse_signals):
        '''
        Function that calculates how many nonzero coefficients each sparse signal representation contains
        and divides it by the number of atoms in the dictionary
        input:
        sparse_signals = the list of sparse signal representations
        returns:
        sparsity = an array with sparsity coefficients for each signal
        '''
        # Compute the sparsity of the signal representations
        num_signals = sparse_signals.shape[1]
        sparsity = np.zeros(num_signals)

        for i in range(num_signals):
            sparsity[i] = np.sum(np.abs(sparse_signals[:,i])>0) / sparse_signals.shape[0]
        return sparsity
    
    # Function that computes the NMSE
    def NMSE(self,x, x_hat):
        return np.linalg.norm(x - x_hat)**2 / np.linalg.norm(x)**2
    
    # Function that computes the Normalized Mean Squared Error for each signal 
    def compute_NMSE(self, X_GT, sparse_signals, dictionary):

        self._ensure_wav_object()

        num_signals = sparse_signals.shape[1]
        nmse = np.zeros(num_signals)
        
        for k in range(num_signals):
            nmse[k] = self.NMSE(X_GT[:,k], dictionary @ sparse_signals[:,k])
        return nmse