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
    def __init__(self, data, eps, eps_pca, k, gamma=0.95):
        super().__init__(data, eps, eps_pca, k, gamma)
        self.L_conn = None
        self.L_sheaf = None
        self.L_trivial = None
        self.L_trivial_norm = None
        self.dictionaries = None
        self.wav_conn = None
        self.wav_trivial = None
        self.wav_trivial_norm = None
        self.wav_sheaf = None

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

    # Function that ensures the connection laplacian is computed
    def _ensure_connection_laplacian(self):
        if self.L_conn is None:
            self.L_conn = self.connection_laplacian()

    # Function that computes the trivial laplacian
    def trivial_laplacian(self):
        self._ensure_dim()
        W = self.get_weight_matrix() # Weight matrix
        d = self.get_degree_vector() # degree vector
        d_sqrt = np.sqrt(d)
        D = self.get_kron_degree_matrix()
        D_diag= np.diag(D)
        D_diag_sqrt = np.sqrt(D_diag)
        L_trivial = np.kron(np.diag(d) - W, np.eye(self.dim)) # Kronecker Graph Laplacian
        return L_trivial
    
    # Function that ensures the trivial laplacian is computed
    def _ensure_trivial_laplacian(self):
        if self.L_trivial is None:
            self.L_trivial = self.trivial_laplacian()
    
    # Function that computes the normalized trivial laplacian
    def trivial_laplacian_norm(self):
        self._ensure_trivial_laplacian()
        L_trivial = self.L_trivial
        D_diag_sqrt = np.sqrt(np.diag(self.get_kron_degree_matrix()))
        L_trivial_norm = np.diag(1./D_diag_sqrt) @ L_trivial @ np.diag(D_diag_sqrt) # Normalized Kronecker Graph Laplacian
        return L_trivial_norm
    
    # Function that ensures the normalized trivial laplacian is computed
    def _ensure_trivial_laplacian_norm(self):
        if self.L_trivial_norm is None:
            self.L_trivial_norm = self.trivial_laplacian_norm()

    # Function that computes the sheaf laplacian
    # Define edge orientation as follows: edge (i,j) has tail i and head j with i<j
    def coboundary_map(self):
        '''
        Function that builds the coboundary map of a sheaf given a graph G and O dictionary of alignment matrices Oij
        Assumption: all restriction maps (alignment matrices) have the same dimension dxd
        Inputs:
        G = graph
        O = dictionary of alignment matrices/ restriction maps (O[i][j] is the restriction map from node i to edge (i,j))
        d = dimension of the matrices
        Returns:
        delta = coboundary map
        '''
        self._ensure_graph()
        self._ensure_dim()
        self._ensure_alignment_matrices()
        G = self.graph
        O = self.alignment_matrices
        d = self.dim
        num_edges = G.number_of_edges()
        num_nodes = G.number_of_nodes()
        delta = np.zeros((num_edges*d,num_nodes*d))
        for edge_idx, edge in enumerate(G.edges()):
            if edge[0]<edge[1]:
                i = int(edge[0])
                j = int(edge[1])
            else:
                i = int(edge[1])
                j = int(edge[0])
            # Use the orthonormal bases as restriction maps
            delta[edge_idx*d:(edge_idx+1)*d, i*d:(i+1)*d] = O[i][j] * G.get_edge_data(i,j)['weight']  # multiply by edge weight
            delta[edge_idx*d:(edge_idx+1)*d, j*d:(j+1)*d] = - O[j][i] * G.get_edge_data(j,i)['weight']
        return delta
    
    # Function that computes the sheaf laplacian
    def sheaf_laplacian(self):
        delta = self.coboundary_map()
        L = delta.T @ delta
        self.L_sheaf = L
        return L
    
    def _ensure_sheaf_laplacian(self):
        if self.L_sheaf is None:
            self.L_sheaf = self.sheaf_laplacian()

    # Function that generates that generates vector fields with kraichnan and samples from them
    def generate_kraichnan_signals(self, num_signals=500, M=100, n=100, sigma=1, U_bar=0, Sigma=None, len_scale=10, SEED=6111983):
        '''
        Function that generates vector fields with kraichnan and samples from them
        Inputs:
        num_signals = number of signals to generate
        M = number of Monte Carlo samples in kraichnan
        n = number of waves in kraichnan
        sigma = strength (kraichnan)
        U_bar = mean flow
        Sigma = covariance of vector components in R^3 (sampler)
        len_scale = length scale
        SEED = random seed
        Returns:
        X = samples
        covariance = empirical covariance of X
        X_GT = ground truth
        '''

        # Monte Carlo samples
        M = M
        N = self.data.shape[0]
        X = np.zeros((2*N, M)) 
        rng = np.random.default_rng(SEED) # random number generator

        self._ensure_orthonormal_bases()
        O = self.orthonormal_bases

        def kraichnan_r3(self, k, Z, n=100, sigma=1, U_bar=0, SEED=42):
            '''
            Function that computes the a vector field for R^3
            k = precomputed normal random vectors (n x 3)
            Z = precomputed normal random numbers (n x 1)
            n = number of waves
            sigma = strength
            U_bar = mean flow
            SEED = random seed
            '''
            x = self.data
            # Define the field
            e1 = np.array([1, 0, 0])
            mult_factor = np.sqrt(sigma**2 / n)
            U = np.tile(U_bar*e1, (len(x),1)).astype(np.float64)
            for i in range(n):
                projector = e1 - (k[i]*k[i][0]) / np.dot(k[i], k[i])
                phase = x @ k[i] # N
                wave = Z[i] * np.cos(phase) + np.sin(phase) # N
                U -= mult_factor * wave[:,None] * projector[None,:]  # broadcast
            return U

        for m in range(M):
            # Compute normal random vectors k_i and random scalars z_i
            k = rng.normal(size=(n,3))
            Z = rng.normal(size=n)
            U = kraichnan_r3(self.data,k,Z,n=n, sigma=sigma, U_bar=U_bar) # Field sample
            for i in range(N):
                X[2*i : 2*i+2, m] = O[i].T @ U[i] # Projection onto the orthonormal basis
        
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
    
    # Function that ensures that the wavelet objects are created
    def _ensure_wav_objects(self):
        if self.wav_conn is None or self.wav_trivial is None or self.wav_trivial_norm is None or self.wav_sheaf is None:
            # Ensure the laplacians are built
            self._ensure_connection_laplacian()
            self._ensure_trivial_laplacian()
            self._ensure_trivial_laplacian_norm()
            self._ensure_sheaf_laplacian()

            # Create the wavelet objects
            wav_conn = Wavelet(-self.L_conn)
            wav_trivial = Wavelet(self.L_trivial)
            wav_trivial_norm = Wavelet(self.L_trivial_norm)
            wav_sheaf = Wavelet(self.L_sheaf)

            self.wav_conn = wav_conn
            self.wav_trivial = wav_trivial
            self.wav_trivial_norm = wav_trivial_norm
            self.wav_sheaf = wav_sheaf

    # Function that generates the dictionaries
    def make_dictionaries(self, scales=[2**(j-2) for j in range(7)]):
        
        self._ensure_wav_objects()
        wav_conn = self.wav_conn
        wav_trivial = self.wav_trivial
        wav_trivial_norm = self.wav_trivial_norm
        wav_sheaf = self.wav_sheaf
        
        # Build the dictionaries
        dict_conn = wav_conn.make_dictionary(scales)
        dict_trivial = wav_trivial.make_dictionary(scales)
        dict_trivial_norm = wav_trivial_norm.make_dictionary(scales)
        dict_sheaf = wav_sheaf.make_dictionary(scales)

        dictionaries = {
            'Connection': dict_conn,
            'Trivial': dict_trivial,
            'Trivial Normalized': dict_trivial_norm,
            'Sheaf': dict_sheaf
        }

        self.dictionaries = dictionaries

        return dictionaries
    
    def _ensure_dictionaries(self):
        if self.dictionaries is None:
            self.make_dictionaries


    def sparsify_signals(self, X):

        # Find the sparse signal representations
        sparse_signals_conn = []
        sparse_signals_trivial = []
        sparse_signals_trivial_norm = []
        sparse_signals_sheaf = []

        self._ensure_wav_objects()
        wav_conn = self.wav_conn
        wav_trivial = self.wav_trivial
        wav_trivial_norm = self.wav_trivial_norm
        wav_sheaf = self.wav_sheaf

        self._ensure_dictionaries()
        dict_conn = self.dictionaries['Connection']
        dict_trivial = self.dictionaries['Trivial']
        dict_trivial_norm = self.dictionaries['Trivial Normalized']
        dict_sheaf = self.dictionaries['Sheaf']

        for k in tqdm(range(X.shape[1])):
            sparse_signals_conn.append(wav_conn.sparse_signal(X[:,k], dict_conn))
            sparse_signals_trivial.append(wav_trivial.sparse_signal(X[:,k], dict_trivial))
            sparse_signals_trivial_norm.append(wav_trivial_norm.sparse_signal(X[:,k], dict_trivial_norm))
            sparse_signals_sheaf.append(wav_sheaf.sparse_signal(X[:,k], dict_sheaf))

        return {
            'Connection': sparse_signals_conn,
            'Trivial': sparse_signals_trivial,
            'Trivial Normalized': sparse_signals_trivial_norm,
            'Sheaf': sparse_signals_sheaf
        }

    def compute_sparsity(self, sparse_signals):

        self._ensure_dictionaries()
        dict_conn = self.dictionaries['Connection']
        dict_trivial = self.dictionaries['Trivial']
        dict_trivial_norm = self.dictionaries['Trivial Normalized']
        dict_sheaf = self.dictionaries['Sheaf']

        # Compute the sparsity of the signal representations
        sparsity_conn = []
        sparsity_trivial = []
        sparsity_trivial_norm = []
        sparsity_sheaf = []

        try:
            for k in range(len(sparse_signals['Connection'])):
                sparsity_conn.append(np.sum(np.abs(sparse_signals['Connection'][k])>0) / dict_conn.shape[1] * 100)
                sparsity_trivial.append(np.sum(np.abs(sparse_signals['Trivial'][k])>0) / dict_trivial.shape[1] * 100)
                sparsity_trivial_norm.append(np.sum(np.abs(sparse_signals['Trivial Normalized'][k])>0) / dict_trivial_norm.shape[1] * 100)
                sparsity_sheaf.append(np.sum(np.abs(sparse_signals['Sheaf'][k])>0) / dict_sheaf.shape[1] * 100)
        except:
            print("The input must be a dictionary 'sparse_signals' with keys 'Connection', 'Trivial', 'Trivial Normalized', 'Sheaf'.")
        
        return {
            'Connection': sparsity_conn,
            'Trivial': sparsity_trivial,
            'Trivial Normalized': sparsity_trivial_norm,
            'Sheaf': sparsity_sheaf
        }
    
    def NMSE(self,x, x_hat):
        return np.linalg.norm(x - x_hat)**2 / np.linalg.norm(x)**2
    
    # Function that computes the Normalized Mean Squared Error
    def compute_NMSE(self, X_GT, sparse_signals):

        self._ensure_dictionaries()
        dict_conn = self.dictionaries['Connection']
        dict_trivial = self.dictionaries['Trivial']
        dict_trivial_norm = self.dictionaries['Trivial Normalized']
        dict_sheaf = self.dictionaries['Sheaf']

        # Compute the NMSE of the sparse signal represntations
        nmse_conn_cube = []
        nmse_trivial_cube = []
        nmse_trivial_norm_cube = []
        nmse_sheaf_cube = []
        
        try: 
            for k in range(len(sparse_signals['Connection'])):
                nmse_conn_cube.append(self.NMSE(X_GT[:,k], dict_conn @sparse_signals['Connection'][k]))
                nmse_trivial_cube.append(self.NMSE(X_GT[:,k], dict_trivial @sparse_signals['Trivial'][k]))
                nmse_trivial_norm_cube.append(self.NMSE(X_GT[:,k], dict_trivial_norm @sparse_signals['Trivial Normalized'][k]))
                nmse_sheaf_cube.append(self.NMSE(X_GT[:,k], dict_sheaf @sparse_signals['Sheaf'][k]))
            
            return {
                'Connection': nmse_conn_cube,
                'Trivial': nmse_trivial_cube,
                'Trivial Normalized': nmse_trivial_norm_cube,
                'Sheaf': nmse_sheaf_cube
            }
        except:
            print("The 1st input must be a matrix 'X_GT' with ground truth signals.")
            print("The 2nd input must be a dictionary 'sparse_signals' with keys 'Connection', 'Trivial', 'Trivial Normalized', 'Sheaf'.")


    # Function that plots all the NMSE densities of all the laplacians
    def plot_nmse(self,nmse):
        sns.kdeplot(nmse['Connection'], label='Connection Laplacian')
        sns.kdeplot(nmse['Trivial'], label='Trivial Laplacian')
        sns.kdeplot(nmse['Trivial Normalized'], label='Normalized Trivial Laplacian')
        sns.kdeplot(nmse['Sheaf'], label='Sheaf Laplacian')
        plt.xlabel('NMSE')
        plt.ylabel('Density')
        plt.legend()
        plt.show()