import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.neighbors import NearestNeighbors
from python.vdm import VDM
import cvxpy as cp
from sklearn.linear_model import OrthogonalMatchingPursuit
from python.wavelet import Wavelet
from python.builder import CochainSample # builder.py file provided by project supervisor
import pandas as pd
from tqdm import tqdm
import seaborn as sns
from collections import defaultdict
from python.utils import *
import warnings
warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)


# Topological Signal Processing Class
class TSP(VDM):
    def __init__(self, data, eps, eps_pca, k, laplacian_code='Connection Normalized', gamma=0.95, h=1, t=1, p=1):
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

        # Wavelet parameters
        self.h = h
        self.t = t
        self.p = p

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
        self.wav = Wavelet(self.laplacian, self.h, self.t, self.p)
        return self.wav
    
    # Function that ensures the wavelet object is created
    def _ensure_wav_object(self):
        if self.wav is None:
            self.create_wav_object()

    # Function that adjusts the wavelet parameters so that the kernel is adapted to the eigenvalues
    def _adjust_kernel_parameters(self):
        self._ensure_wav_object()
        self.wav._ensure_eig_laplacian()
        eigvals = self.wav.eigvals
        l_max = np.max(np.abs(eigvals))
        self.h = 1.0
        self.p = 1.0
        self.t = 3.0 / (l_max ** self.p + 1e-12)
        self.wav.set_kernel_parameters(self.h, self.t, self.p)

    # Function that computes a wavelet dictionary with all shifts and scales in the list scales
    def create_dictionary(self, scales, normalize=False, adjust_kernel=False):
        self._ensure_wav_object()
        wav = self.wav
        if adjust_kernel:
            self._adjust_kernel_parameters()
        return wav.make_dictionary(scales, normalize=normalize)

    ################################################################
    # Signal Compression + Denoising Functions
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
    def compute_NMSE(self, signals, sparse_signals, dictionary):

        self._ensure_wav_object()

        num_signals = sparse_signals.shape[1]
        nmse = np.zeros(num_signals)
        
        for k in range(num_signals):
            nmse[k] = self.NMSE(signals[:,k], dictionary @ sparse_signals[:,k])
        return nmse


    # Function that computes the reconstruction SNR
    def snr_rec(self, signal, reconstruction):
        return np.sum(signal**2) / np.sum((signal - reconstruction)**2)

    # Function that reconstructs a signal
    def reconstruct_signals(self, sparse_signals, wav_dict):
        self._ensure_wav_object()
        reconstructed_signals = np.zeros((wav_dict.shape[0], sparse_signals.shape[1]))
        for i in range(sparse_signals.shape[1]):
            #print("Shape wav_dict:", wav_dict.shape)
            #print("Shape sparse_signals:", sparse_signals.shape)
            reconstructed_signals[:,i] = wav_dict @ sparse_signals[:,i]
        return reconstructed_signals
    
    # Function that computes the reconstruction SNR for all signals
    def compute_snr_rec(self, gt_signals, reconstructed_signals):
        num_signals = reconstructed_signals.shape[1]
        snr_rec = np.zeros(num_signals)
        for k in range(num_signals):
            snr_rec[k] = self.snr_rec(gt_signals[:,k], reconstructed_signals[:,k])
        return snr_rec
    
    # Function that computes the gain of the reconstruction SNR
    def snr_gain(self, snr_rec, snr):
        return snr_rec / snr
    
    # Function that computes the gain of the reconstruction SNR for all signals
    def compute_snr_gain(self, snr_rec, snr):
        num_signals = snr_rec.shape[0]
        snr_gain = np.zeros(num_signals)
        for k in range(num_signals):
            snr_gain[k] = self.snr_gain(snr_rec[k], snr)
        return snr_gain
    

def signal_compression_exp1(point_cloud, hyperparameters):
    ''' 
    Function that takes in input a point cloud and a dictionary of hyperparameters and performs signal compression experiment 1 on the data
    Returns:
    - sparsity_results
    - nmse_results
    '''
    # Initialize result dictionaries
    sparsity_results = defaultdict(dict)
    nmse_results = defaultdict(dict)
    # Hyperparameters
    num_scales = hyperparameters['num_scales']
    laplacians = hyperparameters['laplacians']
    num_signals = hyperparameters['num_signals']
    eps = hyperparameters['eps']
    eps_pca = hyperparameters['eps_pca']
    k = hyperparameters['k']
    gamma = hyperparameters['gamma']
    adjust_kernel = hyperparameters['adjust_kernel']

    for laplacian in laplacians:

        # Create TSP (topological signal processing) object
        Tsp = TSP(point_cloud, eps=eps, eps_pca=eps_pca, k=k, laplacian_code=laplacian, gamma=gamma)

        signals = None

        for num_scal in num_scales[::-1]:
            # Create dictionary
            dictionary = Tsp.create_dictionary(scales=[2**(j-num_scal//2) for j in range(num_scal)], adjust_kernel=adjust_kernel)

            if num_scal == num_scales[-1]:
                # Generate signals for the corresponding laplacian and dictionary
                signals = Tsp.generate_random_lc_signals(dictionary, num_signals=num_signals)

            # Sparsify signals
            sparse_signals = Tsp.sparsify_signals(signals, dictionary)

            # Compute sparsity
            sparsity =  Tsp.compute_sparsity(sparse_signals)
            
            # Compute NMSE
            nmse = Tsp.compute_NMSE(signals, sparse_signals, dictionary)

            # Add sparsity and NMSE to the dictionaries
            sparsity_results[laplacian][num_scal] = sparsity
            nmse_results[laplacian][num_scal] = nmse
        
    return sparsity_results, nmse_results



def signal_compression_exp2(point_cloud, hyperparameters):
    '''
    Function that takes in input a point cloud and a dictionary of hyperparameters and performs signal compression experiment 2 on the data
    Returns:
    - sparsity_results
    - nmse_results
    '''
    # Initialize result dictionaries
    sparsity_results = defaultdict(dict)
    nmse_results = defaultdict(dict)

    # Hyperparameters
    num_scales = hyperparameters['num_scales']
    laplacians = hyperparameters['laplacians']
    num_signals = hyperparameters['num_signals']
    eps = hyperparameters['eps']
    eps_pca = hyperparameters['eps_pca']
    k = hyperparameters['k']
    gamma = hyperparameters['gamma']
    SEED = hyperparameters['SEED']
    adjust_kernel = hyperparameters['adjust_kernel']
    sigma = 3

    # Signals
    np.random.seed(SEED)
    signals = np.random.normal(scale=sigma, size=(2*point_cloud.shape[0],num_signals)) # signal dimension = manifold_dim * num_points

    for laplacian in laplacians:

        # Create TSP (topological signal processing) object
        Tsp = TSP(point_cloud, eps=eps, eps_pca=eps_pca, k=k, laplacian_code=laplacian, gamma=gamma)

        for num_scal in num_scales[::-1]:
            # Create dictionary
            dictionary = Tsp.create_dictionary(scales=[2**(j-num_scal//2) for j in range(num_scal)], adjust_kernel=adjust_kernel)

            # Sparsify signals
            sparse_signals = Tsp.sparsify_signals(signals, dictionary)

            # Compute sparsity
            sparsity =  Tsp.compute_sparsity(sparse_signals)
            
            # Compute NMSE
            nmse = Tsp.compute_NMSE(signals, sparse_signals, dictionary)

            # Add sparsity and NMSE to the dictionaries
            sparsity_results[laplacian][num_scal] = sparsity
            nmse_results[laplacian][num_scal] = nmse

    return sparsity_results, nmse_results


def signal_compression_exp3(point_cloud, hyperparameters):
    '''
    Function that takes in input a point cloud and a dictionary of hyperparameters and performs signal compression experiment 3 on the data
    Returns:
    - sparsity_results
    - nmse_results
    '''
    # Initialize result dictionaries
    sparsity_results = defaultdict(dict)
    nmse_results = defaultdict(dict)
    # Hyperparameters
    num_scales = hyperparameters['num_scales']
    laplacians = hyperparameters['laplacians']
    num_signals = hyperparameters['num_signals']
    eps = hyperparameters['eps']
    eps_pca = hyperparameters['eps_pca']
    k = hyperparameters['k']
    gamma = hyperparameters['gamma']
    SEED = hyperparameters['SEED']
    adjust_kernel = hyperparameters['adjust_kernel']

    for laplacian in laplacians:

        # Create TSP (topological signal processing) object
        Tsp = TSP(point_cloud, eps=eps, eps_pca=eps_pca, k=k, laplacian_code=laplacian, gamma=gamma)

        signals = None

        for num_scal in num_scales[::-1]:
            # Create dictionary
            dictionary = Tsp.create_dictionary(scales=[2**(j-num_scal//2) for j in range(num_scal)], adjust_kernel=adjust_kernel)

            if num_scal == num_scales[-1]:
                # Generate signals for the corresponding laplacian and dictionary
                signals, cov, signals_GT = Tsp.generate_kraichnan_signals(num_signals=num_signals, SEED=SEED)

            # Sparsify signals
            sparse_signals = Tsp.sparsify_signals(signals, dictionary)

            # Compute sparsity
            sparsity =  Tsp.compute_sparsity(sparse_signals)
            
            # Compute NMSE
            nmse = Tsp.compute_NMSE(signals, sparse_signals, dictionary)

            # Add sparsity and NMSE to the dictionaries
            sparsity_results[laplacian][num_scal] = sparsity
            nmse_results[laplacian][num_scal] = nmse

    return sparsity_results, nmse_results



def signal_compression(point_cloud, hyperparameters, signals):
    '''
    Function that takes in input a point cloud and a dictionary of hyperparameters and performs signal compression experiment 3 on the data
    Returns:
    - sparsity_results
    - nmse_results
    '''
    # Initialize result dictionaries
    sparsity_results = defaultdict(dict)
    nmse_results = defaultdict(dict)
    # Hyperparameters
    num_scales = hyperparameters['num_scales']
    laplacians = hyperparameters['laplacians']
    eps = hyperparameters['eps']
    eps_pca = hyperparameters['eps_pca']
    k = hyperparameters['k']
    gamma = hyperparameters['gamma']
    SEED = hyperparameters['SEED']
    h = hyperparameters['h']
    t = hyperparameters['t']
    p = hyperparameters['p']
    normalize = hyperparameters['normalize']
    adjust_kernel = hyperparameters['adjust_kernel']

    for laplacian in laplacians:

        # Create TSP (topological signal processing) object
        Tsp = TSP(point_cloud, eps=eps, eps_pca=eps_pca, k=k, laplacian_code=laplacian, gamma=gamma, h=h, t=t, p=p)

        for num_scal in num_scales:
            # Create dictionary
            dictionary = Tsp.create_dictionary(scales=[2**(j-num_scal//2) for j in range(num_scal)], normalize=normalize, adjust_kernel=adjust_kernel)
            
            #print(f"Inside Signal Compression Function: Number of NaN values in dictionary: {np.isnan(dictionary).sum()}")

            # Sparsify signals
            sparse_signals = Tsp.sparsify_signals(signals, dictionary)

            # Compute sparsity
            sparsity =  Tsp.compute_sparsity(sparse_signals)
            
            # Compute NMSE
            nmse = Tsp.compute_NMSE(signals, sparse_signals, dictionary)

            # Add sparsity and NMSE to the dictionaries
            sparsity_results[laplacian][num_scal] = sparsity
            nmse_results[laplacian][num_scal] = nmse

    return sparsity_results, nmse_results

# Function that adds noise to a signal based on the SNR
def add_noise(signal, SNR):
    P_signal = np.mean(signal**2)
    P_noise = P_signal / SNR
    return signal + np.random.normal(scale=np.sqrt(P_noise), size=signal.shape)

def plot_sparsity_vs_nmse(num_scales, sparsity_results, nmse_results):
    # Scatterplot of sparsity vs. nmse
    fig, ax = plt.subplots(1,2, figsize=(15,5))
    colors = {
        'Connection': plt.cm.Reds(np.linspace(0.4, 1, len(num_scales))),
        'Connection Normalized': plt.cm.Blues(np.linspace(0.4, 1, len(num_scales))),
        'Trivial': plt.cm.Greens(np.linspace(0.4, 1, len(num_scales))),
        'Trivial Normalized': plt.cm.Purples(np.linspace(0.4, 1, len(num_scales))),
        'Sheaf': plt.cm.Greys(np.linspace(0.4, 1, len(num_scales)))
    }
    markers = {'Connection': 'o', 'Connection Normalized': '*', 'Trivial': 'X', 'Trivial Normalized': '^', 'Sheaf': 'X',}
    for laplacian in ['Connection','Trivial']:
        for l, num_scal in enumerate(num_scales):
            ax[0].scatter(
                sparsity_results[laplacian][num_scal], nmse_results[laplacian][num_scal],
                color=colors[laplacian][l],
                marker=markers[laplacian],
                label=f"{laplacian} with {num_scal} scales"
            )
            ax[0].set_xlabel("Sparsity")
            ax[0].set_ylabel("NMSE")
            ax[0].set_title("Connection Laplacian vs. Trivial Laplacian")
            ax[0].legend()

    for laplacian in ['Connection Normalized', 'Trivial Normalized']:
        for l, num_scal in enumerate(num_scales):
            ax[1].scatter(
                sparsity_results[laplacian][num_scal], nmse_results[laplacian][num_scal],
                color=colors[laplacian][l],
                marker=markers[laplacian],
                label=f"{laplacian} with {num_scal} scales"
            )
            ax[1].set_xlabel("Sparsity")
            ax[1].set_ylabel("NMSE")
            ax[1].set_title("Normalized Connection Laplacian vs. Normalized Trivial Laplacian")
            ax[1].set_xlim(0.0995,0.1015)
            ax[1].legend()
    plt.show()


def plot_avg_results_vs_num_scales(num_scales, laplacians, sparsity_results, nmse_results):
    # Compute average sparsity and nmse for each laplacian and number of scales
    cube_sparsity_avg = {
        laplacian: {
            num_scal: np.mean(sparsity_results[laplacian][num_scal])
            for num_scal in num_scales
        }
        for laplacian in laplacians
    }
    cube_nmse_avg = {
        laplacian: {
            num_scal: np.mean(nmse_results[laplacian][num_scal])
            for num_scal in num_scales
        }
        for laplacian in laplacians
    }

    # Plot nmse curves
    fig, ax = plt.subplots(1,2, figsize=(15,5))
    for laplacian in laplacians:
        # NMSE vs. Number of Scales
        ax[0].plot(num_scales, [y[1] for y in sorted(cube_nmse_avg[laplacian].items(), key=lambda x: x[0])],label=laplacian)
        ax[0].set_xlabel("Number of scales")
        ax[0].set_ylabel("NMSE")
        ax[0].set_title(f"Average NMSE vs. Number of Scales")
        ax[0].legend()
        # Sparsity vs. Number of Scales
        ax[1].plot(num_scales, [y[1] for y in sorted(cube_sparsity_avg[laplacian].items(), key=lambda x: x[0])],label=laplacian)
        ax[1].set_xlabel("Number of scales")
        ax[1].set_ylabel("Sparsity")
        ax[1].set_title(f"Average Sparsity vs. Number of Scales")
        ax[1].legend()
    plt.show()


def plot_sparsity_vs_snr_rec(num_scales, sparsity_results, snr_rec_results):
    # Scatterplot of sparsity vs. nmse
    fig, ax = plt.subplots(1,2, figsize=(15,5))
    colors = {
        'Connection': plt.cm.Reds(np.linspace(0.4, 1, len(num_scales))),
        'Connection Normalized': plt.cm.Blues(np.linspace(0.4, 1, len(num_scales))),
        'Trivial': plt.cm.Greens(np.linspace(0.4, 1, len(num_scales))),
        'Trivial Normalized': plt.cm.Purples(np.linspace(0.4, 1, len(num_scales))),
        'Sheaf': plt.cm.Greys(np.linspace(0.4, 1, len(num_scales)))
    }
    markers = {'Connection': 'o', 'Connection Normalized': '*', 'Trivial': 'X', 'Trivial Normalized': '^', 'Sheaf': 'X',}
    for laplacian in ['Connection','Trivial']:
        for l, num_scal in enumerate(num_scales):
            ax[0].scatter(
                sparsity_results[laplacian][num_scal], snr_rec_results[laplacian][num_scal],
                color=colors[laplacian][l],
                marker=markers[laplacian],
                label=f"{laplacian} with {num_scal} scales"
            )
            ax[0].set_xlabel("Sparsity")
            ax[0].set_ylabel("Reconstruction SNR")
            ax[0].set_title("Connection Laplacian vs. Trivial Laplacian")
            ax[0].legend()

    for laplacian in ['Connection Normalized', 'Trivial Normalized']:
        for l, num_scal in enumerate(num_scales):
            ax[1].scatter(
                sparsity_results[laplacian][num_scal], snr_rec_results[laplacian][num_scal],
                color=colors[laplacian][l],
                marker=markers[laplacian],
                label=f"{laplacian} with {num_scal} scales"
            )
            ax[1].set_xlabel("Sparsity")
            ax[1].set_ylabel("Reconstruction SNR")
            ax[1].set_title("Normalized Connection Laplacian vs. Normalized Trivial Laplacian")
            ax[1].set_xlim(0.0995,0.1015)
            ax[1].legend()
    plt.show()

def plot_sparsity_vs_gain(num_scales, sparsity_results, snr_rec_results):
    # Scatterplot of sparsity vs. nmse
    fig, ax = plt.subplots(1,2, figsize=(15,5))
    colors = {
        'Connection': plt.cm.Reds(np.linspace(0.4, 1, len(num_scales))),
        'Connection Normalized': plt.cm.Blues(np.linspace(0.4, 1, len(num_scales))),
        'Trivial': plt.cm.Greens(np.linspace(0.4, 1, len(num_scales))),
        'Trivial Normalized': plt.cm.Purples(np.linspace(0.4, 1, len(num_scales))),
        'Sheaf': plt.cm.Greys(np.linspace(0.4, 1, len(num_scales)))
    }
    markers = {'Connection': 'o', 'Connection Normalized': '*', 'Trivial': 'X', 'Trivial Normalized': '^', 'Sheaf': 'X',}
    for laplacian in ['Connection','Trivial']:
        for l, num_scal in enumerate(num_scales):
            ax[0].scatter(
                sparsity_results[laplacian][num_scal], snr_rec_results[laplacian][num_scal],
                color=colors[laplacian][l],
                marker=markers[laplacian],
                label=f"{laplacian} with {num_scal} scales"
            )
            ax[0].set_xlabel("Sparsity")
            ax[0].set_ylabel("Gain")
            ax[0].set_title("Connection Laplacian vs. Trivial Laplacian")
            ax[0].legend()

    for laplacian in ['Connection Normalized', 'Trivial Normalized']:
        for l, num_scal in enumerate(num_scales):
            ax[1].scatter(
                sparsity_results[laplacian][num_scal], snr_rec_results[laplacian][num_scal],
                color=colors[laplacian][l],
                marker=markers[laplacian],
                label=f"{laplacian} with {num_scal} scales"
            )
            ax[1].set_xlabel("Sparsity")
            ax[1].set_ylabel("Gain")
            ax[1].set_title("Normalized Connection Laplacian vs. Normalized Trivial Laplacian")
            ax[1].set_xlim(0.0995,0.1015)
            ax[1].legend()
    plt.show()


def plot_avg_results_vs_num_scales2(num_scales, laplacians, snr_rec_results, gain_results):
    # Compute average sparsity and nmse for each laplacian and number of scales
    cube_sparsity_avg = {
        laplacian: {
            num_scal: np.mean(snr_rec_results[laplacian][num_scal])
            for num_scal in num_scales
        }
        for laplacian in laplacians
    }
    cube_nmse_avg = {
        laplacian: {
            num_scal: np.mean(gain_results[laplacian][num_scal])
            for num_scal in num_scales
        }
        for laplacian in laplacians
    }

    # Plot nmse curves
    fig, ax = plt.subplots(1,2, figsize=(15,5))
    for laplacian in laplacians:
        # NMSE vs. Number of Scales
        ax[0].plot(num_scales, [y[1] for y in sorted(cube_nmse_avg[laplacian].items(), key=lambda x: x[0])],label=laplacian)
        ax[0].set_xlabel("Number of scales")
        ax[0].set_ylabel("Reconstruction SNR")
        ax[0].set_title(f"Average SNR vs. Number of Scales")
        ax[0].legend()
        # Sparsity vs. Number of Scales
        ax[1].plot(num_scales, [y[1] for y in sorted(cube_sparsity_avg[laplacian].items(), key=lambda x: x[0])],label=laplacian)
        ax[1].set_xlabel("Number of scales")
        ax[1].set_ylabel("Gain")
        ax[1].set_title(f"Average Gain vs. Number of Scales")
        ax[1].legend()
    plt.show()
