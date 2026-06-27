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
import gstools as gs
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

    def generate_kraichnan_signals(
        self,
        num_signals=500,
        Sigma=None,
        len_scale=10,
        SEED=42):
        """
        Generate tangent bundle signals using GSTools.

        Parameters
        ----------
        num_signals : int
            Number of random fields to generate.

        Sigma : ndarray, optional
            3x3 covariance between vector components.

        len_scale : float
            Spatial correlation length.

        SEED : int
            Random seed.

        Returns
        -------
        X : ndarray
            Tangent bundle signals (2N x num_signals)

        covariance : ndarray
            Empirical covariance matrix

        X_GT : ndarray
            Ground truth signals
        """

        N = self.data.shape[0]

        self._ensure_wav_object()
        self._ensure_orthonormal_bases()

        O = self.orthonormal_bases

        dummy_X = np.zeros((2 * N, num_signals))
        dummy_cov = np.zeros((2 * N, 2 * N))

        sample = CochainSample(
            X=dummy_X,
            covariance=dummy_cov,
            X_GT=dummy_X.copy(),
            points=self.data,
            local_bases=O,
            V=N,
        )

        sampled = sample.random_tangent_bundle_signals(
            Sigma=Sigma,
            len_scale=len_scale,
            M=num_signals,
            seed=SEED,
        )

        return sampled.X, sampled.covariance, sampled.X_GT
        

    # Function that computes sparse signal representations using OMP or CVXPY
    def sparsify_signals(self, X, dictionary, num_atoms, method = 'OMP'):
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
            sparse_signals[:,k] = self.wav.sparse_signal(X[:,k], dictionary, num_atoms, method=method)
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
    




############################################################################################################
########################################### SIGNAL COMPRESSION #############################################
############################################################################################################


def signal_compression_exp(point_cloud, hyperparameters):
    '''
    Function that takes in input a point cloud and a dictionary of hyperparameters and performs signal compression experiment 3 on the data
    Returns:
    - sparsity_results
    - nmse_results
    '''
    # Initialize result dictionaries
    #sparsity_results = defaultdict(dict)
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
    num_atoms = hyperparameters['num_atoms']
    len_scale=10

    ###
    #print('Experiment 3 running')
    ###

    Tsp_signal = TSP(
        point_cloud,
        eps=eps,
        eps_pca=eps_pca,
        k=k,
        laplacian_code=laplacians[0],
        gamma=gamma
    )

    signals, cov, signals_GT = Tsp_signal.generate_kraichnan_signals(
        num_signals=num_signals,
        len_scale=len_scale,
        SEED=SEED
    )

    for laplacian in laplacians:

        #print(f'Laplacian: {laplacian}')

        # Create TSP (topological signal processing) object
        Tsp = TSP(point_cloud, eps=eps, eps_pca=eps_pca, k=k, laplacian_code=laplacian, gamma=gamma)

        for num_scal in num_scales[::-1]:
            
            ###
            #print(f'Number of scales: {num_scal}')
            #print(f'Scales: {[2**(j-num_scal//2) for j in range(num_scal)]}')
            ###

            # Create dictionary
            dictionary = Tsp.create_dictionary(scales=[2**(j-num_scal//2) for j in range(num_scal)], normalize=False, adjust_kernel=adjust_kernel)

            #if num_scal == num_scales[-1]:
                # Generate signals for the corresponding laplacian and dictionary
            #    signals, cov, signals_GT = Tsp.generate_geometric_signals(num_signals=num_signals, SEED=SEED)

            sparse_signals = dict()
            sparsity = dict()

            nmse = dict()

            for num in num_atoms:
                sparse_signals[num] = Tsp.sparsify_signals(signals, dictionary, num)

                #sparsity[num] = Tsp.compute_sparsity(sparse_signals[num])

                nmse[num] = Tsp.compute_NMSE(signals, sparse_signals[num],dictionary)

            #sparsity_results[laplacian][num_scal] = sparsity
            nmse_results[laplacian][num_scal] = nmse

    #return sparsity_results, nmse_results
    return nmse_results



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



############################################################################################################
########################################## SIGNAL DENOISING ################################################
############################################################################################################




# Function that adds noise to a signal based on the SNR
def add_noise(signal, SNR):
    P_signal = np.mean(signal**2)
    P_noise = P_signal / SNR
    return signal + np.random.normal(scale=np.sqrt(P_noise), size=signal.shape)



def signal_denoising(point_cloud, hyperparameters, gt_signals):
    '''
    Function that takes in input a point cloud and a dictionary of hyperparameters and performs signal compression experiment 3 on the data
    Returns:
    - sparsity_results
    - nmse_results
    '''
    # Initialize result dictionaries
    sparsity_results = defaultdict(dict)
    nmse_results = defaultdict(dict)
    snr_rec_results = defaultdict(dict)
    snr_gain_results = defaultdict(dict)

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
    SNR = hyperparameters['SNR']

    # Add noise
    signals = add_noise(gt_signals, SNR)

    for laplacian in laplacians:

        # Create TSP (topological signal processing) object
        Tsp = TSP(point_cloud, eps=eps, eps_pca=eps_pca, k=k, laplacian_code=laplacian, gamma=gamma, h=h, t=t, p=p)

        for num_scal in num_scales:
            # Create dictionary
            dictionary = Tsp.create_dictionary(scales=[2**(j-num_scal//2) for j in range(num_scal)], normalize=normalize, adjust_kernel=adjust_kernel)
            
            # Sparsify signals
            sparse_signals = Tsp.sparsify_signals(signals, dictionary)

            # Compute sparsity
            sparsity =  Tsp.compute_sparsity(sparse_signals)
            
            # Compute NMSE
            nmse = Tsp.compute_NMSE(gt_signals, sparse_signals, dictionary)

            # Reconstruct signals
            reconstructed_signals = Tsp.reconstruct_signals(sparse_signals, dictionary)

            # Compute Reconstruction SNR
            SNR_rec = Tsp.compute_snr_rec(gt_signals, reconstructed_signals)

            # Compute SNR Gain
            SNR_gain = Tsp.compute_snr_gain(SNR_rec, SNR)

            # Add results to the dictionaries
            sparsity_results[laplacian][num_scal] = sparsity
            nmse_results[laplacian][num_scal] = nmse
            snr_rec_results[laplacian][num_scal] = SNR_rec
            snr_gain_results[laplacian][num_scal] = SNR_gain

    return sparsity_results, nmse_results, snr_rec_results, snr_gain_results



############################################################################################################
############################################## PLOTTING ####################################################
############################################################################################################



def plot_nmse(nmse_results, scale=3):

    plt.figure(figsize=(7, 5))

    for laplacian in nmse_results:

        if laplacian=='Sheaf':
            continue
        
        atom_dict = nmse_results[laplacian][scale]

        Ks = sorted(atom_dict.keys())
        ys = [np.mean(atom_dict[K]) for K in Ks]

        plt.plot(Ks, ys, marker="o", label=laplacian)

    plt.xlabel("Number of non-zero coefficients")
    plt.ylabel("NMSE")
    plt.yscale("log")
    plt.grid(True, which="both", alpha=0.4)
    plt.legend()
    plt.title('NMSE vs Number of Non-Zero Coefficients (Sphere, Signal Compression)')
    plt.tight_layout()
    plt.show()