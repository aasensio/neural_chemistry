import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from .encoding import GaussianEncoding 
from .mlp import MLP, MLPConditioning
from .normalization import normalize, denormalize
from .io import ensure_files_exist
from tqdm import tqdm
import scipy.interpolate as interp

    
class ChemistryEmulator2(object):
    def __init__(self, gpu, verbose=False):

        self.verbose = verbose

        REQUIRED_FILES = [
            {
                'name': '2025-07-14-19_09_54.best.pth',
                'url': 'https://cloud.iac.es/index.php/s/2TqdbogWqcfppCB/download',
            },
            {
                'name': '2025-07-17-15_23_49.best.pth',
                'url': 'https://cloud.iac.es/index.php/s/R2wNkNMZMT3dTtG/download',
            }
        ]

        DOWNLOAD_DIRECTORY = os.path.join(os.path.expanduser('~'), "neural_chemistry")

        ensure_files_exist(REQUIRED_FILES, DOWNLOAD_DIRECTORY)

        if self.verbose:
            print(f"Loading model weights")

        path = str(__file__).split('/')
        filename = os.path.join(os.path.expanduser('~'), 'neural_chemistry/2025-07-14-19_09_54.best.pth')
        chk0 = torch.load(filename, map_location=lambda storage, loc: storage, weights_only=False)

        filename = os.path.join(os.path.expanduser('~'), 'neural_chemistry/2025-07-17-15_23_49.best.pth')
        chk1 = torch.load(filename, map_location=lambda storage, loc: storage, weights_only=False)

        self.n_mols = 250

        # Load the hyperparameters from the checkpoint        
        self.hyperparameters = chk0['hyperparameters']

        # Load the normalization parameters from the checkpoint
        self.normalization = chk0['normalization']

        # Check if CUDA is available and we want to compute on it
        self.cuda = torch.cuda.is_available()
        self.gpu = gpu

        # We can select the GPU by passing the index, or use CPU with -1
        if self.gpu < 0:
            self.cuda = False
        self.device = torch.device(f"cuda:{self.gpu}" if self.cuda else "cpu")

        #----------------------
        # Define the model 0
        #----------------------
        self.encoding0 = GaussianEncoding(input_size=1,
                                                 sigma=self.hyperparameters['embedding']['sigma'],
                                                 encoding_size=self.hyperparameters['embedding']['encoding_size']).to(self.device)
        
        self.model0 = MLP(n_input=self.encoding0.encoding_size,
                                n_output=1,
                                dim_hidden=self.hyperparameters['mlp']['n_hidden_mlp'],                                 
                                n_hidden=self.hyperparameters['mlp']['num_layers_mlp'],
                                activation=nn.GELU(approximate='tanh')).to(self.device)
                
        self.condition_pars0 = MLPConditioning(n_input=6,
                                                n_output=self.hyperparameters['mlp']['n_hidden_mlp'] // 2,
                                                  dim_hidden=self.hyperparameters['condition_pars']['n_hidden'],
                                                  n_hidden=self.hyperparameters['condition_pars']['num_layers'],
                                                  activation=nn.GELU(approximate='tanh')).to(self.device)
        
        self.condition_mol0 = MLPConditioning(n_input=self.n_mols,
                                                n_output=self.hyperparameters['mlp']['n_hidden_mlp'] // 2,
                                                  dim_hidden=self.hyperparameters['condition_mol']['n_hidden'],
                                                  n_hidden=self.hyperparameters['condition_mol']['num_layers'],
                                                  activation=nn.GELU(approximate='tanh')).to(self.device)
        
         #----------------------
        # Define the model 1
        #----------------------
        self.encoding1 = GaussianEncoding(input_size=1,
                                                 sigma=self.hyperparameters['embedding']['sigma'],
                                                 encoding_size=self.hyperparameters['embedding']['encoding_size']).to(self.device)
        
        self.model1 = MLP(n_input=self.encoding1.encoding_size,
                                n_output=1,
                                dim_hidden=self.hyperparameters['mlp']['n_hidden_mlp'],                                 
                                n_hidden=self.hyperparameters['mlp']['num_layers_mlp'],
                                activation=nn.GELU(approximate='tanh')).to(self.device)
                
        self.condition_pars1 = MLPConditioning(n_input=6,
                                                n_output=self.hyperparameters['mlp']['n_hidden_mlp'] // 2,
                                                  dim_hidden=self.hyperparameters['condition_pars']['n_hidden'],
                                                  n_hidden=self.hyperparameters['condition_pars']['num_layers'],
                                                  activation=nn.GELU(approximate='tanh')).to(self.device)
        
        self.condition_mol1 = MLPConditioning(n_input=self.n_mols,
                                                n_output=self.hyperparameters['mlp']['n_hidden_mlp'] // 2,
                                                  dim_hidden=self.hyperparameters['condition_mol']['n_hidden'],
                                                  n_hidden=self.hyperparameters['condition_mol']['num_layers'],
                                                  activation=nn.GELU(approximate='tanh')).to(self.device)
                
        if self.verbose:
            print('Base model:')
            print('N. total parameters MLP :            {0}'.format(sum(p.numel() for p in self.model0.parameters() if p.requires_grad)))
            print('N. total parameters CONDITION-PARS : {0}'.format(sum(p.numel() for p in self.condition_pars0.parameters() if p.requires_grad)))
            print('N. total parameters CONDITION-MOL :  {0}'.format(sum(p.numel() for p in self.condition_mol0.parameters() if p.requires_grad)))
            print('Refinement model:')
            print('N. total parameters MLP :            {0}'.format(sum(p.numel() for p in self.model1.parameters() if p.requires_grad)))
            print('N. total parameters CONDITION-PARS : {0}'.format(sum(p.numel() for p in self.condition_pars1.parameters() if p.requires_grad)))
            print('N. total parameters CONDITION-MOL :  {0}'.format(sum(p.numel() for p in self.condition_mol1.parameters() if p.requires_grad)))
            print("Setting weights of the model...")

        self.encoding0.load_state_dict(chk0['encoding_dict'])
        self.condition_pars0.load_state_dict(chk0['condition_pars_dict'])
        self.condition_mol0.load_state_dict(chk0['condition_mol_dict'])
        self.model0.load_state_dict(chk0['model_dict'])

        # Freeze all neural networks and set to evaluation mode
        for param in self.model0.parameters():
            param.requires_grad = False
        for param in self.condition_mol0.parameters():
            param.requires_grad = False
        for param in self.condition_pars0.parameters():
            param.requires_grad = False

        self.model0.eval()
        self.condition_pars0.eval()
        self.condition_mol0.eval()


        self.encoding1.load_state_dict(chk1['encoding_dict'])
        self.condition_pars1.load_state_dict(chk1['condition_pars_dict'])
        self.condition_mol1.load_state_dict(chk1['condition_mol_dict'])
        self.model1.load_state_dict(chk1['model_dict'])

        # Freeze all neural networks and set to evaluation mode
        for param in self.model1.parameters():
            param.requires_grad = False
        for param in self.condition_mol1.parameters():
            param.requires_grad = False
        for param in self.condition_pars1.parameters():
            param.requires_grad = False

        self.model1.eval()
        self.condition_pars1.eval()
        self.condition_mol1.eval()

    def evaluate(self, t, T, nh, crir, sulfur, uv_flux, Av, batch_size=32, species=None):
        self.n_models = len(T)

        # Generate the indices for all molecules and expand to the number of physical conditions that we want
        if species is None:
            mol_idx = np.arange(self.n_mols)
            self.n_mols_compute = self.n_mols
        else:
            mol_idx = np.array(species)
            self.n_mols_compute = len(mol_idx)
        self.mol_idx = np.tile(mol_idx, (self.n_models, 1))

        # Define the time axis and check that it lies in the correct range
        good_times = ((t >= 1.0) & (t <= 1e7)).all()
        if good_times:
            self.logt = np.log10(t)        
            self.logt, _, _ = normalize(self.logt, xmin=self.normalization['logt'][0], xmax=self.normalization['logt'][1], axis=0)                
        else:
            raise ValueError("Time axis is not in the correct range [1,1e7] yr")
                
        # Check input parameters
        good_nh = ((nh >= 1e3) & (nh <= 1e8)).all()
        if not good_nh:
            raise ValueError("nh is not in the correct range [1e3,1e8] cm-3")
        good_T = ((T >= 5.0) & (T <= 100.0)).all()
        if not good_T:
            raise ValueError("T is not in the correct range [5,100] K")
        good_crir = ((crir >= 1e-17) & (crir <= 1e-15)).all()
        if not good_crir:
            raise ValueError("crir is not in the correct range [1e-17,1e-15] s-1")
        good_sulfur = ((sulfur >= 7.5e-8) & (sulfur <= 1.5e-5)).all()
        if not good_sulfur:
            raise ValueError("sulfur is not in the correct range [7.5e-8,1.5e-5] cm-3")
        good_uv = ((uv_flux >= 0.1) & (uv_flux <= 1e5)).all()
        if not good_uv:
            raise ValueError("uv_flux is not in the correct range [0.1,1e5]")
        good_Av = ((Av >= 0.0) & (Av <= 18.0)).all()
        if not good_Av:
            raise ValueError("Av is not in the correct range [0.0,18.0]")
        
        if self.verbose:
            print("Working on parameters...")
        self.pars = np.vstack([np.log10(nh), T, np.log10(crir), np.log10(sulfur), np.log10(uv_flux), Av]).T
        self.pars, _, _ = normalize(self.pars, xmin=self.normalization['pars'][0], xmax=self.normalization['pars'][1], axis=0)
        self.pars = np.tile(self.pars[:, None, :], (1, self.n_mols_compute, 1))

                    
        self.batch_size = batch_size
        
        # Generate tensor from the arrays
        logt = torch.tensor(self.logt.astype('float32'))
        pars = torch.tensor(self.pars.astype('float32'))
        mol = torch.tensor(self.mol_idx.astype('int'))
        
        # Move to GPU if available
        logt = logt.to(self.device)
        pars = pars.to(self.device)
        mol = mol.to(self.device)

        # Split the data in batches
        n = self.pars.shape[0]        
        ind = np.arange(n)
        if n > self.batch_size:
            ind = np.array_split(ind, n // self.batch_size)
        else:
            ind = [ind]

        out_all = np.zeros((n, self.n_mols_compute, len(logt)), dtype='float32')

        if self.verbose:
            disable_tqdm = False
        else:
            disable_tqdm = True
        
        with torch.no_grad():

            for i in tqdm(range(len(ind)), disable=disable_tqdm):
                # Transform molecule to one-hot encoding
                mol_1hot = F.one_hot(mol[ind[i], :], num_classes=self.n_mols).float()
    
                #--------------------
                # Base model
                #--------------------                
                # FiLM conditioning
                beta_pars, gamma_pars = self.condition_pars0(pars[ind[i], :, :])
                beta_mol, gamma_mol = self.condition_mol0(mol_1hot)

                beta = torch.cat((beta_pars, beta_mol), dim=-1)
                gamma = torch.cat((gamma_pars, gamma_mol), dim=-1)
                                
                # Encode time
                logt_encoded = self.encoding0(logt[:, None], alpha=1.0)
                
                # MLP
                out0 = self.model0(logt_encoded[None, None, :, :], beta=beta[:, :, None, :], gamma=gamma[:, :, None, :]).squeeze(-1)


                #--------------------
                # Refinement model
                #--------------------                
                # FiLM conditioning
                beta_pars, gamma_pars = self.condition_pars1(pars[ind[i], :, :])
                beta_mol, gamma_mol = self.condition_mol1(mol_1hot)

                beta = torch.cat((beta_pars, beta_mol), dim=-1)
                gamma = torch.cat((gamma_pars, gamma_mol), dim=-1)
                                
                # Encode time
                logt_encoded = self.encoding1(logt[:, None], alpha=1.0)
                
                # MLP
                out1 = self.model1(logt_encoded[None, None, :, :], beta=beta[:, :, None, :], gamma=gamma[:, :, None, :]).squeeze(-1)

                out = out0 + out1
                
                out_all[ind[i], :, :] = out.cpu().numpy()

        # Abundance normalization was obtained for 64 time steps. Resample to the desired time steps
        logab_min = self.normalization['abund_min_max'][0]
        logab_max = self.normalization['abund_min_max'][1]

        logt_ref = np.linspace(-1, 1, 64)
        logab_min_interp = interp.interp1d(logt_ref, logab_min, axis=1, kind='linear', bounds_error=False, fill_value=(np.nan, np.nan))
        logab_max_interp = interp.interp1d(logt_ref, logab_max, axis=1, kind='linear', bounds_error=False, fill_value=(np.nan, np.nan))

        logab_min_new = logab_min_interp(self.logt)
        logab_max_new = logab_max_interp(self.logt)
        
        # Undo the normalization
        out_all = denormalize(out_all, logab_min_new[None, mol_idx, :], logab_max_new[None, mol_idx, :])

        # Undo the log
        out_all = 10.0**out_all - 1e-45
                
        return out_all
