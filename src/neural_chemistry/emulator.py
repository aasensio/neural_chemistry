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

    
class ChemistryEmulator(object):
    def __init__(self, gpu, verbose=False):

        self.verbose = verbose

        REQUIRED_FILES = [
            {
                'name': 'weights.pth',
                'url': 'https://cloud.iac.es/index.php/s/W8Qgw8Yy95BqstR/download',
            }
        ]

        DOWNLOAD_DIRECTORY = os.path.join(os.path.expanduser('~'), "neural_chemistry")

        ensure_files_exist(REQUIRED_FILES, DOWNLOAD_DIRECTORY)

        if self.verbose:
            print(f"Loading model weights")
        
        filename = os.path.join(os.path.expanduser('~'), "neural_chemistry/weights.pth")        

        chk = torch.load(filename, map_location=lambda storage, loc: storage, weights_only=False)

        self.n_mols = 192

        # Load the hyperparameters from the checkpoint        
        self.hyperparameters = chk['hyperparameters']

        # Load the normalization parameters from the checkpoint
        self.normalization = chk['normalization']

        # Check if CUDA is available and we want to compute on it
        self.cuda = torch.cuda.is_available()
        self.gpu = gpu

        # We can select the GPU by passing the index, or use CPU with -1
        if self.gpu < 0:
            self.cuda = False
        self.device = torch.device(f"cuda:{self.gpu}" if self.cuda else "cpu")
                                
        # Define the model
        self.encoding = GaussianEncoding(input_size=1,
                                                 sigma=self.hyperparameters['embedding']['sigma'],
                                                 encoding_size=self.hyperparameters['embedding']['encoding_size']).to(self.device)
        
        self.model = MLP(n_input=self.encoding.encoding_size,
                                n_output=1,
                                dim_hidden=self.hyperparameters['mlp']['n_hidden_mlp'],                                 
                                n_hidden=self.hyperparameters['mlp']['num_layers_mlp'],
                                activation=nn.ReLU()).to(self.device)
                
        self.condition_pars = MLPConditioning(n_input=5,
                                                n_output=self.hyperparameters['mlp']['n_hidden_mlp'] // 2,
                                                  dim_hidden=self.hyperparameters['condition_pars']['n_hidden'],
                                                  n_hidden=self.hyperparameters['condition_pars']['num_layers'],
                                                  activation=nn.ReLU()).to(self.device)
        
        self.condition_mol = MLPConditioning(n_input=self.n_mols,
                                                n_output=self.hyperparameters['mlp']['n_hidden_mlp'] // 2,
                                                  dim_hidden=self.hyperparameters['condition_mol']['n_hidden'],
                                                  n_hidden=self.hyperparameters['condition_mol']['num_layers'],
                                                  activation=nn.ReLU()).to(self.device)
                
        if self.verbose:
            print('N. total parameters MLP :            {0}'.format(sum(p.numel() for p in self.model.parameters() if p.requires_grad)))
            print('N. total parameters CONDITION-PARS : {0}'.format(sum(p.numel() for p in self.condition_pars.parameters() if p.requires_grad)))
            print('N. total parameters CONDITION-MOL :  {0}'.format(sum(p.numel() for p in self.condition_mol.parameters() if p.requires_grad)))
            print("Setting weights of the model...")

        self.encoding.load_state_dict(chk['encoding_dict'])
        self.condition_pars.load_state_dict(chk['condition_pars_dict'])
        self.condition_mol.load_state_dict(chk['condition_mol_dict'])
        self.model.load_state_dict(chk['model_dict'])

        # Freeze all neural networks and set to evaluation mode
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.condition_mol.parameters():
            param.requires_grad = False
        for param in self.condition_pars.parameters():
            param.requires_grad = False

        self.model.eval()
        self.condition_pars.eval()
        self.condition_mol.eval()        

    def evaluate(self, t, T, nh, crir, sulfur, uv_flux, batch_size=32, species=None):
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
        good_nh = ((nh >= 1e4) & (nh <= 1e7)).all()
        if not good_nh:
            raise ValueError("nh is not in the correct range [1e4,1e7] cm-3")
        good_T = ((T >= 10.0) & (T <= 80.0)).all()
        if not good_T:
            raise ValueError("T is not in the correct range [10,80] K")
        good_crir = ((crir >= 1e-17) & (crir <= 1e-15)).all()
        if not good_crir:
            raise ValueError("crir is not in the correct range [1e-17,1e-15] s-1")
        good_sulfur = ((sulfur >= 7.5e-8) & (sulfur <= 1.5e-5)).all()
        if not good_sulfur:
            raise ValueError("sulfur is not in the correct range [7.5e-8,1.5e-5] cm-3")
        good_uv = ((uv_flux >= 0.1) & (uv_flux <= 1e4)).all()
        if not good_uv:
            raise ValueError("uv_flux is not in the correct range [0.1,1e4]")
        
        if self.verbose:
            print("Working on parameters...")
        self.pars = np.vstack([np.log10(nh), T, np.log10(crir), np.log10(sulfur), np.log10(uv_flux)]).T
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
        
        with torch.no_grad():

            for i in tqdm(range(len(ind))):

                # Transform molecule to one-hot encoding
                mol_1hot = F.one_hot(mol[ind[i], :], num_classes=self.n_mols).float()
                
                # FiLM conditioning
                beta_pars, gamma_pars = self.condition_pars(pars[ind[i], :, :])
                beta_mol, gamma_mol = self.condition_mol(mol_1hot)

                beta = torch.cat((beta_pars, beta_mol), dim=-1)
                gamma = torch.cat((gamma_pars, gamma_mol), dim=-1)
                                
                # Encode time
                logt_encoded = self.encoding(logt[:, None], alpha=1.0)
                
                # MLP
                out = self.model(logt_encoded[None, None, :, :], beta=beta[:, :, None, :], gamma=gamma[:, :, None, :]).squeeze(-1)
                
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
