import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import time
from tqdm import tqdm
try:
    import nvidia_smi
    NVIDIA_SMI = True
except:
    NVIDIA_SMI = False
from collections import OrderedDict
import pathlib
import zarr
import sys
sys.path.append('../modules')
import encoding
import mlp
import normalization


class Dataset(torch.utils.data.Dataset):
    """
    Dataset class that will provide data during training. Modify it accordingly
    for your dataset. This one shows how to do augmenting during training for a 
    very simple training set    
    """
    def __init__(self):
        """
        Very simple training set made of 200 Gaussians of width between 0.5 and 1.5
        We later augment this with a velocity and amplitude.
        
        Args:
            n_training (int): number of training examples including augmenting
        """
        super(Dataset, self).__init__()
        
        self.n_times = 64

        # We currently read the whole dataset in memory and do the parameter transformations
        # here. If the dataset is too large, we can do the transformations in the __getitem__
        print("Reading data...")
        f = zarr.open('/scratch1/aasensio/chemistry/merged_db_150k_192mol.zarr', 'r')

        t = f['t'][:]
        nh = f['nh'][:]
        T = f['T'][:]
        crir = f['crir'][:]
        sulfur = f['sulfur'][:]
        uv_flux = f['uv_flux'][:]
        self.logab = f['logab']
        self.logab_min = f['logab_min'][:]
        self.logab_max = f['logab_max'][:]
                
        # Count the number of models        
        self.n_mols = f['logab_min'].shape[0]
        self.n_models = self.logab.shape[0] // self.n_mols
        
        # Time
        print("Working on time...")
        self.logt = np.log10(t)
        self.logt, self.logt_min, self.logt_max = normalization.normalize(self.logt, axis=0)
    
        # Input parameters
        print("Working on parameters...")
        self.pars = np.vstack([np.log10(nh), T, np.log10(crir), np.log10(sulfur), np.log10(uv_flux)]).T
        self.pars, self.pars_min, self.pars_max = normalization.normalize(self.pars, axis=0)
        self.pars = np.tile(self.pars[:, None, :], (1, self.n_mols, 1))
        

        # Molecule indices
        print("Working on molecules...")
        mol_idx = np.arange(self.n_mols)
        self.mol_idx = np.tile(mol_idx, (self.n_models, 1))

        print("Flattening and reordering arrays...")
        self.pars = self.pars.reshape((self.n_models*self.n_mols, 5))
        self.mol_idx = self.mol_idx.reshape((self.n_models*self.n_mols))

        self.n_training = self.pars.shape[0]
        
    def __getitem__(self, index):
        # Get input and output
        pars = self.pars[index, :]
        logt = self.logt[:]        
        mol_idx = self.mol_idx[index]
        logab = self.logab[index, :]

        return logt.astype('float32'), pars.astype('float32'), logab.astype('float32'), mol_idx.astype(int)

    def __len__(self):
        return self.n_training        

class Training(object):
    def __init__(self, hyperparameters):

        self.hyperparameters = hyperparameters

        self.cuda = torch.cuda.is_available()
        self.gpu = hyperparameters['gpu']
        self.smooth = hyperparameters['smooth']
        self.device = torch.device(f"cuda:{self.gpu}" if self.cuda else "cpu")

        if (NVIDIA_SMI):
            nvidia_smi.nvmlInit()
            self.handle = nvidia_smi.nvmlDeviceGetHandleByIndex(self.gpu)
            print("Computing in {0} : {1}".format(self.device, nvidia_smi.nvmlDeviceGetName(self.handle)))
        
        self.batch_size = hyperparameters['batch_size']        
                
        kwargs = {'num_workers': 4, 'pin_memory': False} if self.cuda else {}

        self.dataset = Dataset()

        self.normalization = {
            'pars': [self.dataset.pars_min, self.dataset.pars_max],            
            'abund_min_max': [self.dataset.logab_min, self.dataset.logab_max],
            'logt': [self.dataset.logt_min, self.dataset.logt_max]
        }
        
        self.validation_split = hyperparameters['validation_split']
        idx = np.arange(self.dataset.n_training)
                
        self.train_index = idx[0:int((1-self.validation_split)*self.dataset.n_training)]
        self.validation_index = idx[int((1-self.validation_split)*self.dataset.n_training):]

        print(f"Dataset size: {self.dataset.n_training}")
        print(f"Training dataset size: {len(self.train_index)}")
        print(f"Validation dataset size: {len(self.validation_index)}")

        # Define samplers for the training and validation sets
        self.train_sampler = torch.utils.data.sampler.SubsetRandomSampler(self.train_index)
        self.validation_sampler = torch.utils.data.sampler.SubsetRandomSampler(self.validation_index)
                
        # Data loaders that will inject data during training
        self.train_loader = torch.utils.data.DataLoader(self.dataset, sampler=self.train_sampler, batch_size=self.batch_size, shuffle=False, persistent_workers=True, **kwargs)
        self.validation_loader = torch.utils.data.DataLoader(self.dataset, sampler=self.validation_sampler, batch_size=self.batch_size, shuffle=False, **kwargs)

        # self.train_dataset = torch.utils.data.Subset(self.dataset, self.train_index)
        # self.validation_dataset = torch.utils.data.Subset(self.dataset, self.validation_index)

        # Data loaders that will inject data during training
        # self.train_loader = torch.utils.data.DataLoader(self.train_dataset, 
        #             batch_size=self.batch_size,
        #             shuffle=False,
        #             drop_last=False, 
        #             **kwargs)

        # self.validation_loader = torch.utils.data.DataLoader(self.validation_dataset, 
        #             batch_size=self.batch_size,
        #             shuffle=False,
        #             drop_last=False,                     
        #             **kwargs)
        
        # Model
        self.encoding = encoding.GaussianEncoding(input_size=1,
                                                 sigma=self.hyperparameters['embedding']['sigma'],
                                                 encoding_size=self.hyperparameters['embedding']['encoding_size']).to(self.device)
        
        self.model = mlp.MLP(n_input=self.encoding.encoding_size,
                                n_output=1,
                                dim_hidden=self.hyperparameters['mlp']['n_hidden_mlp'],                                 
                                n_hidden=self.hyperparameters['mlp']['num_layers_mlp'],
                                activation=nn.ReLU()).to(self.device)
                
        self.condition_pars = mlp.MLPConditioning(n_input=5,
                                                n_output=self.hyperparameters['mlp']['n_hidden_mlp'] // 2,
                                                  dim_hidden=self.hyperparameters['condition_pars']['n_hidden'],
                                                  n_hidden=self.hyperparameters['condition_pars']['num_layers'],
                                                  activation=nn.ReLU()).to(self.device)
        
        self.condition_mol = mlp.MLPConditioning(n_input=self.dataset.n_mols,
                                                n_output=self.hyperparameters['mlp']['n_hidden_mlp'] // 2,
                                                  dim_hidden=self.hyperparameters['condition_mol']['n_hidden'],
                                                  n_hidden=self.hyperparameters['condition_mol']['num_layers'],
                                                  activation=nn.ReLU()).to(self.device)
                
        print('N. total parameters MLP :            {0}'.format(sum(p.numel() for p in self.model.parameters() if p.requires_grad)))
        print('N. total parameters CONDITION-PARS : {0}'.format(sum(p.numel() for p in self.condition_pars.parameters() if p.requires_grad)))
        print('N. total parameters CONDITION-MOL :  {0}'.format(sum(p.numel() for p in self.condition_mol.parameters() if p.requires_grad)))

    def init_optimize(self):

        self.lr = hyperparameters['lr']
        self.wd = hyperparameters['wd']
        self.n_epochs = hyperparameters['n_epochs']
        
        print('Learning rate : {0}'.format(self.lr))        
        
        p = pathlib.Path('weights/')
        p.mkdir(parents=True, exist_ok=True)

        current_time = time.strftime("%Y-%m-%d-%H:%M:%S").replace(':', '_')
        self.out_name = 'weights/{0}'.format(current_time)

        # Train all networks of the whole model        
        parameters = list(self.model.parameters()) + list(self.condition_pars.parameters()) + list(self.condition_mol.parameters())
        self.optimizer = torch.optim.Adam(parameters, lr=self.lr, weight_decay=self.wd)
        self.loss_fn = nn.MSELoss().to(self.device)
        
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.n_epochs, eta_min=0.1*self.lr)

    def optimize(self):
        self.loss = []
        self.loss_val = []
        best_loss = 1e100

        self.alpha = torch.tensor(0.0).to(self.device)
        self.iter = 0
        
        print('Model : {0}'.format(self.out_name))

        for epoch in range(1, self.n_epochs + 1):            
            loss = self.train(epoch)
            loss_val = self.test()
            
            self.scheduler.step()

            checkpoint = {
                'epoch': epoch + 1,
                'encoding_dict': self.encoding.state_dict(),
                'model_dict': self.model.state_dict(),
                'condition_pars_dict': self.condition_pars.state_dict(),
                'condition_mol_dict': self.condition_mol.state_dict(),
                'best_loss': best_loss,
                'loss': self.loss,
                'loss_val': self.loss_val,
                'normalization': self.normalization,
                'optimizer': self.optimizer.state_dict(),
                'hyperparameters': self.hyperparameters
            }

            torch.save(checkpoint, f'{self.out_name}.pth')

            if (loss_val < best_loss):
                print(f"Saving model {self.out_name}.best.pth")                
                best_loss = loss_val
                torch.save(checkpoint, f'{self.out_name}.best.pth')

            if (self.hyperparameters['save_all_epochs']):
                torch.save(checkpoint, f'{self.out_name}.ep_{epoch}.pth')

    def alpha_schedule(self, iter):
        if (iter < self.hyperparameters['alpha_initial_iteration']):
            y = 0.0
        elif (iter > self.hyperparameters['alpha_final_iteration']):
            y = 1.0
        else:
            x0 = self.hyperparameters['alpha_initial_iteration']
            x1 = self.hyperparameters['alpha_final_iteration']
            y0 = 0.0
            y1 = 1.0
            y = np.clip((y1 - y0) / (x1 - x0) * (iter - x0) + y0, y0, y1)
        
        return torch.tensor(float(y)).to(self.device)

    def train(self, epoch):
        self.model.train()
        self.condition_pars.train()
        self.condition_mol.train()
        print("Epoch {0}/{1}".format(epoch, self.n_epochs))
        t = tqdm(self.train_loader)
        loss_avg = 0.0
        
        for param_group in self.optimizer.param_groups:
            current_lr = param_group['lr']

        for batch_idx, (logt, pars, logab, mol) in enumerate(t):
            logt = logt.to(self.device)
            pars = pars.to(self.device)
            logab = logab.to(self.device)
            mol = mol.to(self.device)

            # Transform molecule to one-hot encoding
            mol = F.one_hot(mol, num_classes=self.dataset.n_mols).float()

            self.optimizer.zero_grad()

            # FiLM conditioning
            beta_pars, gamma_pars = self.condition_pars(pars)
            beta_mol, gamma_mol = self.condition_mol(mol)

            beta = torch.cat((beta_pars, beta_mol), dim=1)
            gamma = torch.cat((gamma_pars, gamma_mol), dim=1)

            # Encode time
            logt_encoded = self.encoding(logt[0, :][:, None], alpha=self.alpha)

            # MLP
            out = self.model(logt_encoded[None, :, :], beta=beta[:, None, :], gamma=gamma[:, None, :]).squeeze()
            
            # Loss
            loss = self.loss_fn(out, logab)
                    
            loss.backward()

            self.optimizer.step()

            if (batch_idx == 0):
                loss_avg = loss.item()
            else:
                loss_avg = self.smooth * loss.item() + (1.0 - self.smooth) * loss_avg

            if (NVIDIA_SMI):
                tmp = nvidia_smi.nvmlDeviceGetUtilizationRates(self.handle)
                gpu_usage = f'{tmp.gpu}'
                tmp = nvidia_smi.nvmlDeviceGetMemoryInfo(self.handle)
                memory_usage = f' {tmp.used / tmp.total * 100.0:4.1f}'                
            else:
                gpu_usage = 'NA'
                memory_usage = 'NA'

            tmp = OrderedDict()
            tmp['gpu'] = gpu_usage
            tmp['mem'] = memory_usage
            tmp['lr'] = current_lr
            tmp['alpha'] = f'{self.alpha:8.6f}'
            tmp['iter'] = f'{self.iter:8.6f}'
            tmp['loss'] = loss_avg
            t.set_postfix(ordered_dict = tmp)
            
            self.loss.append(loss_avg)

            self.alpha = torch.clamp(self.alpha_schedule(self.iter), 0.0, 1.0)

            self.iter += 1
        
        return loss_avg

    def test(self):
        self.model.eval()
        self.condition_pars.eval()
        self.condition_mol.eval()
        t = tqdm(self.validation_loader)
        loss_avg = 0.0

        with torch.no_grad():
            for batch_idx, (logt, pars, logab, mol) in enumerate(t):
                logt = logt.to(self.device)
                pars = pars.to(self.device)
                logab = logab.to(self.device)
                mol = mol.to(self.device)

                # Transform molecule to one-hot encoding
                mol = F.one_hot(mol, num_classes=self.dataset.n_mols).float()

                # FiLM conditioning
                beta_pars, gamma_pars = self.condition_pars(pars)
                beta_mol, gamma_mol = self.condition_mol(mol)

                beta = torch.cat((beta_pars, beta_mol), dim=1)
                gamma = torch.cat((gamma_pars, gamma_mol), dim=1)
                
                # Encode time
                logt_encoded = self.encoding(logt[0, :][:, None], alpha=self.alpha)

                # MLP
                out = self.model(logt_encoded[None, :, :], beta=beta[:, None, :], gamma=gamma[:, None, :]).squeeze()
                
                # Loss
                loss = self.loss_fn(out, logab)

                if (batch_idx == 0):
                    loss_avg = loss.item()
                else:
                    loss_avg = self.smooth * loss.item() + (1.0 - self.smooth) * loss_avg
                
                t.set_postfix(loss=loss_avg)

                self.loss_val.append(loss_avg)
            
        return loss_avg

if (__name__ == '__main__'):

    hyperparameters = {
        'batch_size': 1024,
        'validation_split': 0.1,
        'gpu': 1,
        'lr': 3e-4,
        'wd': 0.0,
        'n_epochs': 100,
        'smooth': 0.15,
        'save_all_epochs': False,
        'embedding': {
            'sigma': 2.0,
            'encoding_size': 512
        },
        'mlp': {
            'n_hidden_mlp': 512,
            'num_layers_mlp': 8
        },
        'condition_pars': {
            'n_hidden': 512,
            'num_layers': 3
        },
        'condition_mol': {
            'n_hidden': 512,
            'num_layers': 3
        },
        'alpha_initial_iteration': 0.0,
        'alpha_final_iteration': 25000.0        
    }
    
    deepnet = Training(hyperparameters)
    deepnet.init_optimize()
    deepnet.optimize()
