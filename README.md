# Neural emulator for interstellar chemistry

This repository contains the code for using a conditional neural field
as an emulator for instellar chemistry.

## Installation

It should be enough to do:

    pip install neural_chemistry

and all requirements should be installed.

### Dependencies
In case it fails, note that this repository makes use of PyTorch and other Python packages. You can have them
installed in your system, but we recommend to follow these instructions to
have a working environment with everything needed for running the code.

Install Miniconda in your system. Go to https://docs.conda.io/projects/miniconda/en/latest/ and download the executable file for your system. For instance, for a typical Linux system, you should do:

    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

or 

    curl -lO https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

Install Miniconda

    chmod +x Miniconda3-latest-Linux-x86_64.sh
    ./Miniconda3-latest-Linux-x86_64.sh

and follow the instructions for selecting the directory.

Create an environment and install the packages:

    conda create -n chemistry python=3.10
    conda activate chemistry
    conda install -c conda-forge numpy tqdm 

Now install PyTorch. Go to https://pytorch.org/ and select from the properties of 
your system from the matrix (Linux/Mac/Windows, CPU/GPU, conda/pip). If you have an 
NVIDIA GPU card in your system, you can take advantage of the acceleration that it
provides (around a factor 10 faster). For instance, in a system with an NVIDIA GPU
card with CUDA drivers on version 12.1, you can install PyTorch using:

    conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

or

    pip3 install torch torchvision torchaudio

if using `pip` for the installation of packages.

In a CPU-only system:

    conda install pytorch torchvision torchaudio cpuonly -c pytorch

or

    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

if using `pip` for the installation of packages.

To make then a local installation, simply clone the repository and type

    pip install -e .

## New version

An example of how to run the code is shown in `example2.py`, that we reproduce here:

    # Define the number of random models 
    
    n_models = 32
    batch_size = 32

    mols = [0, 1, 2, 3, 4, 5]

    nh = 10.0**np.random.uniform(np.log10(1e4), np.log10(1e7), size=n_models)
    T = np.random.uniform(10.0, 80.0, size=n_models)
    crir = 10.0**np.random.uniform(np.log10(1e-17), np.log10(1e-15), size=n_models)
    sulfur = 10.0**np.random.uniform(np.log10(7.5e-8), np.log10(1.5e-5), size=n_models)
    uv_flux = 10.0**np.random.uniform(np.log10(0.1), np.log10(1e4), size=n_models)
    Av = np.random.uniform(0.0, 18.0, size=n_models)
    
    t = np.logspace(0, 7, 120)

    net = neural_chemistry.ChemistryEmulator2(gpu=0, verbose=True)
    abundance = net.evaluate(t, T, nh, crir, sulfur, uv_flux, Av, batch_size=batch_size, species=None)

    abundance_subset = net.evaluate(t, T, nh, crir, sulfur, uv_flux, Av, batch_size=batch_size, species=[0,1,2,3,4,5])


We choose 64 models with random properties inside the range of validity. Select the output times and call the emulator. The `batch_size` defines the number of models that will be computed in parallel. This depends on the amount of memory you have but it can be a large number. The keyword `species` indicates a list of the indices of which species you
want to compute, from the list of 250 species that can be found in `list_molecules:20ene24.txt`. If abstent or
set to `None`, all species are computed.

## Old version

An example of how to run the old code is shown in `example.py`, that we reproduce here:

    # Define the number of random models 
    n_models = 64
    batch_size = 32

    nh = 10.0**np.random.uniform(np.log10(1e4), np.log10(1e7), size=n_models)
    T = np.random.uniform(10.0, 80.0, size=n_models)
    crir = 10.0**np.random.uniform(np.log10(1e-17), np.log10(1e-15), size=n_models)
    sulfur = 10.0**np.random.uniform(np.log10(1.5e-5), np.log10(7.5e-8), size=n_models)
    uv_flux = 10.0**np.random.uniform(np.log10(0.1), np.log10(1e4), size=n_models)
    
    t = np.logspace(0, 7, 64)

    net = emulator.ChemistryEmulator(gpu=0, batch_size=32)
    abundance = net.evaluate(t, T, nh, crir, sulfur, uv_flux, batch_size=32, species=None)

## Weights

The weights are downloadad automatically the first time any of the emulators is used.