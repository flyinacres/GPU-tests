"# GPU-tests" 

Using a test file based upon https://towardsdatascience.com/rtx-2060-vs-gtx-1080ti-in-deep-learning-gpu-benchmarks-cheapest-rtx-vs-most-expensive-gtx-card-cd47cd9931d2

Attempting to get my outdated GPU (GTX 1080Ti) working with modern code.
Leveraging the fastai libraries and data.

Successfully able to get the code working. The GPU isn't struggling (so far) with the models and data I've used to this point...


Noting the steps I took to get the Anaconda environment set up with my GPU on Windows 11, and working in a Jupyter notebook. Check the 
notebooks themselves to see the changes I made to get them working with these library versions.


> conda --version
conda 23.7.4
> python --version
python 3.11.5

> conda create --name cuda_test python=3.11

(lots of stuff, this is now an environment in the Anaconda Navigator)

> conda activate cuda_test

# Just checking nvidia
> nvidia-smi

> conda install cudatoolkit -c anaconda -y

# Had to install the earliest versions still available to work with my system...
> conda install pytorch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 pytorch-cuda=11.8 -c p
ytorch -c nvidia

> python
>>> import torch
>>> torch.cuda.is_available()
True
>>> import numpy as np
>>> a = np.random.randn(5)
>>> a
array([ 2.50904347,  0.18383574,  2.52352469, -0.44537827, -1.16293558])
>>> torch.from_numpy(a)
tensor([ 2.5090,  0.1838,  2.5235, -0.4454, -1.1629], dtype=torch.float64)
>>> b = torch.from_numpy(a)
>>> b.to("cuda")
tensor([ 2.5090,  0.1838,  2.5235, -0.4454, -1.1629], device='cuda:0',
       dtype=torch.float64)
>>>exit()


To get this to work with Jupyter labs notebooks...

> conda install ipykernel

> python -m ipykernel install --user --name cuda_test --display-name "Python (cuda)"

Now "Python (cuda)" is a selectable version of python when creating a new notebook from the main 
Jupyter webpage.

In a cell: 
> import torch
> torch.cuda.is_available()
True
