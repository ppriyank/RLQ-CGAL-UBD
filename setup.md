## setup 

After `Git pull` do `mkdir outputs`

### anaconda 
```
conda create --name rql python=3.8
conda activate rql
python3 -m pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117  

OR 

conda create --name rql python=3.10  
conda activate rql   
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121  
```

### Rest of libraries 
```
python -m pip install timm==0.9.7 matplotlib tensorboardX Ninja decord gdown termcolor
python -m pip install scikit-learn tabulate tensorboard lmdb yacs pandas einops 
python -m pip install albumentations h5py scipy torchcontrib

python -m pip install openmim
mim install 'mmcv==2.0.0'
mim install 'mmengine'
mim install 'mmagic'

python -m pip install  mmengine 'mmcv==2.0.0' mmagic
python -m pip install pytorch-msssim jpeg4py transformers==4.30.0
pip install transformers --upgrade
pip install diffusers==0.24.0 fastreid insightface onnxruntime
```

### timm dependency issues 
One of these paths : 
```
vim ~/anaconda3/envs/rql/lib/python3.8/site-packages/torch/_six.py
vim ~/.conda/envs/rql/lib/python3.6/site-packages/timm/models/layers/helpers.py
vim ~/.local/lib/python3.7/site-packages/timm/models/layers/helpers.py
vim ~/anaconda3/envs/rql/lib/python3.6/site-packages/timm/models/layers/helpers.py
vim ~/my-envs/rql/lib/python3.8/site-packages/timm/models/layers/helpers.py
```

`Comment` & `Reaplce`
```
from torch._six import
import collections.abc as container_abcs
```
