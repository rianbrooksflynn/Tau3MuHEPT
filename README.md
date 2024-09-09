# Tau3MuHEPT
This repo is to reproduce the results of Tau3MuHEPT project. Maintained by Benjamin Simon (Purdue).


## 1. Clone the Repo 
Then, clone the repo:
```
git clone git@github.com:simon71701/Tau3MuHEPT.git
cd Tau3MuHEPT
```

## 2. install anaconda
Run:
```
cd ../Tau3MuHEPT
wget https://repo.anaconda.com/archive/Anaconda3-2021.11-Linux-x86_64.sh
sh Anaconda3-2021.11-Linux-x86_64.sh
```
It will ask where to install, rememeber install it at `../Tau3MuHEPT`. It may take a while to install. Type `yes` for all options. After installation, activate `conda` command by `source ~/.bashrc` everythime logging the server.

Now we should be at the `base` environment. Type `python`, we shall see a version of `Python 3.9.7`. Then type `quit()` to quit python, and start installing packages.

## 3. create conda environment and install packages

First, create the conda environment from the provided .yml file and activate it.
```
conda env create -f Tau3MuHEPT/tau3mu_hept.yml
conda activate tau3mu_hept
```

## 4. Get the data
To run the code, you will need datasets in the form of '.pkl' files. Request those datasets and put them in the right place.

When running the code, those dataframes will be processed according to the specific setting, and the processed files will be saved under `$ProjectDir/data/processed-[setting]-[cut_id]`. In this project, for simplicity we call them `SignalPU0, SignalPU200, BkgPU200` as `pos0, pos200, neg200` respectively.

Please note that the processed files may take up lots of disk space (5 gigabytes+), and when processing them it may also take up lots of memory (10 gigabytes+).

# Train a model

We provide example settings for training HEPT models, and the corresponding configurations can be found in `$ProjectDir/src/configs/`. To train an event-classification model with a specific setting, one can do:

```
python src/train_contrastive.py --setting [setting_name] --cuda [GPU_id] --cut [cut_id]
```

`GPU_id` is the id of the GPU to use. To use CPU, please set it to `-1`. `cut_id` is the id of the cut to use. Its default value is `None` and can be set to `cut1` or `cut1+2`. Note that when some cut is used, the `pos_neg_ratio` may need to be adjusted because many positive samples will be dropped.


One thing to notice is that if you have had processed files for a specific setting, even then you change some data-options in the config file, the processed files will not be changed in the next run. So, if you want to change the data-options in a config file, you need to delete the corresponding processed files first. This is because the code will search `.pt` files given the `setting_name`; if it finds any `.pt` files under `$ProjectDir/data/processed-[setting_name]-[cut_id]`, it will assume that the processed files for the specified setting are already there and will not re-process data with the new options. If you've only changed options under 'model' or 'optimizer', you do you need to delete the processed files.

To train a binary hit clustering model with a specific setting, one can do:

```
python src/train_cluster.py --setting [setting_name] --cuda [GPU_id] --cut [cut_id]
```

Similarly, for a multiclass hit clustering model:

```
python src/train_multicluster.py --setting [setting_name] --cuda [GPU_id] --cut [cut_id]
```

# Workflow of the code

1. Class `Tau3MuDataset` in the dataset files in `$ProjectDir/src/utils/` is used to build datasets that can be used to train pytorch models. The code will first call this class to process dataframes, including graph building, node/edge feature generations, dataset splits, etc. After this process, the fully processed data shall be saved on the disk.

2. Then the model will be trained by the class `Tau3MuGNNs` in `train_contrastive.py`, and during the training some metrics will show on the progress bar.

3. The trained model will be saved into `$ProjectDir/data/[log]/[time_step-setting_name-cut_id]/model.pt`, where `[time_step-setting_name-cut_id]` is the log id for this model and will be needed to load the model later and `[log]` is specified in the config file.


# Training Logs
Standard output provides basic training logs, while more detailed logs and interpretation visualizations can be found on tensorboard:
```
python -m tensorboard.main serve --logdir=$LogDir
```

# Constructing a Model outisde of the Workflow

To construct a HEPT model from a given config, you can follow this workflow in a python script:

```
from models.model import get_model
from models import Decoder

config = yaml.safe_load(Path(f'configs/[config_name].yml').open('r'))
data_loaders, x_dim, dataset = get_data_loaders('[config_name]'', config['data'], 1, endcap=0)
model = get_model(config['model_kwargs'],dataset)
decoder = Decoder(config['model_kwargs']['out_dim'])
```

## Loading pre-trained weights
To load a pre-trained model from a specific log:

```
setting = '[config_name]'
log = '[log_name]'

config = yaml.safe_load(Path(f'configs/[config_name].yml').open('r'))
data_loaders, x_dim, dataset = get_data_loaders('[config_name]'', config['data'], 1, endcap=0)

model = get_model(config['model_kwargs'],dataset)  
decoder = Decoder(config['model_kwargs']['out_dim'])

with open(f'[log]/decoder.pt', 'rb') as handle:
    decoder_state_dict = torch.load(handle,map_location=torch.device('cpu'))['model_state_dict']

with open(f'[log]/model.pt', 'rb') as handle:
    model_state_dict = torch.load(handle,map_location=torch.device('cpu'))['model_state_dict']

decoder.load_state_dict(decoder_state_dict)
model.load_state_dict(model_state_dict)

```






