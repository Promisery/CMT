# Customizable Multimodal Trajectory Prediction via Nodes of Interest Selection for Autonomous Vehicles

This repository is the official implementation of the paper [**Customizable Multimodal Trajectory Prediction via Nodes of Interest Selection for Autonomous Vehicles**](https://doi.org/10.1016/j.eswa.2025.128222).

**Note:** For research purpose, this repository releases a simplified version of the CMT model on the [nuScenes dataset](https://www.nuscenes.org/download). The simplified model consumes approximately 4GB GPU memory for batch size=64, which is about 1/5 of the original model. Although the performance of the simplified version slightly drops, state-of-the-art performance can still be obtained through a simple model ensemble.

## Installation

1. Clone this repository 

2. Set up a new conda environment 
``` shell
conda create --name cmt python=3.9
```

3. Install dependencies
```shell
conda activate cmt

# nuScenes devkit
pip install nuscenes-devkit

# Pytorch: The code has been tested with Pytorch 1.12.1, CUDA 11.6, but should work with newer versions
pip install torch torchvision

# Additional utilities
pip install ray psutil imageio tensorboard numpy matplotlib scipy einops scikit-learn tqdm entmax timm
```


## Dataset

1. Download the [nuScenes dataset](https://www.nuscenes.org/download). For this project we just need the following.
    - Metadata for the Trainval split (v1.0)
    - Map expansion pack (v1.3)

2. Organize the nuScenes root directory as follows
```plain
└── nuScenes/
    ├── maps/
    |   ├── basemaps/
    |   ├── expansion/
    |   ├── prediction/
    |   ├── 36092f0b03a857c6a3403e25b4b7aab3.png
    |   ├── 37819e65e09e5547b8a3ceaefba56bb2.png
    |   ├── 53992ee3023e5494b90c316c183be829.png
    |   └── 93406b464a165eaba6d9de76ca09f5da.png
    └── v1.0-trainval
        ├── attribute.json
        ├── calibrated_sensor.json
        ...
        └── visibility.json         
```

3. Run the following script to extract pre-processed data. This speeds up training significantly.
```shell
python preprocess.py -c configs/preprocess_nuscenes.yml -r path/to/nuScenes/root/directory -d path/to/directory/with/preprocessed/data
```

## Pretrained Models

We trained 3 models with different seeds (2022, 2023 and 2024). The pretrained models are located at `./checkpoints`. To evaluate the performance of each model, run
```shell
python ./evaluate.py -c configs/cmt.yml -r path/to/nuScenes/root/directory -d path/to/directory/with/preprocessed/data -o logs/2022 -w checkpoints/2022.tar -s 2022
python ./evaluate.py -c configs/cmt.yml -r path/to/nuScenes/root/directory -d path/to/directory/with/preprocessed/data -o logs/2023 -w checkpoints/2023.tar -s 2023
python ./evaluate.py -c configs/cmt.yml -r path/to/nuScenes/root/directory -d path/to/directory/with/preprocessed/data -o logs/2024 -w checkpoints/2024.tar -s 2024
```

To ensemble the pretrained models, run
```shell
python ./ensemble.py -c configs/cmt.yml configs/cmt.yml configs/cmt.yml -r path/to/nuScenes/root/directory -d path/to/directory/with/preprocessed/data -o logs/ensemble -w checkpoints/2022.tar checkpoints/2023.tar checkpoints/2024.tar -s 2024
```

The results can be found at `./logs`. The expected *MinADE_5* metric should be 1.19~1.21 for each model and 1.17 for ensemble.


## Training and Evaluation

To train and evaluate the models from scratch, run
```shell
./run.sh
```

The logs, checkpoints and results are saved at `./logs`. To check the logs, run
```shell
tensorboard --logdir ./logs
```

## Citation

```
@article{CMT,
    title = {Customizable Multimodal Trajectory Prediction via Nodes of Interest Selection for Autonomous Vehicles},
    journal = {Expert Systems with Applications},
    pages = {128222},
    year = {2025},
    issn = {0957-4174},
    doi = {10.1016/j.eswa.2025.128222},
    author = {Titong Jiang and Qing Dong and Yuan Ma and Xuewu Ji and Yahui Liu},
}
```


## Acknowledgement

This repository is developed based on the following wonderful projects:

**[PGP](https://github.com/nachiket92/PGP)**

**[x-transformers](https://github.com/lucidrains/x-transformers)**