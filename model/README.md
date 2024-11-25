# Learning Clustering-based Prototypes for Compositional Zero-Shot Learning

## Setup

```bash
conda create --name cluspro python=3.7
conda activate cluspro
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
# pip3 install git+https://github.com/openai/CLIP.git
```

Alternatively, you can use `pip install -r requirements.txt` to install all the dependencies.



## Download Dataset
We experiment with three datasets: MIT-States, UT-Zappos, and C-GQA.

```bash
sh download_data.sh
```

If you already have setup the datasets, you can use symlink and ensure the following paths exist:
`<DATASET_ROOT>/<DATASET>` where `<DATASET> = {'mit-states', 'ut-zappos', 'cgqa'}`.

## Training

```py
python -u train.py \
--dataset <DATASET> \
--dataset_path <DATASET_ROOT>/<DATASET> \
--save_path <SAVE_ROOT>/<DATASET> \
--num_workers 10 \
--seed 0 
```


## Evaluation

We evaluate our models in two settings: closed-world and open-world.

### Closed-World Evaluation

```py
python -u test.py \
--dataset_path <DATASET_ROOT>/<DATASET> \
--save_path <SAVE_ROOT>/<DATASET> \
--num_workers 10 \
--seed 0 \
--load_model <SAVE_ROOT>/<DATASET>/val_best.pt
```

### Open-World Evaluation

For our open-world evaluation, we compute the feasbility calibration and then evaluate on the dataset.


Just run:

```py
python -u test.py \
--dataset_path <DATASET_ROOT>/<DATASET> \
--save_path <SAVE_ROOT>/<DATASET> \
--open_world \
--num_workers 10 \
--seed 0 \
--load_model <SAVE_ROOT>/<DATASET>/val_best.pt
```



