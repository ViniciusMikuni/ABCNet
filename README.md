# ABCNet: An attention-based method for particle tagging.

This is the main repository for the [ABCNet paper](https://arxiv.org/abs/2001.05311).
The implementation uses a modified version of [GAPNet](https://arxiv.org/abs/1905.08705) to suit the High Energy Physics needs.
This repository is divided into two main folders: classification and segmentation, for the quark-gluon tagging and pileup mitigation applications, respectively. 
The input ```.h5``` files are expected to have the following structure:

* **data**: [N,P,F], 
* **label**:[N,P]
* **pid**: [N]
* **global**: [N,G]

N = Number of events

F = Number of features per point

P = Number of points

G = Number of global features

For **classification**, only the **pid** is required, while for segmentation only **label** is required.

The files to be used for the training (```train_files.txt```), test (```test_files.txt```) and evaluation (```evaluate_files.txt```) are required to be listed in the respective text files. 

# Requirements

[Tensorflow](https://www.tensorflow.org/)

[h5py](https://www.h5py.org/)

# Classification

To train use:

```bash
cd classification
python train.py  --data_dir ../data/QG/  --log_dir qg_test
```

A ```logs``` folder will be created with the training results under the main directory.
To evaluate the training use:
```bash
python evaluate.py  --data_dir ../data/QG --model_path ../logs/qg_test --batch 500 --name qg_test --modeln 1
```

# Segmentation

To train use:

```bash
cd segmentation
python train.py  --data_dir ../data/PU/  --log_dir pu_test
```

To evaluate the training use:
```bash
python evaluate.py  --data_dir ../data/PU --model_path ../logs/ou_test --batch 500 --name pu_test 
```



# License

MIT License

# Acknowledgements
ABCNet uses a modified version of [GAPNet](https://arxiv.org/abs/1905.08705) and [PointNet](https://github.com/charlesq34/pointnet).