# Simple Distance Regression

Given a two vectors as input (position and direction) an ML algorithm regresses the distance to the boundary of simple geometries. At the moment two geometries have implemented:
1. 2D box
2. 3D cube

For both cases, one should first generate a dataset and then use it to train the MLP algorithm.

## Getting Started

First clone the code and setup the required environment to run on.

### Code

```bash
git clone git@xgitlab.cels.anl.gov:whopkins/distanceRegression.git -b VangelisDevs
cd distanceRegression/VangelisCode
```

### Environment

The environment configuration instructions are tailored for the BNL IC machines (interactive or batch sessions).

```bash
source /direct/u0b/software/jupyter/anaconda3/bin/activate
conda activate /usatlas/u/ekourlit/.conda/envs/tf_gpu
```

## Dataset Generation

Generate the dataset using the scripts `prepare<N>Ddata.py`, where `<N>` is either 2 or 3, corresponding to the two cases listed above. The dataset is by default saved to a `.pickle` file. The code is self-explanatory, check description and arguments with `-h`.

### Example
```bash
python prepare3Ddata.py --saveFile 
```
The above command saves a file called `3Ddata.pickle` with structure:
```
             i         X         Y         Z    Xprime    Yprime    Zprime         L
0            0  0.900101  0.324281  0.274795 -0.441440  0.342839  0.829212  0.874571
1            1  0.487065  0.808866  0.132964 -0.489615 -0.361369  0.793530  0.994792
2            2  0.971668  0.089752  0.754579 -0.020757  0.967323  0.252698  0.940997
...        ...       ...       ...       ...       ...       ...       ...       ...
```

## MLP Regression

Use the dataset to train an MLP regression algorithm. For the 3D cube case and the example dataset generated above:
```bash
python -i 3DCubeRegression.py --dataFile 3Ddata.pickle
```
A lot of the MLP configuration can be set at the `settings` dictionary inside the script. For example:
```python
settings = {
		'Structure' 			: 	[10, 50, 10, 6],
		'Optimizer' 			: 	'adam',
		'Activation' 			: 	'relu',
		'Batch'				:	100,
		'Epochs'			:	3,
		'Training Sample Size'		:	int(trainX.shape[0]*(1-validation_split))
}
```
After the model is trained, predictions for a validation dataset (portion of the input dataset from the `.pickle` file) are stored into the `predY` array, which can be easily manipulated in an interactive `python` session. The corresponding truth values array can be acquired by `valY.numpy()`.  
By adding the argument `--plots` at the script execution few MLP performance evaluation plots are also saved.
