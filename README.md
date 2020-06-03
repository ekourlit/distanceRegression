# Distance Regression

Given a two 3D vectors as input (position and direction) an ML algorithm regresses the distance to the boundary of a geometry.

## Getting Started

First clone the code and setup the required environment to run on.

### Code

```bash
git clone git@xgitlab.cels.anl.gov:whopkins/distanceRegression.git
cd distanceRegression
```

### Environment

The required environment that supports the usage of GPUs in the BNL IC machines can be set up with:

```bash
source /direct/u0b/software/jupyter/anaconda3/bin/activate
conda activate /usatlas/u/ekourlit/.conda/envs/tf_gpu
```

while in the `atlaslogin0<X>.hep.anl.gov` or the `atlasdpb0.hep.anl.gov` machines:
```bash
conda activate /users/ekourlitis/.conda/envs/tf_gpu
```

An optimized for Intel CPUs environment using [Math Kernel Library](https://software.intel.com/content/www/us/en/develop/tools/math-kernel-library.html) can be also set up on ANL machines:
```bash
conda activate /users/ekourlitis/.conda/envs/tf_mkl
```
For the moment, the MKL environment has been only used for limited inference studies.

## Datasets

Datasets to train and validate the regression algorithm can be found at `/users/ekourlitis/atlasfs02/PolyconeBoxData/GeantData/geom_type_1_nsensors_2_dense_1`. These have been generated using the VecGeom example geometry ([info](https://indico.cern.ch/event/773049/contributions/3474755/attachments/1937701/3213528/CHEP19_Adelaide_SWENZEL_final.pdf)). This particular example regards a cube with 24 polycone sensors around it.

More data can be found at: `/data/atlasfs02/a/users/whopkins/polycone_onestep/`.

## DNN Regression

The above datasets can be used to train and validate a DNN regression algorithm. For example:
```bash
python distanceRegression.py --trainData /users/ekourlitis/atlasfs02/PolyconeBoxData/GeantData/geom_type_1_nsensors_2_dense_1/train_merged.csv --validationData /users/ekourlitis/atlasfs02/PolyconeBoxData/GeantData/geom_type_1_nsensors_2_dense_1/validation_6.csv
```
A lot of the MLP configuration can be set at the `settings` dictionary inside the script. For example:
```python
settings = {
		'Structure'				:	[128,128,128],
		'Optimizer'				:	Adam(learning_rate=0.005),
		'Activation'			        :	'relu',
		'OutputActivation'		        :	'relu',
		'Loss'					:	'mse',
		'Batch'					:	128,
		'Epochs'				:	50
}
```
After the model is trained, it is saved into the `data` directory with a timestamp appended to the name, for example, `mlp_model_202005141913`, along with its configuration in `modelConfig_202005141913.txt`. On a later time, the pre-trained model can be loaded by adding the argument `--model` on the above command:
```bash
python distanceRegression.py --validationData /users/ekourlitis/atlasfs02/PolyconeBoxData/GeantData/geom_type_1_nsensors_2_dense_1/validation_6.csv --model data/mlp_model_202005141913
```

Finally, by adding the argument `--plots` at the script execution timestamped performance evaluation plots are also saved into the `plots` directory.
