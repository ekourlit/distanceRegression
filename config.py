from models import getNoOverestimateLossFunction
from tensorflow.keras.optimizers import *

# --------------------------
# --- set normalisations ---
# --------------------------

# lengthNormalisation = 5543
lengthNormalisation = 1.0
# positionNormalisation = 1600
positionNormalisation = 1.0

# -----------------
# -- NN settings --
# -----------------

settings = {
    'Description'       :    "simple sequential feed-forward network to estimate the distance to the surface of a unit sphere",
    # 'Structure'         :    {'Position' : [512,512,512], 'Direction' : [], 'Concatenated' : [1024,1024]},
    'Layers'            :    4,
    'Nodes'             :    400,
    'Activation'        :    'relu',
    'OutputActivation'  :    'relu',
    'Loss'              :    'getNoOverestimateLossFunction', #getNoOverestimateLossFunction #mae #mse
    'negPunish'         :    8.0, # only for noOverestimateLossFunction
    'Optimizer'         :    'Adam',
    'LearningRate'      :    1e-5,
    'Batch'             :    3000,
    'Epochs'            :    200
}
# convert the str to function call
dispatcher = {'getNoOverestimateLossFunction':getNoOverestimateLossFunction, 'Adam':Adam}