from models import getNoOverestimateLossFunction
from tensorflow.keras.optimizers import *
from datetime import date,datetime

# -----------------
# --- Timestamp ---
# -----------------
timestamp = str(datetime.now().year)+str("{:02d}".format(datetime.now().month))+str("{:02d}".format(datetime.now().day))+str("{:02d}".format(datetime.now().hour))+str("{:02d}".format(datetime.now().minute))

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
    'Description'       :   "simple sequential feed-forward network to estimate the distance to the surface of a unit sphere",
    # 'Structure'         :    {'Position' : [512,512,512], 'Direction' : [], 'Concatenated' : [1024,1024]},
    'Timestamp'         :   timestamp,
    'Layers'            :   4,
    'Nodes'             :   400,
    'Activation'        :   'relu',
    'OutputActivation'  :   None,
    'Loss'              :   'getNoOverestimateLossFunction', #getNoOverestimateLossFunction #mae #mse
    'negPunish'         :   5.0, # only for noOverestimateLossFunction
    'Optimizer'         :   'Adam',
    'LearningRate'      :   1e-4,
    'b1'                :   0.9,
    'b2'                :   0.999,
    # 'Amsgrad'           :    false,
    'Batch'             :   4096,
    'Epochs'            :   200
}
# convert the str to function call
dispatcher = {'getNoOverestimateLossFunction':getNoOverestimateLossFunction, 'Adam':Adam}
