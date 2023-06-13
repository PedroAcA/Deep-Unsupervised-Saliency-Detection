#!/usr/bin/env python

#!/usr/bin/env python

from yacs.config import CfgNode as CN


_C = CN()

_C.SYSTEM = CN()

# Number of GPUS to use in the experiment
_C.SYSTEM.NUM_GPUS = 1
# Number of workers for doing things
_C.SYSTEM.NUM_WORKERS = 0
_C.SYSTEM.DEVICE = 'cuda:0'
_C.SYSTEM.CHKPT_FREQ = 5
_C.SYSTEM.LOG_FREQ = 10
_C.SYSTEM.EXP_NAME = 'default'
_C.SYSTEM.EXP_TYPE = 'full'
_C.SYSTEM.SEED = None

# Dataset setting
_C.DATA = CN()
_C.DATA.TRAIN = CN()
_C.DATA.TRAIN.ROOT = '../datasets/MSRA-B/imgs/'
_C.DATA.TRAIN.NOISE_ROOT = '../datasets/MSRA-B/pseudo_labels/full_dataset/'
_C.DATA.TRAIN.LIST = '../datasets/MSRA-B/msra_b_train.lst'
_C.DATA.TRAIN.GT_ROOT = '../datasets/MSRA-B/gt/'
_C.DATA.VAL = CN()
_C.DATA.VAL.ROOT = '../datasets/MSRA-B/imgs/'
_C.DATA.VAL.NOISE_ROOT = '../datasets/MSRA-B/pseudo_labels/full_dataset/'
_C.DATA.VAL.LIST = '../datasets/MSRA-B/msra_b_val.lst'
_C.DATA.VAL.GT_ROOT = '../datasets/MSRA-B/gt/'
_C.DATA.TEST = CN()
_C.DATA.TEST.ROOT = '../datasets/MSRA-B/imgs/'
_C.DATA.TEST.LIST = '../datasets/MSRA-B/msra_b_test.lst'
_C.DATA.TEST.GT_ROOT = '../datasets/MSRA-B/gt/'

# Optimizer
_C.SOLVER = CN()
_C.SOLVER.OPTIMIZER = 'SGD'
_C.SOLVER.SCHEDULER = 'ReduceLROnPlateu'
_C.SOLVER.LAMBDA = 1e-7 #set in a way that total batch loss is in range [0,10) before noise module update and in range [0, 1) after noise module update
_C.SOLVER.EPOCHS = 20
_C.SOLVER.IMG_SIZE = (256, 256)
# Optimizer Settings
_C.SOLVER.BETAS = (0.9, 0.99)
_C.SOLVER.MOMENTUM = 0.9
_C.SOLVER.LR = 1e-3
_C.SOLVER.WEIGHT_DECAY = 0.0
# Scheduler Settings
_C.SOLVER.STEP_SIZE = 10
_C.SOLVER.FACTOR = 0.9
_C.SOLVER.MIN_LR = 1e-12
_C.SOLVER.PAITENCE = 10
_C.SOLVER.THRESHOLD = 1e-6
_C.SOLVER.COOLDOWN = 1
_C.SOLVER.BATCH_SIZE = 2
_C.SOLVER.ITER_SIZE = 1 # how many batches should be accumulated before updating weights
_C.SOLVER.NUM_MAPS = 3
_C.SOLVER.NUM_IMG = 3000

# Noise
_C.NOISE = CN()
_C.NOISE.ALPHA = 0.01

#Model
_C.MODEL = CN()
_C.MODEL.PRE_TRAINED = True
_C.MODEL.TRANSFER_LEARNING = False
# Miscellaneous
_C.SAVE_ROOT = '../experiments/'

cfg = _C  # users can `from config import cfg`
