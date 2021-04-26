import os

SEED_VALUE = 42
EPOCHS = 10
BATCH_SIZE = 32

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
model_checkpoint_path = "out/checkpoints/cp.ckpt"
lime_out_path = os.path.join(BASE_DIR,"out","lime")

