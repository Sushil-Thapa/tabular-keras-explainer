import os

SEED_VALUE = 42
EPOCHS = 10
BATCH_SIZE = 32

BASE_DIR = os.path.dirname(__file__)
model_checkpoint_path = os.path.join(BASE_DIR, "out","checkpoints","cp.ckpt")
lime_out_path = os.path.join(BASE_DIR,"out","lime")

