# Let's have a cell with global hyperparameters for the CNNs in this notebook
# Path to a directory with image dataset and subfolders for training, validation and final testing
DATA_PATH = r"datasets"

# Number of threads for data loader
NUM_WORKERS = 4

# Image size: even though image sizes are bigger than 64, we use this to speed up training
SIZE_H = SIZE_W = 96

# Number of classes in the dataset
NUM_CLASSES = 2

# Epochs: number of passes over the training data, we use it this small to reduce training babysitting time
EPOCH_NUM = 1

# Batch size: for batch gradient descent optimization, usually selected as 2**K elements
BATCH_SIZE = 256

# Images mean and std channelwise
image_mean = [0.485, 0.456, 0.406]
image_std = [0.229, 0.224, 0.225]

# Last layer (embeddings) size for CNN models
EMBEDDING_SIZE = 128

# model name
ckpt_name = "model_base.ckpt"
