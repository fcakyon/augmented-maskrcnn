import torch

configurations = {
    1: dict(
        # Random seed to reproduce the results:
        SEED=999,
        # Path to dataset root
        DATA_ROOT="path_to_images",
        # Path to annotation file (should be in coco object detection format)
        COCO_PATH="path_to_coco_json_file",
        # Training model checkpoints will be saved as this name:
        EXPERIMENT_NAME='exp1',
        # Path to the model to be used during predict
        MODEL_PATH="experiments/exp1/maskrcnn-best.pt",
        # Optimizer ("adam" or "sgd")
        OPTIMIZER="sgd",
        # Optimizer learning rate
        LEARNING_RATE=0.0001,
        # Number of iterations to print
        PRINT_FREQ=40,
        # Batch size:
        BATCH_SIZE=1,
        # Total epoch number:
        NUM_EPOCH=5,
        # Don't touch this:
        DEVICE=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        # Thread number to be used for dataloading:
        NUM_WORKERS=0,
    ),
}
