import torch

configurations = {
    1: dict(
        # Random seed to reproduce the results:
        SEED=999,

        # Path to dataset root
        DATA_ROOT="C:/Users/FCA/Documents/Projects/Github/midv500-to-coco/data",

        # Path to annotation file (should be in coco object detection format)
        COCO_PATH="C:/Users/FCA/Documents/Projects/Github/midv500-to-coco/coco/midv500_coco.json",

        # The root to buffer best/last model weights during/after training
        ARTIFACT_DIR='artifacts/',

        # Training model checkpoints will be saved as this name:
        EXPERIMENT_NAME='maskrcnn',

        # Path to the model to be used during predict
        MODEL_PATH="artifacts/maskrcnn-best.pt",

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
