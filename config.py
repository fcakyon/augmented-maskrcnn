configurations = {
    1: dict(
        # Random seed to reproduce the results:
        SEED=999,
        # Path to dataset root
        DATA_ROOT="dir_to_images",
        # Path to annotation file (should be in coco object detection format)
        COCO_PATH="path_to_coco_json_file",
        # Training model checkpoints and losses will be saved in the folder experiments/EXPERIMENT_NAME/:
        EXPERIMENT_NAME='exp1',
        # Number of trainable (not frozen) resnet layers starting from final block (btw 0-5)
        TRAINABLE_BACKBONE_LAYERS=3,
        # Path to the model to be used during predict
        MODEL_PATH="experiments/exp1/maskrcnn-best.pt",
        # Optimizer ("adam" or "sgd")
        OPTIMIZER="sgd",
        # Optimizer learning rate
        LEARNING_RATE=0.0001,
        # Optimizer weight decay
        WEIGHT_DECAY=0.0005,
        # Number of iterations to print/log
        PRINT_FREQ=40,
        # Train split rate of the dataset
        TRAIN_SPLIT_RATE=0.8,
        # Batch size:
        BATCH_SIZE=1,
        # Total epoch number:
        NUM_EPOCH=10,
        # Device to perform train/inference (cuda:0, cpu etc.):
        DEVICE="cuda:0",
        # Thread number to be used for dataloading:
        NUM_WORKERS=0,
    ),
}
