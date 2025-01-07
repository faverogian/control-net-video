export SLURM=0                          # (int) Whether to use SLURM for training   
export PROJECT_ROOT=""                  # (str) Path to the project root           
export EXPERIMENT_DIR=""                # (str) Path to the experiment directory
export DATA_PATH=""                     # (str) Path to the data directory

export IMAGE_SIZE=32                    # (int) Size of the input images          
export IMAGE_CHANNELS=15                # (int) Number of channels in the input images
export BATCH_SIZE=16                    # (int) Batch size for training
export NUM_EPOCHS=500                   # (int) Number of epochs to train for
export GRADIENT_ACCUMULATION_STEPS=1    # (int) Number of gradient accumulation steps
export LEARNING_RATE=0.0001             # (float) Learning rate
export LR_WARMUP_STEPS=250              # (int) Number of warmup steps for the learning rate
export SAVE_IMAGE_EPOCHS=50             # (int) Number of epochs between saving images

export PRED_PARAM="v"                   # (str) Diffusion parameterization ('v' or 'eps')
export SCHEDULE="cosine"                # (str) Learning rate schedule ('cosine', 'shifted_cosine')
export NOISE_D=64                       # (int) Reference noise dimensionality (simple diffusion, Hoogeboom et al. 2023)
export MIXED_PRECISION="fp16"           # (str) Mixed precision training ('fp16' or 'bf16' or 'no')
export NUM_WORKERS=24                   # (int) Number of workers for the data loader

export CONDITIONING_CHANNELS=3          # (int) Number of channels in the conditioning tensor
export CLASSES=3                        # (int) Number of classes in the dataset
export ENCODER_TYPE="nn"                # (str) Type of encoder for the end-to-end model
export CFG_W=3.5                        # (int) Classifier-free guidance scale

export SAMPLING_STEPS=256               # (int) Number of sampling steps for the reverse diffusion process
export INFERENCE_ACC_STEPS=1            # (int) Number of accumulation steps for inference (ie. how many samples to average)
export EVALUATION_BATCHES=1             # (int) Number of batches to evaluate for each epoch

export SEED=42
export USE_COMET=0
export COMET_PROJECT_NAME=""
export COMET_WORKSPACE=""
export COMET_EXPERIMENT_NAME=""
export COMET_API_KEY=""

export RESUME=0

export TRAINING_CONFIG="{
  \"slurm\": $SLURM,
  \"resume\": $RESUME,
  \"project_root\": \"$PROJECT_ROOT\",
  \"experiment_dir\": \"$EXPERIMENT_DIR\",
  \"data_path\": \"$DATA_PATH\",
  \"image_size\": $IMAGE_SIZE,
  \"image_channels\": $IMAGE_CHANNELS,
  \"batch_size\": $BATCH_SIZE,
  \"num_epochs\": $NUM_EPOCHS,
  \"gradient_accumulation_steps\": $GRADIENT_ACCUMULATION_STEPS,
  \"learning_rate\": $LEARNING_RATE,
  \"lr_warmup_steps\": $LR_WARMUP_STEPS,
  \"save_image_epochs\": $SAVE_IMAGE_EPOCHS,
  \"pred_param\": \"$PRED_PARAM\",
  \"schedule\": \"$SCHEDULE\",
  \"noise_d\": $NOISE_D,
  \"mixed_precision\": \"$MIXED_PRECISION\",
  \"num_workers\": $NUM_WORKERS,
  \"conditioning_channels\": $CONDITIONING_CHANNELS,
  \"classes\": $CLASSES,
  \"encoder_type\": \"$ENCODER_TYPE\",
  \"cfg_w\": $CFG_W,
  \"sampling_steps\": $SAMPLING_STEPS,
  \"inference_acc_steps\": $INFERENCE_ACC_STEPS,
  \"evaluation_batches\": $EVALUATION_BATCHES,
  \"seed\": $SEED,
  \"use_comet\": $USE_COMET,
  \"comet_project_name\": \"$COMET_PROJECT_NAME\",
  \"comet_workspace\": \"$COMET_WORKSPACE\",
  \"comet_experiment_name\": \"$COMET_EXPERIMENT_NAME\",
  \"comet_api_key\": \"$COMET_API_KEY\"
}"

# Run the Python script
port=$(shuf -i 1025-65535 -n 1)
accelerate launch --multi-gpu \
                  --main-process-port=$port \
                  --num-machines=1 \
                  --num-processes=4 \
                  --mixed_precision='fp16' \
                  --gpu_ids=0,1,2,3 \
                  $PROJECT_ROOT$EXPERIMENT_DIR/train.py