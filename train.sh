PIPELINE_CONFIG_PATH='/home/hz/hz/models/research/object_detection/samples/WIDERFACE/ssd_mobilenet_v2.config'
MODEL_DIR='/home/hz/hz/models/research/object_detection/models/WIDERFACE'
NUM_TRAIN_STEPS=200000
SAMPLE_1_OF_N_EVAL_EXAMPLES=1
python object_detection/model_main.py \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --model_dir=${MODEL_DIR} \
    --num_train_steps=${NUM_TRAIN_STEPS} \
    --sample_1_of_n_eval_examples=$SAMPLE_1_OF_N_EVAL_EXAMPLES \
    --eval_interval_secs=3600 \
    --logtostderr