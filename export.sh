export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
INPUT_TYPE=image_tensor
PIPELINE_CONFIG_PATH=/home/hz/hz/models/research/object_detection/models/WIDERFACE/pipeline.config
TRAINED_CKPT_PREFIX=/home/hz/hz/models/research/object_detection/models/WIDERFACE/model.ckpt-200000
EXPORT_DIR=/home/hz/hz/models/research/object_detection/mobilenetv2-ssd
python object_detection/export_inference_graph.py \
    --input_type=${INPUT_TYPE} \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --trained_checkpoint_prefix=${TRAINED_CKPT_PREFIX} \
    --output_directory=${EXPORT_DIR}
