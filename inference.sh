# From models/research/
SPLIT=val  # or test
TF_RECORD_FILES=$(ls -1 object_detection/data/widerface/${SPLIT}/* | tr '\n' ',')||(echo 'Can not find tf-records!' && exit 1)

MODLE=mobilenetv2-ssd
MODLEPATH=object_detection/$MODLE/
mkdir $MODLEPATH/evaluation||(echo 'Can not make dir!' && exit 1)

PYTHONPATH=$PYTHONPATH:$(readlink -f ..) \
python -m object_detection/inference/infer_detections \
  --input_tfrecord_paths=$TF_RECORD_FILES \
  --output_tfrecord_path=$MODLEPATH/evaluation/${SPLIT}_detections.tfrecord-00000-of-00001 \
  --inference_graph=$MODLEPATH/frozen_inference_graph.pb \
  --discard_image_pixels


# From models/research/
NUM_SHARDS=1  # Set to NUM_GPUS if using the parallel evaluation script above

mkdir -p ${SPLIT}_eval_metrics

echo "
label_map_path: '../object_detection/data/oid_bbox_trainable_label_map.pbtxt'
tf_record_input_reader: { input_path: '${SPLIT}_detections.tfrecord@${NUM_SHARDS}' }
" > ${SPLIT}_eval_metrics/${SPLIT}_input_config.pbtxt

echo "
metrics_set: 'open_images_V2_detection_metrics'
" > ${SPLIT}_eval_metrics/${SPLIT}_eval_config.pbtxt


# From tensorflow/models/research/oid
SPLIT=validation  # or test

PYTHONPATH=$PYTHONPATH:$(readlink -f ..) \
python -m object_detection/metrics/offline_eval_map_corloc \
  --eval_dir=${SPLIT}_eval_metrics \
  --eval_config_path=${SPLIT}_eval_metrics/${SPLIT}_eval_config.pbtxt \
  --input_config_path=${SPLIT}_eval_metrics/${SPLIT}_input_config.pbtxt
