import scipy.io as sio
import os
import io

import PIL.Image
import cv2
import tensorflow as tf

from object_detection.utils import dataset_util

import contextlib2
from object_detection.dataset_tools import tf_record_creation_util

num_shards=10

flags = tf.app.flags
flags.DEFINE_string('output_path', 'object_detection/data/test.record', 'Path to output TFRecord')
# flags.DEFINE_string('output_path', 'object_detection/data/WIDERFACE/widerface_train.record', 'Path to output TFRecord')
flags.DEFINE_string('input_path', '/home/hz/hz/WIDERFACE/wider_face_split/wider_face_train.mat', 'Path to input widerface')
flags.DEFINE_string('dataset_root', '/home/hz/hz/WIDERFACE/WIDER_train/images/', 'Root to widerface events image')
FLAGS = flags.FLAGS

class Example():
  def __init__(self, dataset_root, event, file_name, face_bbxes):
    self.set_image(dataset_root, event, file_name)
    self.set_bbx(face_bbxes)

  def set_image(self, dataset_root, event, file_name):
    self.file_name = file_name.encode('utf8')

    image_path = os.path.join(dataset_root,event,file_name) + '.jpg'
    print image_path
    cv_image = cv2.imread(image_path)
    self.image = cv_image
    self.h = cv_image.shape[0]
    self.w = cv_image.shape[1]

    with tf.gfile.GFile(image_path, 'rb') as fid:
      self.encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(self.encoded_jpg)
    image = PIL.Image.open(encoded_jpg_io)
    if image.format != 'JPEG':
      raise ValueError('Image format not JPEG')

  def set_bbx(self, face_bbxes):
    self.xmins = []
    self.xmaxs = []
    self.ymins = []
    self.ymaxs = []
    self.classes_text = []
    self.classes = []

    for face in face_bbxes:
      xmin = face[0]
      ymin = face[1]
      w = face[2]
      h = face[3]

      xmax = xmin + w
      ymax = ymin + h
      if w <= 0:
        if w == 0:
          w = 1
        else:
          w = -w
          xmin, xmax = xmax, xmin
      if h <= 0:
        if h == 0:
          h = 1
        else:
          h = -h
          ymin, ymax = ymax, ymin

      self.xmins.append( xmin/float(self.w) )
      self.xmaxs.append( (xmin+w)/float(self.w) )
      self.ymins.append( ymin/float(self.h) )
      self.ymaxs.append( (ymin+h)/float(self.h) )
      self.classes_text.append('face'.encode('utf8'))
      self.classes.append(1)

    #   cv2.rectangle(self.image, (xmin,ymin), (xmax,ymax),(255,0,0))
    # print(self.w,self.h)
    # print(self.xmins, self.xmaxs, self.ymins, self.ymaxs)
    # cv2.imwrite('img.jpg',self.image)
    # cv2.waitKey(0)


def create_tf_example(example):
  print(example.w,example.h, example.file_name)
  
  
  # TODO(user): Populate the following variables from your example.
  height = example.h # Image height
  width = example.w # Image width
  filename = example.file_name # Filename of the image. Empty if image is not from file
  encoded_image_data = example.encoded_jpg # Encoded image bytes
  image_format = b'jpeg' # b'jpeg' or b'png'

  xmins = example.xmins # List of normalized left x coordinates in bounding box (1 per box)
  xmaxs = example.xmaxs # List of normalized right x coordinates in bounding box
             # (1 per box)
  ymins = example.ymins # List of normalized top y coordinates in bounding box (1 per box)
  ymaxs = example.ymaxs # List of normalized bottom y coordinates in bounding box
             # (1 per box)
  classes_text = example.classes_text # List of string class name of bounding box (1 per box)
  classes = example.classes # List of integer class id of bounding box (1 per box)

  tf_example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/filename': dataset_util.bytes_feature(filename),
      'image/source_id': dataset_util.bytes_feature(filename),
      'image/encoded': dataset_util.bytes_feature(encoded_image_data),
      'image/format': dataset_util.bytes_feature(image_format),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
      'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
      'image/object/class/label': dataset_util.int64_list_feature(classes),
  }))
  return tf_example


def main(_):
  #writer = tf.python_io.TFRecordWriter(FLAGS.output_path)

  # dataset_root = u'/home/hz/hz/WIDERFACE/WIDER_train/images/'
  dataset_root = FLAGS.dataset_root
  matfile = sio.loadmat(FLAGS.input_path)
  # matfile  = sio.loadmat('/home/hz/hz/WIDERFACE/wider_face_split/wider_face_train.mat')
  event_list = matfile['event_list']
  file_list = matfile['file_list']
  face_bbox_list = matfile['face_bbx_list']

  no_object_files = []
  with contextlib2.ExitStack() as tf_record_close_stack:
    output_tfrecords = tf_record_creation_util.open_sharded_output_tfrecords(
        tf_record_close_stack, FLAGS.output_path, num_shards)

    index = 0
    for event, event_file_list, event_face_bbx_list in zip(event_list,file_list,face_bbox_list):
      # print(event_file_list[0].shape, event_face_bbx_list[0].shape)
      for file_list, face_bbx_list in zip(event_file_list[0], event_face_bbx_list[0]):
        
        file_name = file_list[0]
        face_bbxes = face_bbx_list
        # print(event[0], file_name[0], face_bbxes[0])
        if face_bbxes.shape[0] <= 0:
          no_object_files.append(file_name)
          continue

        example = Example(dataset_root, event[0][0], file_name[0], face_bbxes[0])

        # tf_example = create_tf_example(example)
        # output_shard_index = index % num_shards
        # output_tfrecords[output_shard_index].write(tf_example.SerializeToString())
        index += 1
        print index
  print no_object_files

if __name__ == '__main__':
  tf.app.run()