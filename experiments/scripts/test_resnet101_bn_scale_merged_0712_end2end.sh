./tools/test_net.py --gpu 1 \
  --def models/pascal_voc/ResNet101_BN_SCALE_Merged/faster_rcnn_end2end/test.prototxt \
  --net /hdd/data/trained_networks/pascal_voc/ResNet/resnet101_faster_rcnn_bn_scale_merged_end2end_iter_70000.caffemodel \
  --imdb voc_0712_test \
  --cfg experiments/cfgs/faster_rcnn_end2end.yml \
