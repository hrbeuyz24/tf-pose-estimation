import argparse
import logging
import sys
import time

from tf_pose import common
import cv2
import numpy as np
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

logger = logging.getLogger('TfPoseEstimatorRun')
logger.handlers.clear()
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


def read_from_video(video_path):
    images = []
    video = cv2.VideoCapture(args.video)
    while (True):
        ret, frame = video.read()
        if ret is False:
            break
        images.append(frame) 
    return images


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tf-pose-estimation run')
    parser.add_argument('--image', type=str, default='./images/p1.jpg')
    parser.add_argument('--model', type=str, default='cmu',
                        help='cmu / mobilenet_thin / mobilenet_v2_large / mobilenet_v2_small')
    parser.add_argument('--resize', type=str, default='0x0',
                        help='if provided, resize images before they are processed. '
                             'default=0x0, Recommends : 432x368 or 656x368 or 1312x736 ')
    parser.add_argument('--resize-out-ratio', type=float, default=4.0,
                        help='if provided, resize heatmaps before they are post-processed. default=1.0')
    parser.add_argument('--video', type=str, default='./video/v1.mp4')
    parser.add_argument('--fps', type=int, default=8)
    parser.add_argument('--result', type=str, default='./result/result.avi')


    args = parser.parse_args()

    w, h = model_wh(args.resize)
    if w == 0 or h == 0:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(432, 368))
    else:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))
    
    images = read_from_video(args.video) 
     
    start_time = time.time() 
    frame_count = 0
    result_image = []
    #videoWriter = cv2.VideoWriter('./result.mp4', cv2.VideoWriter_fourcc('I','4','2','0'), 30, (640, 480))
    for image in images:
         size = image.shape
         humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)
         frame_count = frame_count + 1 
         t = time.time() - start_time
         logger.info('current fps : %.4f'%(frame_count / t))
         image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
         result_image.append(image)

    sz = (size[1], size[0])
    print(args.fps)
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    path = args.result
    videoWriter = cv2.VideoWriter(path,fourcc, args.fps, sz)
    for image in result_image:
        videoWriter.write(image) 
        # cv2.imwrite('./result/result.jpg', image)
    videoWriter.release()
 
    #image = common.read_imgfile(args.image, None, None)
     
    #if image is None:
    #    logger.error('Image can not be read, path=%s' % args.image)
    #    sys.exit(-1)

    #t = time.time()
    #humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)
    #elapsed = time.time() - t

    #logger.info('inference image: %s in %.4f seconds.' % (args.image, elapsed))

    #image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)

    #cv2.imwrite('result.jpg', image)
