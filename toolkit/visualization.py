import os
import sys
import cv2
import time
import argparse

sys.path.append(os.getcwd())
import scripts.utils.file_utils as utf
import scripts.utils.event_utils as ute

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="a simple event visualization tool")
    parser.add_argument('-i', '--input_path', type=str,
                        default='./datasets/demo/samples/demo-01.aedat4')
    parser.add_argument('-t', '--dt', type=float, default=1E4,
                        help='determine temporal window size to view /ms')
    parser.add_argument('-sz', '--size', type=list, default=[346, 260],
                        help='determine spatial window size to view /pixel')
    parser.add_argument('--flag', action='store_true', help='whether to write .png files')
    args = parser.parse_args()

    print(os.getcwd())

    ev, fr, *_ = utf.load_file(args.input_path)

    for i, packet in enumerate(ute.pack_along_timestamp(ev, size=args.size, duration=args.dt)):
        # if i < 1000: continue
        t = i * args.dt
        img = ute.projection_image(packet, args.size)
        cv2.imshow('image', img)
        cv2.waitKey(1)
        # time.sleep(0.5)
