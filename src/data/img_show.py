import os
import sys
import cv2
import numpy
from argparse import ArgumentParser




# 显示文件夹中的color gt，按任意键下一张
def show_color_gt(gt_path):
    for filename in os.listdir(gt_path):
        token = filename.split('.')[0]
        if '_c' in token:
            print(filename)
            color_gt = cv2.imread(gt_path + '/' + filename)
            cv2.imshow("color gt", color_gt)
            cv2.waitKey(30)
            if cv2.waitKey(30) == ord('q'):
                cv2.destroyAllWindows()
                sys.exit(0)
        


def main():
    parser = ArgumentParser()
    parser.add_argument('--function', type=str, default='color_gt',
                        choices=['color_gt', 'results'], help='funtion select')
    parser.add_argument('--token', type=str, default=None,
                        help='show the original img and color gt along with the result')
    args = parser.parse_args()

    gt_path = '/home/gs/workspace/datasets/nuscenes_processed/map-labels-v1.2'
    if args.function == 'color_gt':
        show_color_gt(gt_path)

if __name__ == '__main__':
    main()