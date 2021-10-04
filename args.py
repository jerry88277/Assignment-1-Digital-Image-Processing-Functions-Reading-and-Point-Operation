import argparse

def init_arguments():
    parser = argparse.ArgumentParser(prog='DL-hw7: semantic segmentation')

    # General
    parser.add_argument('-img_PATH', '--image_path', default = 'data/')
    parser.add_argument('-save_PATH', '--save_path', default = 'output/')
    parser.add_argument('-mn', '--method_name', type=str, default='NNI')

    return parser