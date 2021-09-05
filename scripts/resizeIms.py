import argparse
import os
import sys

import torchvision.transforms.functional as F

from PIL import Image

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--resize_dim', type=int, default=161)
    parser.add_argument('--quality', type=int, default=-1)
    parser.add_argument('--ext', default=None)
    args = parser.parse_args()

    assert args.ext in [None, 'jpg', 'png']

    image_dir = args.input_dir
    destination_dir = args.output_dir
    resize_dim = args.resize_dim

    os.makedirs(args.output_dir, exist_ok=True)

    label_dirs = os.listdir(image_dir)
    for i, d in enumerate(label_dirs):
        print(i, d)
        if os.path.isfile(os.path.join(image_dir, d)):
            continue
        if os.path.join(image_dir, d) == destination_dir:
            continue
        files = os.listdir(os.path.join(image_dir, d))
        os.makedirs(os.path.join(destination_dir, d), exist_ok=True)
        for f in files:
            im = Image.open(os.path.join(image_dir, d, f))
            im = im.convert('RGB')
            im = F.resize(im, resize_dim)

            if args.ext is not None:
                f = os.path.splitext(f)[0] + '.' + args.ext

            out_fname = os.path.join(destination_dir, d, f)
            if args.ext == 'jpg' and args.quality > 0:
                im.save(out_fname, quality=args.quality, optimize=True)
            else:
                im.save(out_fname, optimize=True)

if __name__ == '__main__':
    main()
