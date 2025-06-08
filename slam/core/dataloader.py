# dataloader.py
import os
import glob
import cv2

def load_sequence(args):
    """
    Return a list `seq` of either on-disk image paths or in-memory BGR frames.
    """
    prefix = os.path.join(args.base_dir, args.dataset)

    if args.dataset == 'kitti':
        img_dir, pat = os.path.join(prefix, '05', 'image_0'), '*.png'
        seq = sorted(glob.glob(os.path.join(img_dir, pat)))

    elif args.dataset == 'parking':
        img_dir, pat = os.path.join(prefix, 'images'), '*.png'
        seq = sorted(glob.glob(os.path.join(img_dir, pat)))

    elif args.dataset == 'malaga':
        img_dir = os.path.join(prefix,
                               'malaga-urban-dataset-extract-07_rectified_800x600_Images')
        seq = sorted(glob.glob(os.path.join(img_dir, '*_left.jpg')))

    elif args.dataset == 'tum-rgbd':
        img_dir = os.path.join(prefix, 'rgbd_dataset_freiburg1_room', 'rgb')
        seq = sorted(glob.glob(os.path.join(img_dir, '*.png')))

    elif args.dataset == 'custom':
        # read every frame of compressed video into memory
        vid = os.path.join(prefix, 'custom.mp4')
        cap = cv2.VideoCapture(vid)
        seq = []
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            seq.append(fr)
        cap.release()
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    if len(seq) < 2:
        raise RuntimeError("Dataset must contain at least two frames.")
    return seq


def load_frame_pair(args, seq, i):
    """
    Convenience wrapper: returns BGR frame i and i+1.
    Works for both path-based and in-memory sequences.
    """
    if args.dataset == 'custom':            # frames are already BGR np.ndarrays
        return seq[i], seq[i + 1]
    return cv2.imread(seq[i]), cv2.imread(seq[i + 1])
