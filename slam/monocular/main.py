# main.py
"""
Entry-point: high-level processing loop
--------------------------------------
$ python main.py --dataset kitti --base_dir ../Dataset
"""
import argparse
import cv2
import lz4.frame
import numpy as np
from tqdm import tqdm


from slam.core.dataloader import load_sequence, load_frame_pair
from slam.core.features_utils import (init_feature_pipeline, detect_and_match, filter_matches_ransac)
from slam.core.keyframe_utils import (Keyframe, make_thumb, is_new_keyframe, update_and_prune_tracks)
from slam.core.visualization_utils import draw_tracks

# --------------------------------------------------------------------------- #
#  CLI
# --------------------------------------------------------------------------- #
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("Feature tracking with key-frames")
    p.add_argument('--dataset',
                   choices=['kitti', 'malaga', 'tum-rgbd', 'custom'],
                   required=True)
    p.add_argument('--base_dir', default='../Dataset')
    # feature/detector settings
    p.add_argument('--detector', choices=['orb', 'sift', 'akaze'],
                   default='orb')
    p.add_argument('--matcher', choices=['bf'], default='bf')
    p.add_argument('--use_lightglue', action='store_true')
    # runtime
    p.add_argument('--fps', type=float, default=10)
    # RANSAC
    p.add_argument('--ransac_thresh', type=float, default=1.0)
    # key-frame params
    p.add_argument('--kf_max_disp', type=float, default=45)
    p.add_argument('--kf_min_inliers', type=float, default=150)
    p.add_argument('--kf_cooldown', type=int, default=3)
    p.add_argument('--kf_thumb_hw', type=int, nargs=2,
                   default=[640, 360])
    return p


# --------------------------------------------------------------------------- #
#  Main processing loop
# --------------------------------------------------------------------------- #
def main():
    args = _build_parser().parse_args()

    seq = load_sequence(args)
    detector, matcher = init_feature_pipeline(args)

    # tracking state
    prev_map, tracks = {}, {}
    next_track_id = 0
    achieved_fps = 0.0
    last_time = cv2.getTickCount() / cv2.getTickFrequency()

    # key-frame list
    kfs: list[Keyframe] = []
    last_kf_idx = -999        # sentinel

    cv2.namedWindow('Feature Tracking', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Feature Tracking', 1200, 600)

    total = len(seq) - 1
    for i in tqdm(range(total), desc='Tracking'):
        img1, img2 = load_frame_pair(args, seq, i)

        # feature extraction / matching
        kp1, kp2, des1, des2, matches = detect_and_match(
            img1, img2, detector, matcher, args)
        matches = filter_matches_ransac(kp1, kp2, matches,
                                        args.ransac_thresh)

        # track maintenance
        frame_no = i + 1
        prev_map, tracks, next_track_id = update_and_prune_tracks(
            matches, prev_map, tracks, kp2, frame_no, next_track_id)

        # ---------------- Key-frame subsystem ---------------- #
        if i == 0:                   # bootstrap KF-0
            kfs.append(Keyframe(0,
                                seq[0] if isinstance(seq[0], str) else "",
                                kp1, des1,
                                make_thumb(img1, tuple(args.kf_thumb_hw))))
            last_kf_idx = 0
        else:
            kf = kfs[-1]
            kf_matches = (matcher.match(kf.desc, des2)
                          if len(kf.desc) and len(des2) else [])
            kf_matches = filter_matches_ransac(kf.kps, kp2,
                                               kf_matches,
                                               args.ransac_thresh)

            if is_new_keyframe(frame_no, kf_matches,
                               len(kf.kps), kp2, kf.kps,
                               args.kf_max_disp,
                               args.kf_min_inliers,
                               args.kf_cooldown,
                               last_kf_idx):
                kfs.append(Keyframe(frame_no,
                                    seq[i + 1] if isinstance(seq[i + 1], str)
                                    else "",
                                    kp2, des2,
                                    make_thumb(img2,
                                               tuple(args.kf_thumb_hw))))
                last_kf_idx = frame_no

        # ---------------------- GUI -------------------------- #
        vis = draw_tracks(img2.copy(), tracks, frame_no)
        for t in prev_map.keys():
            cv2.circle(vis, tuple(map(int, kp2[t].pt)),
                       3, (0, 255, 0), -1)

        cv2.putText(vis,
                    f"KF idx: {last_kf_idx}  |  total KFs: {len(kfs)}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 255, 0), 2)

        # thumb strip (last 4)
        thumbs = [cv2.imdecode(
                  np.frombuffer(lz4.frame.decompress(k.thumb), np.uint8),
                  cv2.IMREAD_COLOR) for k in kfs[-4:]]
        bar = (np.hstack(thumbs) if thumbs else
               np.zeros((*args.kf_thumb_hw[::-1], 3), np.uint8))
        cv2.imshow('Keyframes', bar)

        cv2.putText(vis,
                    (f"Frame {frame_no}/{total} | "
                     f"Tracks: {len(tracks)} | "
                     f"FPS: {achieved_fps:.1f}"),
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 255, 255), 2)

        cv2.imshow('Feature Tracking', vis)
        wait_ms = int(1000 / args.fps) if args.fps > 0 else 1
        if cv2.waitKey(wait_ms) & 0xFF == 27:    # ESC
            break

        # update FPS
        now = cv2.getTickCount() / cv2.getTickFrequency()
        achieved_fps = 1.0 / (now - last_time)
        last_time = now

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
