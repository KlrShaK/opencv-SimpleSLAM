import os
import glob
import argparse
import time
from tqdm import tqdm

import cv2
import numpy as np

# Optional LightGlue imports
try:
    import torch
    from lightglue import LightGlue, ALIKED
    from lightglue.utils import rbd , load_image
    LIGHTGLUE_AVAILABLE = True
except ImportError:
    LIGHTGLUE_AVAILABLE = False


def get_detector(detector_type, max_features=6000):
    if detector_type == 'orb':
        return cv2.ORB_create(max_features)
    elif detector_type == 'sift':
        return cv2.SIFT_create()
    elif detector_type == 'akaze':
        return cv2.AKAZE_create()
    raise ValueError(f"Unsupported detector: {detector_type}")


def get_matcher(matcher_type, detector_type=None):
    if matcher_type == 'bf':
        norm = cv2.NORM_HAMMING if detector_type in ['orb','akaze'] else cv2.NORM_L2
        return cv2.BFMatcher(norm, crossCheck=True)
    raise ValueError(f"Unsupported matcher: {matcher_type}")


def bgr_to_tensor(image):
    """Convert a OpenCV‐style BGR uint8 into (3,H,W) torch tensor in [0,1] ."""
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
    tensor = torch.from_numpy(img_rgb).permute(2,0,1).unsqueeze(0)
    return tensor.cuda() if torch.cuda.is_available() else tensor

def tensor_to_bgr(img_tensor):
    """Convert a (1,3,H,W) or (3,H,W) torch tensor in [0,1] to RGB → OpenCV‐style BGR uint8 image"""
    if img_tensor.dim() == 4:
        img_tensor = img_tensor[0]
    img_np = img_tensor.permute(1,2,0).cpu().numpy()
    img_np = (img_np * 255).clip(0,255).astype(np.uint8)
    return cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

def convert_lightglue_to_opencv(keypoints0, keypoints1, matches):
        """
        Convert filtered LightGlue keypoints and matches into OpenCV-compatible KeyPoint and DMatch objects.
        Here, keypoints0 and keypoints1 are assumed to already be filtered (only matched keypoints).
        
        Returns:
            opencv_kp0: List of cv2.KeyPoint objects for image0.
            opencv_kp1: List of cv2.KeyPoint objects for image1.
            opencv_matches: List of cv2.DMatch objects where each match is simply (i,i).
        """
        n_matches = keypoints0.shape[0]  # number of matched keypoints
        opencv_kp0 = [cv2.KeyPoint(float(pt[0]), float(pt[1]), 1) for pt in keypoints0]
        opencv_kp1 = [cv2.KeyPoint(float(pt[0]), float(pt[1]), 1) for pt in keypoints1]

        opencv_matches = [
        cv2.DMatch(int(q), int(t), 0, 0.0) for q, t in matches
        ]

        return opencv_kp0, opencv_kp1, opencv_matches


def lightglue_match(img1, img2, extractor, matcher, min_conf=0.0):
        """
        Uses ALIKED and LightGlue to extract features and match them,
        then converts the outputs to OpenCV-compatible formats.
        Returns:
            pos_kp0: numpy array of keypoint coordinates for image0 (shape (K,2)).
            pos_kp1: numpy array of keypoint coordinates for image1 (shape (K,2)).
            opencv_matches: list of cv2.DMatch objects.
            opencv_kp0: list of cv2.KeyPoint objects for image0.
            opencv_kp1: list of cv2.KeyPoint objects for image1.
        """
        # Convert image to torch tensor
        img1_tensor = bgr_to_tensor(img1)
        img2_tensor = bgr_to_tensor(img2)
        feats0 = extractor.extract(img1_tensor)
        feats1 = extractor.extract(img2_tensor)
        matches_out = matcher({'image0': feats0, 'image1': feats1})
        # Remove batch dimension using rbd from lightglue.utils
        feats0 = rbd(feats0)
        feats1 = rbd(feats1)
        matches_out = rbd(matches_out)
        
        # Get matches and keypoints as numpy arrays.
        matches = matches_out['matches']  # shape (K,2)
        # keypoints returned by LightGlue are numpy arrays of (x,y)
        keypoints0 = feats0['keypoints'][matches[..., 0]]
        keypoints1 = feats1['keypoints'][matches[..., 1]]
        
        # Convert the keypoints and matches to OpenCV formats.
        opencv_kp0, opencv_kp1, opencv_matches = convert_lightglue_to_opencv(keypoints0, keypoints1, matches)
        
        return opencv_kp0, opencv_kp1, opencv_matches, keypoints0, keypoints1

# def lightglue_match(img1, img2, extractor, matcher, min_conf=0.0):
#     """
#     Run LightGlue matching between img1 and img2, returning OpenCV-style keypoints and DMatches.

#     Args:
#         img1, img2       : BGR images as NumPy arrays.
#         extractor        : ALIKED extractor (e.g. ALIKED(...).eval()).
#         matcher          : LightGlue matcher (e.g. LightGlue(...).eval()).
#         min_conf (float) : Minimum confidence threshold [0.0–1.0] to keep a match.

#     Returns:
#         pts0, pts1, dm   : Two lists of cv2.KeyPoint and a list of cv2.DMatch.
#     """
#     # Feature extraction
#     f0, f1 = extractor.extract(img1), extractor.extract(img2)

#     # 2) LightGlue matching
#     matches = matcher({'image0': f0, 'image1': f1})
#     f0, f1, matches = rbd(f0), rbd(f1), rbd(matches)

#     idx = matches['matches']                # shape (N,2), may contain -1
#     scores0 = matches.get('matching_scores0', None)
#     scores1 = matches.get('matching_scores1', None)

#     # 3) Filter out invalid (-1) matches
#     valid = (idx[:, 0] >= 0) & (idx[:, 1] >= 0)
#     if scores0 is not None and scores1 is not None and min_conf > 0.0:
#         valid &= (scores0 >= min_conf) & (scores1 >= min_conf)

#     idx = idx[valid]

#     # 4) Convert to OpenCV KeyPoints + DMatches
#     kp0 = f0['keypoints'][idx[..., 0]]
#     kp1 = f1['keypoints'][idx[..., 1]]

#     pts0, pts1, dm = convert_lightglue_to_opencv(kp0, kp1, idx)
#     return pts0, pts1, dm


def feature_detect_and_match(img1,img2,detector,matcher):
    kp1,des1=detector.detectAndCompute(img1,None)
    kp2,des2=detector.detectAndCompute(img2,None)
    if des1 is None or des2 is None:
        return [],[],[]
    matches=matcher.match(des1,des2)
    return kp1,kp2,sorted(matches,key=lambda m:m.distance)

import cv2

def update_and_prune_tracks(matches, prev_map, tracks, kp_curr, frame_idx, next_track_id, prune_age=30):
    """
    Update feature tracks given a list of matches, then prune stale ones.

    Args:
        matches        : List[cv2.DMatch] between previous and current keypoints.
        prev_map       : Dict[int, int] mapping prev-frame kp index -> track_id.
        tracks         : Dict[int, List[Tuple[int, int, int]]], 
                         each track_id → list of (frame_idx, x, y).
        kp_curr        : List[cv2.KeyPoint] for current frame.
        frame_idx      : int, index of the current frame (1-based or 0-based).
        next_track_id  : int, the next unused track ID.
        prune_age      : int, max age (in frames) before a track is discarded.

    Returns:
        curr_map       : Dict[int, int] mapping curr-frame kp index → track_id.
        tracks         : Updated tracks dict.
        next_track_id  : Updated next unused track ID.
    """
    curr_map = {}

    # 1) Assign each match to an existing or new track_id
    for m in matches:
        q = m.queryIdx    # keypoint idx in previous frame
        t = m.trainIdx    # keypoint idx in current frame

        # extract integer pixel coords
        x, y = int(kp_curr[t].pt[0]), int(kp_curr[t].pt[1])

        if q in prev_map:
            # continue an existing track
            tid = prev_map[q]
        else:
            # start a brand-new track
            tid = next_track_id
            tracks[tid] = []
            next_track_id += 1

        # map this current keypoint to the track
        curr_map[t] = tid
        # append the new observation
        tracks[tid].append((frame_idx, x, y))

    # 2) Prune any track not seen in the last `prune_age` frames
    for tid, pts in list(tracks.items()):
        last_seen_frame = pts[-1][0]
        if (frame_idx - last_seen_frame) > prune_age:
            del tracks[tid]

    return curr_map, tracks, next_track_id


def draw_tracks(vis, tracks, current_frame, max_age=10, sample_rate=5, max_tracks=50):
    """
    Draw each track's path with color decaying from green (new) to red (old).
    Only draw up to max_tracks most recent tracks, and sample every sample_rate-th track.
    """
    recent=[(tid,pts) for tid,pts in tracks.items() if pts and current_frame-pts[-1][0]<=max_age]
    recent.sort(key=lambda x:x[1][-1][0],reverse=True)
    drawn=0
    for tid,pts in recent:
        if drawn>=max_tracks: break
        if tid%sample_rate!=0: continue
        pts=[p for p in pts if current_frame-p[0]<=max_age]
        for j in range(1,len(pts)):
            frame_idx,x0,y0=pts[j-1]
            _,x1,y1=pts[j]
            age=current_frame-frame_idx
            ratio=age/max_age
            b=0
            g=int(255*(1-ratio))
            r=int(255*ratio)
            cv2.line(vis,(int(x0),int(y0)),(int(x1),int(y1)),(b,g,r),2)
        drawn+=1
    return vis

# TODO REMOVE


def main():
    parser=argparse.ArgumentParser("Feature tracking with RANSAC filtering")
    parser.add_argument('--dataset',choices=['kitti','malaga','parking','custom'],required=True)
    parser.add_argument('--base_dir',default='.././Dataset')
    parser.add_argument('--detector',choices=['orb','sift','akaze'],default='orb')
    parser.add_argument('--matcher',choices=['bf'],default='bf')
    parser.add_argument('--use_lightglue',action='store_true')
    parser.add_argument('--fps',type=float,default=10)
    parser.add_argument('--ransac_thresh',type=float,default=1.0,
                        help='RANSAC threshold for fundamental matrix')
    args=parser.parse_args()

    interval_ms=int(1000/args.fps) if args.fps>0 else 0
    # init modules once
    if args.use_lightglue:
        if not LIGHTGLUE_AVAILABLE: raise ImportError('LightGlue unavailable')
        extractor=ALIKED(max_num_keypoints=2048).eval().cuda()
        matcher_lg=LightGlue(features='aliked').eval().cuda()
    else:
        detector=get_detector(args.detector)
        matcher_cv=get_matcher(args.matcher,args.detector)

    # load sequence
    prefix=os.path.join(args.base_dir, args.dataset)
    is_custom=False
    if args.dataset=='kitti': img_dir,pat=os.path.join(prefix,'05','image_0'),'*.png'
    elif args.dataset=='parking': img_dir,pat=os.path.join(prefix,'images'),'*.png'
    elif args.dataset=='malaga': img_dir,pat=os.path.join(prefix,'malaga-urban-dataset-extract-07_rectified_1024x768_Images'),' *_left.jpg'
    else:
        vid=os.path.join(prefix,'custom_compress.mp4')
        cap=cv2.VideoCapture(vid)
        seq=[]
        while True:
            ok,fr=cap.read()
            if not ok: break; seq.append(fr)
        cap.release(); is_custom=True
    if not is_custom: seq=sorted(glob.glob(os.path.join(img_dir,pat)))

    # tracking data
    track_id=0; prev_map={}; tracks={}
    cv2.namedWindow('Feature Tracking',cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Feature Tracking',1200,600)

    total=len(seq)-1; prev_time=time.time(); achieved_fps=0.0
    for i in tqdm(range(total),desc='Tracking'):
        # load images
        if is_custom: img1,img2=seq[i],seq[i+1]
        # elif args.use_lightglue: img1=load_image(seq[i]).cuda(); img2=load_image(seq[i+1]).cuda()
        else: img1=cv2.imread(seq[i]);img2=cv2.imread(seq[i+1])
        
        # match features
        if args.use_lightglue:
            kp_map1, kp_map2, matches,_ ,_ =lightglue_match(img1,img2,extractor,matcher_lg)
            # print("type(kp_map1), type(kp_map2), type(matches)")
            # print(type(kp_map1), type(kp_map2), type(matches))
            # print(kp_map1[:50], "\n\n\n",kp_map2[:50],"\n\n\n", matches[:50])
            vis= img2.copy()
            import random
            vis = cv2.drawMatches(img1, kp_map1, img2, kp_map2, random.sample(matches, 50), None,
                                  flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            # for idx,(p0,p1) in enumerate(zip(pts1,pts2)):
            #     if idx>=50: break
            #     x0,y0 = p0.pt; x1,y1 = p1.pt
            #     vis= img2.copy() if not args.use_lightglue else tensor_to_bgr(img2)
            #     cv2.circle(vis, (int(x1),int(y1)), 4, (0,0,255), 2)
        else:
            kp_map1, kp_map2, matches=feature_detect_and_match(img1, img2, detector, matcher_cv)
            # print("type(kp_map1), type(kp_map2), type(matches)")
            # print(type(kp_map1), type(kp_map2), type(matches))
            # print(kp_map1[:50], "\n\n\n",kp_map2[:50],"\n\n\n", matches[:50])



        # filter with RANSAC
        if len(matches)>=8:
            pts1_arr=np.float32([kp_map1[m.queryIdx].pt for m in matches])
            pts2_arr=np.float32([kp_map2[m.trainIdx].pt for m in matches])
            F,mask=cv2.findFundamentalMat(pts1_arr,pts2_arr,
                                          cv2.FM_RANSAC,args.ransac_thresh,0.99)
            mask=mask.ravel().astype(bool)
            matches=[m for m,mk in zip(matches,mask) if mk]
        
        # update & prune tracks
        frame_no = i + 1
        prev_map, tracks, track_id = update_and_prune_tracks(matches, prev_map, tracks, kp_map2, frame_no, track_id, prune_age=30)
        
        # draw
        vis= img2.copy()

        vis=draw_tracks(vis,tracks,i+1)
        for t,tid in prev_map.items():
            cv2.circle(vis,tuple(map(int, kp_map2[t].pt)),3,(0,255,0),-1)

        text=f"Frame {i+1}/{total} | Tracks: {len(tracks)} | FPS: {achieved_fps:.1f}"
        cv2.putText(vis,text,(10,30),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)

        # show
        key=cv2.waitKey(0)
        cv2.imshow('Feature Tracking',vis)
        # measure fps
        now=time.time();dt=now-prev_time
        if dt>0: achieved_fps=1.0/dt
        prev_time=now
        if key&0xFF==27: break
    cv2.destroyAllWindows()

if __name__=='__main__': 
    main()
