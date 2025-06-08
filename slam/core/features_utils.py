# featuresExtractNmatch.py
import cv2
import numpy as np

# optional LightGlue imports guarded by try/except
try:
    import torch
    from lightglue import LightGlue, ALIKED
    from lightglue.utils import rbd , load_image
except ImportError:
    raise ImportError('LightGlue unavailable, please install it using instructions on https://github.com/cvg/LightGlue')



# --------------------------------------------------------------------------- #
#  Initialisation helpers
# --------------------------------------------------------------------------- #
def init_feature_pipeline(args):
    """
    Instantiate detector & matcher according to CLI arguments.
    Returns (detector, matcher)
    """
    if args.use_lightglue:
        detector = ALIKED(max_num_keypoints=2048).eval().cuda()
        matcher  = LightGlue(features='aliked').eval().cuda()
    else:
        detector = _get_opencv_detector(args.detector)
        matcher  = _get_opencv_matcher(args.matcher, args.detector)
    return detector, matcher


def _get_opencv_detector(detector_type, max_features=6000):
    if detector_type == 'orb':
        return cv2.ORB_create(max_features)
    if detector_type == 'sift':
        return cv2.SIFT_create()
    if detector_type == 'akaze':
        return cv2.AKAZE_create()
    raise ValueError(f"Unsupported detector: {detector_type}")


def _get_opencv_matcher(matcher_type, detector_type):
    if matcher_type != 'bf':
        raise ValueError(f"Unsupported matcher: {matcher_type}")
    norm = cv2.NORM_HAMMING if detector_type in ['orb', 'akaze'] else cv2.NORM_L2
    return cv2.BFMatcher(norm, crossCheck=True)


# --------------------------------------------------------------------------- #
#  Pure-OpenCV pipeline
# --------------------------------------------------------------------------- #
def _opencv_detect_and_match(img1, img2, detector, matcher):
    kp1, des1 = detector.detectAndCompute(img1, None)
    kp2, des2 = detector.detectAndCompute(img2, None)

    if des1 is None or des2 is None:
        return [], [], [], [], []        # gracefully handle empty images

    matches = sorted(matcher.match(des1, des2), key=lambda m: m.distance)

    return kp1, kp2, des1, des2, matches


# --------------------------------------------------------------------------- #
#  LightGlue pipeline
# --------------------------------------------------------------------------- #
def _bgr_to_tensor(image):
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    tensor  = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0)
    return tensor.cuda() if torch.cuda.is_available() else tensor


def _convert_lightglue_to_opencv(kp0, kp1, matches):
    n = kp0.shape[0]
    cv_kp0 = [cv2.KeyPoint(float(x), float(y), 1) for x, y in kp0]
    cv_kp1 = [cv2.KeyPoint(float(x), float(y), 1) for x, y in kp1]
    cv_matches = [cv2.DMatch(int(i), int(j), 0, 0.0) for i, j in matches]
    return cv_kp0, cv_kp1, cv_matches


def _lightglue_detect_and_match(img1, img2, extractor, matcher):
    t0, t1 = _bgr_to_tensor(img1), _bgr_to_tensor(img2)

    f0, f1   = extractor.extract(t0), extractor.extract(t1)
    matches  = matcher({'image0': f0, 'image1': f1})

    f0, f1   = rbd(f0), rbd(f1)
    matches  = rbd(matches)

    kp0, kp1 = f0['keypoints'], f1['keypoints']
    des0, des1 = f0['descriptors'], f1['descriptors']
    cv_kp0, cv_kp1, cv_matches = _convert_lightglue_to_opencv(kp0, kp1,
                                                              matches['matches'])
    return cv_kp0, cv_kp1, des0, des1, cv_matches


# --------------------------------------------------------------------------- #
#  Public API
# --------------------------------------------------------------------------- #
def detect_and_match(img1, img2, detector, matcher, args):
    """
    Front-end entry. Chooses OpenCV or LightGlue depending on CLI flag.
    """
    if args.use_lightglue:
        return _lightglue_detect_and_match(img1, img2, detector, matcher)
    return _opencv_detect_and_match(img1, img2, detector, matcher)


def filter_matches_ransac(kp1, kp2, matches, thresh=1.0):
    """
    Drop outliers using the fundamental matrix + RANSAC.
    """
    if len(matches) < 8:
        return matches

    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

    _, mask = cv2.findFundamentalMat(pts1, pts2,
                                     cv2.FM_RANSAC, thresh, 0.99)
    mask = mask.ravel().astype(bool)
    return [m for m, ok in zip(matches, mask) if ok]
