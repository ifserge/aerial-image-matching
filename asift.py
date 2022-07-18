"""
Affine invariant feature-based image matching.
Based on Affine-SIFT algorithm[1].
The original implementation is based on SIFT, support for other common detectors is also added for testing use. Homography RANSAC is used to reject outliers.
Threading is used for faster affine sampling. Multicore CPUs with Hyper-threading is strongly recommended for better performance.
Copyleft Lang Zhou, zhoulang731@tongji.edu.cn
GitHub: https://github.com/Mars-Rover-Localization/PyASIFT
Created April 2021
Last modified October 2021
[1] http://www.ipol.im/pub/algo/my_affine_sift/
"""
from contextlib import contextmanager

# Third party modules
import numpy as np
import cv2
# Built-in modules
from multiprocessing.pool import ThreadPool     # Use multiprocessing to avoid GIL
import sys

# Third party modules, opencv-contrib-python is needed
import cv2
import numpy as np


MAX_SIZE = 256

FLANN_INDEX_KDTREE = 1

FLANN_INDEX_LSH = 6


def clock():
    return cv2.getTickCount() / cv2.getTickFrequency()


@contextmanager
def Timer(msg):
    print(msg)
    start = clock()
    try:
        yield
    finally:
        print("%.4f ms" % ((clock() - start) * 1000))


def log_keypoints(kp_pairs, path: str = 'sample/keypoints.txt'):
    with open(path, 'w') as log:
        for kp1, kp2 in kp_pairs:
            log.write(f"{np.int32(kp1.pt)}      {np.int32(kp2.pt)}\n")
    log.close()

    print(f"Keypoints logged at {path}")


def image_resize(src, ratio: float):
    dim = (int(src.shape[-1] * ratio), int(src.shape[0] * ratio))
    return cv2.resize(src, dim, interpolation=cv2.INTER_AREA)


def image_split(src):
    w = src.shape[1]
    half = int(w / 2)
    left_img = src[:, half:]
    right_img = src[:, :half]

    return left_img, right_img



def init_feature(name):
    chunks = name.split('-')

    if chunks[0] == 'sift':
        detector = cv2.SIFT_create()
        norm = cv2.NORM_L2
    elif chunks[0] == 'surf':
        detector = cv2.xfeatures2d.SURF_create(800)
        norm = cv2.NORM_L2
    elif chunks[0] == 'orb':
        detector = cv2.ORB_create(400)
        norm = cv2.NORM_HAMMING
    elif chunks[0] == 'akaze':
        detector = cv2.AKAZE_create()
        norm = cv2.NORM_HAMMING
    elif chunks[0] == 'brisk':
        detector = cv2.BRISK_create()
        norm = cv2.NORM_HAMMING
    else:
        return None, None   # Return None if unknown detector name

    if 'flann' in chunks:
        if norm == cv2.NORM_L2:
            flann_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        else:
            flann_params = dict(algorithm=FLANN_INDEX_LSH,
                                table_number=6,  # 12
                                key_size=12,  # 20
                                multi_probe_level=1)  # 2

        matcher = cv2.FlannBasedMatcher(flann_params)
    else:
        matcher = cv2.BFMatcher(norm)

    return detector, matcher


def filter_matches(kp1, kp2, matches, ratio=0.7):
    mkp1, mkp2 = [], []

    for m in matches:
        if len(m) == 2 and m[0].distance < m[1].distance * ratio:
            m = m[0]
            mkp1.append(kp1[m.queryIdx])
            mkp2.append(kp2[m.trainIdx])

    p1 = np.float32([kp.pt for kp in mkp1])
    p2 = np.float32([kp.pt for kp in mkp2])
    kp_pairs = zip(mkp1, mkp2)

    return p1, p2, list(kp_pairs)


def draw_match(result_title, img1, img2, kp_pairs, status=None, H=None):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # Create visualized result image
    vis = np.zeros((max(h1, h2), w1 + w2), np.uint8)
    vis[:h1, :w1] = img1
    vis[:h2, w1:w1 + w2] = img2
    vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

    if H is not None:
        corners = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]])
        corners = np.int32(cv2.perspectiveTransform(corners.reshape(1, -1, 2), H).reshape(-1, 2) + (w1, 0))
        cv2.polylines(vis, [corners], True, (255, 255, 255))

    if status is None:
        status = np.ones(len(kp_pairs), np.bool_)
    p1, p2 = [], []  # python 2 / python 3 change of zip unpacking

    for kpp in kp_pairs:
        p1.append(np.int32(kpp[0].pt))
        p2.append(np.int32(np.array(kpp[1].pt) + [w1, 0]))

    green = (0, 255, 0)
    red = (0, 0, 255)

    for (x1, y1), (x2, y2), inlier in zip(p1, p2, status):
        if inlier:
            color = green
            cv2.circle(vis, (x1, y1), 2, color, -1)
            cv2.circle(vis, (x2, y2), 2, color, -1)
        else:
            color = red
            r = 2
            thickness = 3
            cv2.line(vis, (x1 - r, y1 - r), (x1 + r, y1 + r), color, thickness)
            cv2.line(vis, (x1 - r, y1 + r), (x1 + r, y1 - r), color, thickness)
            cv2.line(vis, (x2 - r, y2 - r), (x2 + r, y2 + r), color, thickness)
            cv2.line(vis, (x2 - r, y2 + r), (x2 + r, y2 - r), color, thickness)

    for (x1, y1), (x2, y2), inlier in zip(p1, p2, status):
        if inlier:
            cv2.line(vis, (x1, y1), (x2, y2), green)

    cv2.imshow(result_title, vis)

    return vis


def affine_skew(tilt, phi, img, mask=None):
    """
    affine_skew(tilt, phi, img, mask=None) -> skew_img, skew_mask, Ai
    phi is in degrees
    Ai is an affine transform matrix from skew_img to img
    """
    h, w = img.shape[:2]

    if mask is None:
        mask = np.zeros((h, w), np.uint8)
        mask[:] = 255

    A = np.float32([[1, 0, 0], [0, 1, 0]])

    # Rotate image
    if phi != 0.0:
        phi = np.deg2rad(phi)
        s, c = np.sin(phi), np.cos(phi)
        A = np.float32([[c, -s], [s, c]])
        corners = [[0, 0], [w, 0], [w, h], [0, h]]
        tcorners = np.int32(np.dot(corners, A.T))
        x, y, w, h = cv2.boundingRect(tcorners.reshape(1, -1, 2))
        A = np.hstack([A, [[-x], [-y]]])
        img = cv2.warpAffine(img, A, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    # Tilt image (resizing after rotation)
    if tilt != 1.0:
        s = 0.8 * np.sqrt(tilt * tilt - 1)
        img = cv2.GaussianBlur(img, (0, 0), sigmaX=s, sigmaY=0.01)
        img = cv2.resize(img, (0, 0), fx=1.0 / tilt, fy=1.0, interpolation=cv2.INTER_NEAREST)
        A[0] /= tilt

    if phi != 0.0 or tilt != 1.0:
        h, w = img.shape[:2]
        mask = cv2.warpAffine(mask, A, (w, h), flags=cv2.INTER_NEAREST)

    Ai = cv2.invertAffineTransform(A)

    return img, mask, Ai


def affine_detect(detector, img, pool=None):
    """
    affine_detect(detector, img, mask=None, pool=None) -> keypoints, descrs
    Apply a set of affine transformations to the image, detect keypoints and
    reproject them into initial image coordinates.
    ThreadPool object may be passed to speedup the computation. Please use multiprocess pool to bypass GIL limitations.
    """
    params = [(1.0, 0.0)]

    # Simulate all possible affine transformations
    for t in 2 ** (0.5 * np.arange(1, 6)):
        for phi in np.arange(0, 180, 72.0 / t):
            params.append((t, phi))


    def f(p):
        t, phi = p
        timg, tmask, Ai = affine_skew(t, phi, img)
        keypoints, descrs = detector.detectAndCompute(timg, tmask)

        for kp in keypoints:
            x, y = kp.pt
            kp.pt = tuple(np.dot(Ai, (x, y, 1)))

        if descrs is None:
            descrs = []

        return keypoints, descrs

    keypoints, descrs = [], []

    ires = pool.imap(f, params)

    for i, (k, d) in enumerate(ires):
        #print(f"Affine sampling: {i + 1} / {len(params)}\r", end='')
        keypoints.extend(k)
        descrs.extend(d)

    #print()

    return keypoints, np.array(descrs)


def affine_detect_nopool(detector, img):
    """
    affine_detect(detector, img, mask=None, pool=None) -> keypoints, descrs
    Apply a set of affine transformations to the image, detect keypoints and
    reproject them into initial image coordinates.
    ThreadPool object may be passed to speedup the computation. Please use multiprocess pool to bypass GIL limitations.
    """
    params = [(1.0, 0.0)]

    # Simulate all possible affine transformations
    for t in 2 ** (0.5 * np.arange(1, 6)):
        for phi in np.arange(0, 180, 72.0 / t):
            params.append((t, phi))


    def f(p):
        t, phi = p
        timg, tmask, Ai = affine_skew(t, phi, img)
        keypoints, descrs = detector.detectAndCompute(timg, tmask)

        for kp in keypoints:
            x, y = kp.pt
            kp.pt = tuple(np.dot(Ai, (x, y, 1)))

        if descrs is None:
            descrs = []

        return keypoints, descrs

    keypoints, descrs = [], []

    ires = [f(x) for x in params]

    for i, (k, d) in enumerate(ires):
        #print(f"Affine sampling: {i + 1} / {len(params)}\r", end='')
        keypoints.extend(k)
        descrs.extend(d)

    #print()

    return keypoints, np.array(descrs)