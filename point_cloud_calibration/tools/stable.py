import numpy as np
import cv2
from tqdm import tqdm


def movingAverage(curve, radius):  # [n]
    windowSize = 2 * radius + 1
    f = np.ones(windowSize) / windowSize  # [2 * radius + 1]

    padded = np.lib.pad(curve, (radius, radius), 'edge')  # [2 * radius + n]
    smoothed = np.convolve(padded, f, mode='same')
    smoothed = smoothed[radius:-radius]  # [n]
    return smoothed


def smooth(trajectory):
    smoothedTrajectory = np.copy(trajectory)
    for i in range(3):
        smoothedTrajectory[:, i] = movingAverage(trajectory[:, i], radius=100)  # 60
    return smoothedTrajectory


def stabilization(videoPath, savePath):
    cap = cv2.VideoCapture(videoPath)
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(savePath, fourcc, fps, (w, h))

    rightBound = nframes

    # Pre-define transformation-store array
    transforms = np.zeros((nframes, 3), np.float32)

    prevGray = None
    for i in tqdm(range(rightBound)):
        success, curr = cap.read()
        currGray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
        # Calculate optical flow (i.e. track feature points)
        if prevGray is None:
            prevGray = currGray
            continue

        prevFeatures = cv2.goodFeaturesToTrack(prevGray, maxCorners=200, qualityLevel=0.01, minDistance=10,
                                               blockSize=10)
        currFeatures, status, err = cv2.calcOpticalFlowPyrLK(prevGray, currGray, prevFeatures, None)

        assert prevFeatures.shape == currFeatures.shape
        # Filter only valid points
        idx = np.where(status == 1)[0]
        prevFeatures = prevFeatures[idx]
        currFeatures = currFeatures[idx]

        prevFeatures -= [w / 2, h / 2]
        currFeatures -= [w / 2, h / 2]

        m, n = cv2.estimateAffinePartial2D(prevFeatures, currFeatures)

        dx = m[0, 2]
        dy = m[1, 2]
        da = np.arctan2(m[1, 0], m[0, 0])

        transforms[i] = [dx, dy, da]
        prevGray = currGray

    # Compute trajectory using cumulative sum of transformations
    trajectory = np.cumsum(transforms, axis=0)
    smoothedTrajectory = smooth(trajectory)

    difference = smoothedTrajectory - trajectory
    smoothedTransforms = transforms + difference

    # Reset stream to first frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    for i in tqdm(range(rightBound)):
        success, frame = cap.read()
        if not success:
            break

        dx = smoothedTransforms[i, 0]
        dy = smoothedTransforms[i, 1]
        da = smoothedTransforms[i, 2]

        m = cv2.getRotationMatrix2D((w / 2, h / 2), -np.rad2deg(da), 1)
        m[1, 2] += dy
        frameStabilized = cv2.warpAffine(frame, m, (w, h))
        out.write(frameStabilized)
    out.release()


if __name__ == "__main__":
    VideoPath = './data/3.avi'
    SavePath = 'video_out.avi'
    stabilization(VideoPath, SavePath)
