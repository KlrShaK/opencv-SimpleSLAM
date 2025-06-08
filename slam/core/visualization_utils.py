# visualization_utils.py
import cv2

def draw_tracks(vis,
                tracks,
                current_frame,
                max_age=10,
                sample_rate=5,
                max_tracks=1000):
    """
    Overlay point-tracks with colour fading (green ➜ red as they age).
    """
    recent = [(tid, pts) for tid, pts in tracks.items()
              if current_frame - pts[-1][0] <= max_age]
    recent.sort(key=lambda x: x[1][-1][0], reverse=True)

    drawn = 0
    for tid, pts in recent:
        if drawn >= max_tracks:
            break
        if tid % sample_rate:
            continue
        pts = [p for p in pts if current_frame - p[0] <= max_age]
        for j in range(1, len(pts)):
            _, x0, y0 = pts[j - 1]
            _, x1, y1 = pts[j]
            ratio = (current_frame - pts[j - 1][0]) / max_age
            colour = (0, int(255 * (1 - ratio)), int(255 * ratio))
            cv2.line(vis, (x0, y0), (x1, y1), colour, 2)
        drawn += 1
    return vis
