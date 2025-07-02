import pycolmap
import pyceres
import numpy as np
from hloc.utils import viz_3d

def create_reconstruction(num_points=50, num_images=2, seed=3, noise=0):
    state = np.random.RandomState(seed)
    rec   = pycolmap.Reconstruction()

    # ---------------------------------------------------------------- 3-D points
    p3d = state.uniform(-1, 1, (num_points, 3)) + np.array([0, 0, 3])
    for p in p3d:
        rec.add_point3D(p, pycolmap.Track(), np.zeros(3))

    # ---------------------------------------------------------------- camera
    w, h = 640, 480
    cam = pycolmap.Camera(
        model="SIMPLE_PINHOLE",
        width=w,
        height=h,
        params=np.array([max(w, h) * 1.2, w / 2, h / 2]),
        camera_id=0,
    )
    rec.add_camera(cam)

    # ---------------------------------------------------------------- images / frames
    for i in range(num_images):
        # 1️⃣  create a frame and set its world pose
        pose  = pycolmap.Rigid3d(pycolmap.Rotation3d(),
                                 state.uniform(-1, 1, 3))
        frame = pycolmap.Frame(frame_id=i)               # NEW
        frame.rig_from_world = pose                      # NEW  :contentReference[oaicite:0]{index=0}
        rec.add_frame(frame)                             # NEW  :contentReference[oaicite:1]{index=1}

        # 2️⃣  make an image that *references* that frame
        im = pycolmap.Image(
            image_id=i,
            name=str(i),
            camera_id=cam.camera_id,
        )
        im.frame_id = frame.frame_id                     # NEW  :contentReference[oaicite:2]{index=2}

        # 3️⃣  add the image (now passes HasFrameId() check)
        rec.add_image(im)

        # 4️⃣  fetch managed copy for further edits
        im = rec.image(i)

        # 5️⃣  synthetic projections with optional noise
        cam_pts = pose * [p.xyz for p in rec.points3D.values()]
        uv      = cam.img_from_cam(cam_pts)
        uv     += state.randn(len(uv), 2) * noise

        im.points2D = pycolmap.Point2DList(
            [pycolmap.Point2D(p, pid)
             for p, pid in zip(uv, rec.points3D.keys())]
        )

    return rec


# def create_reconstruction(num_points=50, num_images=2, seed=3, noise=0):
#     state = np.random.RandomState(seed)
#     rec = pycolmap.Reconstruction()
#     p3d = state.uniform(-1, 1, (num_points, 3)) + np.array([0, 0, 3])
#     for p in p3d:
#         rec.add_point3D(p, pycolmap.Track(), np.zeros(3))
#     w, h = 640, 480
#     cam = pycolmap.Camera(
#         model="SIMPLE_PINHOLE",
#         width=w,
#         height=h,
#         params=np.array([max(w, h) * 1.2, w / 2, h / 2]),
#         camera_id=0,
#     )
#     rec.add_camera(cam)
#     for i in range(num_images):
#         im = pycolmap.Image(
#             image_id=i,
#             name=str(i),
#             camera_id=cam.camera_id,
#         )

#         im.registered = True
#         p2d = cam.img_from_cam(
#             im.cam_from_world * [p.xyz for p in rec.points3D.values()]
#         )
#         p2d_obs = np.array(p2d) + state.randn(len(p2d), 2) * noise
#         im.points2D = pycolmap.ListPoint2D(
#             [pycolmap.Point2D(p, id_) for p, id_ in zip(p2d_obs, rec.points3D)]
#         )
#         rec.add_image(im)


#         pose=pycolmap.Rigid3d(
#                 pycolmap.Rotation3d(), state.uniform(-1, 1, 3)
#             )
#         im.frame.set_rig_from_world(pose)
#     return rec



rec_gt = create_reconstruction()