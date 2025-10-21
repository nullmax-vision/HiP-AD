import numpy as np
from projects.mmdet3d_plugin.datasets.pipelines.vectorize_numpy import VectorizeMapNumpy

ego_fut_ts = 6

anchor = np.load("data/kmeans/b2d_plan_spat_6x8_5m.npy")

vectorizer = VectorizeMapNumpy(2, ego_fut_ts +1)
anchor_pt1 = anchor.copy()
anchor_pt1 = anchor_pt1.reshape(-1, ego_fut_ts, 2)
anchor_pt1 = anchor_pt1[:, :1, :]
anchor_pt1 = np.concatenate([np.zeros(anchor_pt1.shape), anchor_pt1], axis=1)
anchor_pt1 = vectorizer(anchor_pt1)
anchor_pt1 = np.stack(anchor_pt1, axis=0)
anchor_pt1 = anchor_pt1[: , 1:]
anchor_pt1 = anchor_pt1.reshape(-1, ego_fut_ts * 2)

anchor_pt1 = anchor_pt1.reshape(*anchor.shape)

np.save("data/kmeans/b2d_plan_spat_6x8_2m.npy", anchor_pt1)

