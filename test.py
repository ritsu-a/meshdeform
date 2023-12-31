import numpy as np
from utils.mesh_process import tetrahedralize, load_mesh, compute_tetrahedron_centroids, save_pointcloud, save_mesh, plt_mesh, plt_meshes, plt_mesh2
import os
import cv2


file_dir = "/share1/pengyang/hande/1205/"
file_list = os.listdir(file_dir)
print(len(file_list), file_list[0])


save_dir = "./test/"
os.makedirs(save_dir, exist_ok =True)
os.makedirs(save_dir + "imgs", exist_ok =True)

meta = np.load(file_dir + file_list[0])

for idx in range(100):
    verts = meta["verts"][idx]
    noisy_pcd = meta["noisy_pcds"][idx]
    print(verts.shape, noisy_pcd.shape)

    faces = meta["faces"][idx]
    plt_mesh2(verts, noisy_pcd, faces, save_dir + f"imgs/data_{idx}.png")
    

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
height, width, channels = cv2.imread("./test/imgs/data_0.png").shape
frame_size = (width, height)
out = cv2.VideoWriter("./test/output.mp4", fourcc, 5, frame_size)


file_list = os.listdir("./test/imgs/")
for file in file_list:
    cv_dst = cv2.imread(f"./test/imgs/{file}")
    out.write(cv_dst)

