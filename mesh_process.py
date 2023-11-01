import numpy as np
import open3d as o3d
import tetgen
import pyvista as pv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import meshio
import ipdb


def tetrahedralize(vertices,faces):
    # 使用pyvista创建网格对象
    faces_with_prefixed_count = np.hstack((3 * np.ones((faces.shape[0], 1), dtype=faces.dtype), faces))
    flattened_faces = faces_with_prefixed_count.flatten()
    mesh = pv.PolyData(vertices, flattened_faces)

    # 使用TetGen进行四面体化
    tet = tetgen.TetGen(mesh)

    # 四面体剖分
    tet.tetrahedralize(order=1, mindihedral=20, minratio=1.5)

    tet_mesh = tet.grid  # 这是四面体化后的网格
    return tet_mesh

# Load object model
def load_mesh(file_path):
    mesh = o3d.io.read_triangle_mesh(file_path)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.triangles,dtype=np.int32)
    print('Loaded mesh, vertices: ', vertices.shape, ', faces: ', faces.shape)
    return vertices, faces


def plot_tetrahedral_mesh(vertices, tetrahedra):
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    tetrahedra = tetrahedra.reshape(-1, 5)[:, 1:]  # 转换成形状为(-1, 5)的数组，并去掉每行的第一个值，它是节点数
    
    # 为每个四面体绘制四个面
    # 为每个四面体绘制四个面
    for tetra in tetrahedra:
        triangles = [
            [tetra[0], tetra[1], tetra[2]],
            [tetra[0], tetra[1], tetra[3]],
            [tetra[0], tetra[2], tetra[3]],
            [tetra[1], tetra[2], tetra[3]]
        ]
        for tri in triangles:
            triangle_coords = [vertices[i] for i in tri]
            ax.add_collection3d(Poly3DCollection([triangle_coords], alpha=0.25, edgecolor='k'))
            
    ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], s=5, c='r', marker='o')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.savefig('test.png')

def save_tet_mesh_as_ply(tet_mesh, file_path):
    tetrahedra = tet_mesh.cells.reshape(-1, 5)[:, 1:]
    
    # 使用meshio保存为.ply格式
    meshio.write_points_cells(
        file_path,
        tet_mesh.points,
        [("tetra", tetrahedra)]
    )

def compute_tetrahedron_centroids(points, cells):
    # 提取四面体的顶点坐标
    tetrahedra_indices = cells.reshape(-1, 5)[:, 1:]
    tetra_coords = points[tetrahedra_indices]
    
    # 计算每个四面体的重心
    centroids = np.mean(tetra_coords, axis=1)
    
    return centroids

def save_pointcloud(centroids, filename):
    
    # 创建PointCloud对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(centroids)
    
    # 保存为PLY
    o3d.io.write_point_cloud(filename, pcd)

def save_mesh(vertices, faces, handle_vertices,static_vertices, filename):
    
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(faces)

    default_color = [0.7, 0.7, 0.7]  # 灰色
    colors = [default_color for _ in range(len(vertices))]

    red_color = [1, 0, 0]  # 红色
    for idx in handle_vertices:
        colors[idx] = red_color

    black_color = [0, 0, 0]  # 黑色
    for idx in static_vertices:
        colors[idx] = black_color

    # 将颜色分配给网格
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    
    o3d.io.write_triangle_mesh(filename, mesh)

# def plt_mesh(vertices, faces, handle_vertices, static_vertices, filename):
#     # 创建一个新的图形和3D坐标轴
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')

#     # 绘制所有的面
#     polys = [vertices[face] for face in faces]
#     ax.add_collection3d(Poly3DCollection(polys, facecolors='cyan', linewidths=1, edgecolors='r', alpha=.25))

#     # 高亮特定的顶点，例如顶点0、1和2


#     for v in handle_vertices:
#         ax.scatter(*vertices[v], color='blue', s=20)  # s参数是点的大小
    
#     for v in static_vertices:
#         ax.scatter(*vertices[v], color='black', s=20)  # s参数是点的大小

#     # 绘制其他的顶点
#     non_highlighted_vertices = set(range(len(vertices))) - set(handle_vertices) - set(static_vertices)
#     for v in non_highlighted_vertices:
#         ax.scatter(*vertices[v], color='red', s=1)

#     # 设置坐标轴的标签
#     # ax.set_xlabel('X')
#     # ax.set_ylabel('Y')
#     # ax.set_zlabel('Z')
#     X = vertices[:, 0]
#     Y = vertices[:, 1]
#     Z = vertices[:, 2]
#     max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() / 2.0
#     mid_x = (X.max()+X.min()) * 0.5
#     mid_y = (Y.max()+Y.min()) * 0.5
#     mid_z = (Z.max()+Z.min()) * 0.5
#     ax.set_xlim(mid_x - max_range, mid_x + max_range)
#     ax.set_ylim(mid_y - max_range, mid_y + max_range)
#     ax.set_zlim(mid_z - max_range, mid_z + max_range)
#     # 保存图形
#     plt.savefig(filename)
#     plt.close()

def plt_mesh(vertices, faces, handle_vertices, static_vertices, filename):
    # 创建一个新的图形和3D坐标轴
    fig = plt.figure(figsize=(15, 5))
    
    # Subplot for whole mesh
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.set_title('Whole Mesh')
    plot_mesh_on_ax(vertices, ax1)
    plot_vertices_with_arrows_on_ax(vertices, handle_vertices, ax1, 'blue')
    
    # Subplot for handle_vertices
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.set_title('Handle Vertices')
    plot_vertices_on_ax(vertices, handle_vertices, ax2, 'blue')

    
    # Subplot for static_vertices
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.set_title('Static Vertices')
    plot_vertices_on_ax(vertices, static_vertices, ax3, 'black')
    
    # Save figure
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_mesh_on_ax(vertices, ax):
    ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], color='red', s=1)
    set_ax_limits(vertices, ax)

def plot_vertices_on_ax(vertices, vertex_indices, ax, color):
    for v in vertex_indices:
        ax.scatter(*vertices[v], color=color, s=1)
    set_ax_limits(vertices, ax)

def set_ax_limits(vertices, ax):
    X = vertices[:, 0]
    Y = vertices[:, 1]
    Z = vertices[:, 2]
    max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() / 2.0
    mid_x = (X.max()+X.min()) * 0.5
    mid_y = (Y.max()+Y.min()) * 0.5
    mid_z = (Z.max()+Z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

def plot_vertices_with_arrows_on_ax(vertices, vertex_indices, ax, color):
    for v in vertex_indices:
        ax.scatter(*vertices[v], color=color, s=20)
        
        # Add an arrow for each vertex in handle_vertices
        # Here, I'm assuming you want an arrow of length 0.1 in the x direction.
        arrow_length = 0.1
        # TODO: 方向和力的方向对齐
        ax.quiver(vertices[v, 0], vertices[v, 1], vertices[v, 2],
                  arrow_length, 0, 0, color=color)
    
    set_ax_limits(vertices, ax)


if __name__ == "__main__":
    
    vertices, faces = load_mesh('/share/lxx/deformable_result/deformable_result/model/1/0516_manifold_3000.obj')
    mesh = tetrahedralize(vertices,faces)
    centroids = compute_tetrahedron_centroids(mesh.points, mesh.cells)
    # 将重心保存为PLY文件
    save_centroids_as_ply(centroids, "centroids.ply")




