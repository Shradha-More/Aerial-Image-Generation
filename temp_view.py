import os
from vedo import Plotter, Points, Mesh
import open3d as o3d
import numpy as np

def load_geometries(output_folder="outputs"):
    pcd_points, mesh_data = None, None

    cloud_path = os.path.join(output_folder, "cloud.ply")
    mesh_path = os.path.join(output_folder, "mesh.obj")

    if os.path.isfile(cloud_path):
        print(f"Loading point cloud: {cloud_path}")
        pcd = o3d.io.read_point_cloud(cloud_path)
        pcd_points = np.asarray(pcd.points)

    if os.path.isfile(mesh_path):
        print(f"Loading mesh: {mesh_path}")
        mesh_o3d = o3d.io.read_triangle_mesh(mesh_path)
        mesh_o3d.compute_vertex_normals()
        mesh_vertices = np.asarray(mesh_o3d.vertices)
        mesh_faces = np.asarray(mesh_o3d.triangles)
        mesh_data = (mesh_vertices, mesh_faces)

    return pcd_points, mesh_data

def visualize_overlay(pcd_points, mesh_data):
    if pcd_points is None and mesh_data is None:
        print("No point cloud or mesh found in outputs folder.")
        return

    plt = Plotter(title="3D Reconstruction Overlay", interactive=True)

    pc_obj = None
    if pcd_points is not None:
        pc_obj = Points(pcd_points, c='green', r=3)
        plt.add(pc_obj)

    mesh_obj = None
    if mesh_data is not None:
        mesh_vertices, mesh_faces = mesh_data
        mesh_obj = Mesh([mesh_vertices, mesh_faces], c='red', alpha=0.5)
        plt.add(mesh_obj)

    # Callback to toggle point cloud visibility
    if pc_obj is not None:
        def toggle_pc_key(event):
            if event.key.lower() == "p":
                pc_obj.visible(not pc_obj.visible())
                plt.render()

        plt.add_callback("KeyPress", toggle_pc_key)

    print("Press 'p' to toggle point cloud visibility on/off.")
    plt.show(interactive=True)

if __name__ == "__main__":
    pcd_points, mesh_data = load_geometries("outputs")
    visualize_overlay(pcd_points, mesh_data)
