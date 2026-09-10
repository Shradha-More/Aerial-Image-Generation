import os
import numpy as np
import open3d as o3d

# Paths
OUT_DIR       = "outputs"
DEPTH_PATH    = os.path.join(OUT_DIR, "depth_norm.npy")
IMAGE_PATH    = os.path.join(OUT_DIR, "deblurred.png")
PCD_PATH      = os.path.join(OUT_DIR, "cloud.ply")
MESH_PATH     = os.path.join(OUT_DIR, "mesh.obj")

# Camera intrinsics (replace with your real values if known)
fx, fy = 500.0, 500.0  # focal lengths in pixels
cx, cy = 320.0, 240.0  # optical center in pixels (image center for 640x480)

def main():
    # Load depth and RGB
    depth_norm = np.load(DEPTH_PATH)
    color_raw = o3d.io.read_image(IMAGE_PATH)

    # Convert normalized depth to metric (scale up to e.g., 5m for demo)
    depth_meters = (depth_norm * 5.0).astype(np.float32)  
    depth_raw = o3d.geometry.Image(depth_meters)

    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_raw, depth_raw,
        depth_scale=1.0,  # already in meters
        depth_trunc=5.0,  # truncate beyond 5m
        convert_rgb_to_intensity=False
    )

    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        width=depth_norm.shape[1],
        height=depth_norm.shape[0],
        fx=fx, fy=fy, cx=cx, cy=cy
    )

    # Create point cloud
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic)
    o3d.io.write_point_cloud(PCD_PATH, pcd)
    print(f"[INFO] Saved point cloud: {PCD_PATH}")

    # Estimate normals for mesh
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    # Poisson surface reconstruction
    mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8)
    mesh.compute_vertex_normals()

    # Save mesh
    o3d.io.write_triangle_mesh(MESH_PATH, mesh)
    print(f"[INFO] Saved mesh: {MESH_PATH}")

if __name__ == "__main__":
    main()
