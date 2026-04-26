#!/usr/bin/env python3
"""Visualize reconstructed PLY meshes as rendered images.

Usage:
    # Single mesh
    python scripts/visualize_mesh.py path/to/mesh.ply -o output.png

    # Compare two experiments side by side
    python scripts/visualize_mesh.py path/to/baseline.ply path/to/improved.ply \
        --labels "Baseline" "PE+CatEmb" -o comparison.png

    # Render all meshes in a reconstruction directory
    python scripts/visualize_mesh.py --dir experiments/artifacts/baseline/Reconstructions/100/Meshes \
        -o renders/ --limit 10
"""

import argparse
import glob
import os
import sys

import numpy as np
import trimesh
import trimesh.transformations as tf


def render_mesh_to_image(mesh, resolution=(800, 800), angle=(45, 30)):
    """Render a mesh to an image using trimesh's built-in renderer.

    Args:
        mesh: trimesh.Trimesh object
        resolution: (width, height) of output image
        angle: (azimuth, elevation) in degrees for camera

    Returns:
        numpy array (H, W, 4) RGBA image, or None if rendering fails
    """
    try:
        scene = trimesh.Scene(mesh)
        # Set camera angle
        az, el = np.radians(angle[0]), np.radians(angle[1])
        camera_transform = tf.rotation_matrix(az, [0, 1, 0]) @ tf.rotation_matrix(el, [1, 0, 0])
        scene.set_camera(angles=angle, distance=2.0)

        png_data = scene.save_image(resolution=resolution)
        if png_data is None:
            return None

        from PIL import Image
        import io
        image = np.array(Image.open(io.BytesIO(png_data)))
        return image
    except Exception as e:
        print(f"  Rendering failed (need display or pyglet): {e}")
        print(f"  Try: pip install pyglet pyrender, or use MeshLab/Open3D instead")
        return None


def save_mesh_screenshot_matplotlib(mesh, output_path, title=None):
    """Fallback: render mesh vertices as a 3D scatter plot using matplotlib."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Subsample vertices for speed
    verts = mesh.vertices
    if len(verts) > 50000:
        idx = np.random.choice(len(verts), 50000, replace=False)
        verts = verts[idx]

    ax.scatter(verts[:, 0], verts[:, 1], verts[:, 2],
               s=0.1, c=verts[:, 1], cmap='coolwarm', alpha=0.5)

    # Equal aspect ratio
    max_range = (verts.max(axis=0) - verts.min(axis=0)).max() / 2
    mid = verts.mean(axis=0)
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)

    ax.view_init(elev=20, azim=45)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    if title:
        ax.set_title(title)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def compare_meshes(mesh_paths, labels, output_path):
    """Render multiple meshes side by side for comparison."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    n = len(mesh_paths)
    fig = plt.figure(figsize=(8 * n, 8))

    for i, (path, label) in enumerate(zip(mesh_paths, labels)):
        mesh = trimesh.load(path, force='mesh')
        ax = fig.add_subplot(1, n, i + 1, projection='3d')

        verts = mesh.vertices
        if len(verts) > 50000:
            idx = np.random.choice(len(verts), 50000, replace=False)
            verts = verts[idx]

        ax.scatter(verts[:, 0], verts[:, 1], verts[:, 2],
                   s=0.1, c=verts[:, 1], cmap='coolwarm', alpha=0.5)

        max_range = (verts.max(axis=0) - verts.min(axis=0)).max() / 2
        mid = verts.mean(axis=0)
        ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
        ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
        ax.set_zlim(mid[2] - max_range, mid[2] + max_range)

        ax.view_init(elev=20, azim=45)
        ax.set_title(f"{label}\n({mesh.vertices.shape[0]} verts, {mesh.faces.shape[0]} faces)")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved comparison: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize DeepSDF reconstructed meshes")
    parser.add_argument("meshes", nargs="*", help="PLY file path(s)")
    parser.add_argument("-o", "--output", default="mesh_render.png", help="Output image path")
    parser.add_argument("--labels", nargs="*", help="Labels for comparison")
    parser.add_argument("--dir", help="Render all meshes in a directory")
    parser.add_argument("--limit", type=int, default=10, help="Max meshes to render from --dir")
    args = parser.parse_args()

    if args.dir:
        os.makedirs(args.output, exist_ok=True)
        plys = sorted(glob.glob(os.path.join(args.dir, "**/*.ply"), recursive=True))[:args.limit]
        print(f"Rendering {len(plys)} meshes from {args.dir}")
        for ply in plys:
            mesh = trimesh.load(ply, force='mesh')
            name = os.path.splitext(os.path.basename(ply))[0]
            out = os.path.join(args.output, f"{name}.png")
            save_mesh_screenshot_matplotlib(mesh, out, title=name)
    elif len(args.meshes) > 1:
        labels = args.labels or [os.path.basename(m) for m in args.meshes]
        compare_meshes(args.meshes, labels, args.output)
    elif len(args.meshes) == 1:
        mesh = trimesh.load(args.meshes[0], force='mesh')
        save_mesh_screenshot_matplotlib(mesh, args.output,
                                        title=os.path.basename(args.meshes[0]))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
