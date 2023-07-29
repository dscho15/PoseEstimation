import trimesh
import sys

if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("Usage: python script_name.py path_to_obj_file")
        sys.exit(1)

    obj = sys.argv[1]

    # load mesh
    mesh = trimesh.load(obj)

    # select 12 3D points uniformly from the mesh
    n_pts = 10000
    pts = mesh.sample(n_pts)

    # select the the 12 points which are furthest away from each other
    pts, _ = trimesh.points.remove_close(pts, radius=20)

    # sample 12 points from the mesh
    pts = mesh.sample(12)

    print("num of pts sampled: ", len(pts))

    # save selected points to .ply file
    mesh = trimesh.Trimesh(vertices=pts)
    output_file = obj.replace(".ply", "_target.ply")
    mesh.export(output_file)

    print("saved to: ", output_file)
