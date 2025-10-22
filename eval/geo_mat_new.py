import os
import argparse
import numpy as np
import trimesh
import networkx as nx
import open3d as o3d
import scipy.io as sio
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from sklearn import neighbors
import time

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def compute_geodesic_distmat(verts: np.ndarray, faces: np.ndarray, nn_k: int = 500) -> np.ndarray:
    """
    Compute an approximate geodesic distance matrix using Dijkstra on a
    graph formed by mesh vertex adjacencies weighted by Euclidean distances
    to the nn_k nearest neighbors.

    Args:
        verts: (N, 3) float array of vertex positions
        faces: (F, 3) int array of triangle indices
        nn_k: number of nearest neighbors for distance weights

    Returns:
        (N, N) float array of geodesic distances (np.inf if disconnected)
    """
    # Build vertex adjacency from mesh connectivity
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    vertex_graph = mesh.vertex_adjacency_graph
    if not nx.is_connected(vertex_graph):
        print('[geo_mat_scape_r] Warning: mesh graph not connected; distances across components will be inf.')
    adj_mat = nx.adjacency_matrix(vertex_graph, range(verts.shape[0]))

    # Euclidean distances between nn_k neighbors (sparse)
    knn_graph = neighbors.kneighbors_graph(verts, n_neighbors=nn_k, mode='distance', include_self=False)

    # Keep distances only on existing mesh edges (element-wise mask)
    distance_adj = csr_matrix((verts.shape[0], verts.shape[0])).tolil()
    distance_adj[adj_mat != 0] = knn_graph[adj_mat != 0]

    # Dijkstra shortest path on undirected graph
    geodesic = shortest_path(distance_adj.tocsr(), directed=False)
    if np.any(np.isinf(geodesic)):
        print('[geo_mat_scape_r] Info: inf values present in geodesic matrix. Consider increasing --nn.')
    return geodesic


def read_shape_off(file: str):
    """
    Read an OFF mesh from file using Open3D.

    Returns:
        verts: (N, 3) float32
        faces: (F, 3) int32
    """
    mesh = o3d.io.read_triangle_mesh(file)
    verts, faces = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
    return verts, faces


def main():
    parser = argparse.ArgumentParser(description='Compute geodesic distance matrices for SCAPE_R meshes.')
    parser.add_argument('--dataset_root', type=str, default='data/scape_r', help='Root folder of scape_r (contains shapes_train/shapes_test)')
    parser.add_argument('--split', type=str, choices=['train', 'test'], default='test', help='Which split to process')
    parser.add_argument('--nn', type=int, default=500, help='Number of nearest neighbors used to weight edges')
    parser.add_argument('--out_dir', type=str, default=None, help='Output folder for .mat files (default: <dataset_root>/geodist)')
    parser.add_argument('--glob', type=str, default='*.off', help='Filename pattern to include (default: *.off)')
    parser.add_argument('--limit', type=int, default=None, help='Optional limit on number of files to process')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing output files')

    args = parser.parse_args()

    shapes_dir = os.path.join(args.dataset_root, f'shapes_{args.split}')
    out_dir = args.out_dir or os.path.join(args.dataset_root, 'geodist')
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.isdir(shapes_dir):
        raise FileNotFoundError(f'Could not find shapes directory: {shapes_dir}')

    # Collect OFF files
    all_files = [f for f in os.listdir(shapes_dir) if f.endswith('.off')]
    all_files.sort()
    if args.limit is not None:
        all_files = all_files[:args.limit]

    total_files = len(all_files)
    print(f'[geo_mat_scape_r] Processing {total_files} file(s) from {shapes_dir}')
    print(f'[geo_mat_scape_r] nn={args.nn}, output={out_dir}, overwrite={args.overwrite}')
    t0 = time.perf_counter()

    for idx, fname in enumerate(all_files, start=1):
        in_path = os.path.join(shapes_dir, fname)
        stem = os.path.splitext(fname)[0]
        out_path = os.path.join(out_dir, f'{stem}.mat')
        start_t = time.perf_counter()
        print(f'  [{idx}/{total_files}] start {fname}')

        if (not args.overwrite) and os.path.exists(out_path):
            dt = time.perf_counter() - start_t
            print(f'  [{idx}/{total_files}] [skip] {fname} -> exists (took {dt:.2f}s)')
            continue

        try:
            verts, faces = read_shape_off(in_path)
            if verts.size == 0 or faces.size == 0:
                dt = time.perf_counter() - start_t
                print(f'  [{idx}/{total_files}] [warn] {fname}: empty verts/faces, skipping (took {dt:.2f}s)')
                continue
            G = compute_geodesic_distmat(verts, faces, nn_k=args.nn)
            # Save with key compatible with read_geodist (expects "G" or "Gamma")
            sio.savemat(out_path, { 'G': G })
            dt = time.perf_counter() - start_t
            print(f'  [{idx}/{total_files}] [ok] {fname} -> {out_path} (took {dt:.2f}s)')
        except Exception as e:
            dt = time.perf_counter() - start_t
            print(f'  [{idx}/{total_files}] [fail] {fname}: {e} (took {dt:.2f}s)')

    total_dt = time.perf_counter() - t0
    print(f'[geo_mat_scape_r] Done. Processed {total_files} file(s) in {total_dt:.2f}s (avg {(total_dt/max(total_files,1)):.2f}s/file).')


if __name__ == '__main__':
    main()
