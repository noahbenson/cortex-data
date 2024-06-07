# -*- coding: utf-8 -*-
################################################################################
# cortexdata/geometry/mesh/_util.py

# Currently this file contains some utilities for the cortexdata mesh library 
# that should be saved somewhere.

# Very useful here:
# https://rodolphe-vaillant.fr/entry/33/curvature-of-a-triangle-mesh-definition-and-computation
def vector_cot(u, v):
    # Note that cot(t) = cos(t)/sin(t) = (u . v) / |u x v|
    # because |u x v| = |u| |v| sin(t) and (u . v) = |u| |v| cos(t).
    dotprod = np.sum(np.asarray(u) * v, axis=0)
    crossprod = np.cross(u, v, axisa=0, axisb=0)
    return dotprod / np.sqrt(np.sum(crossprod**2, axis=-1))
def mixed_voronoi_area(mesh):
    """Calculates and returns the mixed Voronoi area of each vertex."""
    from scipy.sparse import csr_array
    faces = mesh.tess.indexed_faces
    nfaces = faces.shape[1]
    nnodes = mesh.vertex_count
    face_sum_mtx = csr_array(
        (np.ones(nfaces*3, dtype=int),
         (faces.flatten(), np.concatenate([np.arange(nfaces)]*3))),
        shape=(nnodes, nfaces),
        dtype=int)
    return (face_sum_mtx @ mesh.face_areas) / 3
def laplacian_matrix(mesh, operator=True):
    from scipy.sparse import csr_array, diags
    edge_faces = mesh.tess.edge_faces
    n = mesh.vertex_count
    # Find the edges of the mesh (we can't calculate on these)
    ii = np.fromiter(map(lambda x:len(x)==2, mesh.tess.edge_faces), bool)
    fs = np.transpose([fs for (ok,fs) in zip(ii,edge_faces) if ok])
    (u, v) = mesh.tess.indexed_edges[:,ii]
    # Find the position of the node that is neither u nor v in each face
    faces = mesh.tess.indexed_faces
    (faces0, faces1) = (faces[:,fs[0]], faces[:,fs[1]])
    fpos_w0 = np.argmin((faces0 == u) | (faces0 == v), axis=0)
    fpos_w1 = np.argmin((faces1 == u) | (faces1 == v), axis=0)
    rng = np.arange(len(fs[0]))
    w0 = faces0[fpos_w0, rng]
    w1 = faces1[fpos_w1, rng]
    xyz = mesh.coordinates
    (xyz_u, xyz_v, xyz_w0, xyz_w1) = (xyz[:,u], xyz[:,v], xyz[:,w0], xyz[:,w1])
    cots0 = vector_cot(xyz_u - xyz_w0, xyz_v - xyz_w0)
    cots1 = vector_cot(xyz_u - xyz_w1, xyz_v - xyz_w1)
    cot_weight = (cots0 + cots1) / 2
    offdiag_mtx = csr_array(
        (np.concatenate([cot_weight, cot_weight]),
         (np.concatenate([u, v]),
          np.concatenate([v, u]))),
        shape=(n, n))
    mtx = offdiag_mtx - diags(offdiag_mtx.sum(axis=0))
    if operator:
        mtx = diags(1 / mixed_voronoi_area(mesh)) @ mtx
    return mtx
def laplacian_smoothing_matrix(mesh, weights=None):
    from scipy.sparse import csr_array, diags
    (u,v) = mesh.tess.indexed_edges
    n = mesh.vertex_count
    if weights is None:
        vals = np.ones(len(u)*2)
    else:
        vals = np.concatenate([weights, weights])
    mtx = csr_array(
        (vals,
         (np.concatenate([u,v]),
          np.concatenate([v,u]))),
        shape=(n, n))
    return diags(1 / mtx.sum(axis=1)) @ mtx
def mesh_laplacian_smoothing(mesh, prop, steps=1, weights=None):
    if steps < 0:
        raise ValueError("negative smoothing steps requested")
    elif steps > 0:
        smmtx = laplacian_smoothing_matrix(mesh, weights=weights)
        for _ in range(steps):
            prop = smmtx @ prop
    return prop
# With smoothing=4, this produces curvature values that are very similar if not
# identical to those produced by freesurfer (FreeSurfer uses 4 steps of
# Laplacian smoothing, but I don't know if they implement it in precisely the
# same way.
def mean_curvature(mesh, smoothing=None):
    """Calculates and returns the mean curvature H at each mesh vertex."""
    # The mean curvature vector is the laplacian (Δp) of the position
    # vector (p), and the mean curvature itself is s/2 |Δp| where s is the
    # dot product of the surface normal at p and -Δp.
    lpcmtx = laplacian_matrix(mesh, operator=True)
    xyz = mesh.coordinates
    lpcxyz = np.zeros_like(xyz)
    for d in range(3):
        lpcxyz[d,:] = lpcmtx @ xyz[d]
    # lpcxyz is now Δp, which is twice the mean curvature 
    H_abs = np.sqrt(np.sum(lpcxyz**2, axis=0)) / 2
    H_sgn = np.sign(np.sum(lpcxyz * mesh.vertex_normals, axis=0))
    H = H_abs * H_sgn
    if smoothing:
        H = mesh_laplacian_smoothing(mesh, H, steps=smoothing)
    return H
# This function seems to produce reasonable answers but has never been tested
# against any ground truth.
def gaussian_curvature(mesh, smoothing=None):
    """Calculates and returns the Gaussian curvature K at each mesh vertex."""
    from scipy.sparse import csr_array
    # First, calculate the sums of the angles at each vertex.
    faces = mesh.tess.indexed_faces
    nfaces = faces.shape[1]
    nnodes = mesh.vertex_count
    corner_sum_mtx = csr_array(
        (np.ones(nfaces * 3),
         (faces.flatten(), np.arange(nfaces * 3))),
        shape=(nnodes, nfaces*3),
        dtype=int)
    angle_sums = corner_sum_mtx @ mesh.face_angles.flatten()
    # From here, we apply the formula:
    K = (2*np.pi - angle_sums) / mixed_voronoi_area(mesh)
    if smoothing:
        K = mesh_laplacian_smoothing(mesh, K, steps=smoothing)
    return K
