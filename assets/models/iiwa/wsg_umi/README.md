# WSG 50 body with UMI soft fingers

`wsg50_umi_fingers.sdf`: the Schunk WSG 50 body of `../wsg_finray/` (gltf
visual, 0.146 x 0.0725 x 0.05 collision box, link at y = -0.049133 in the
model frame) carrying two UMI-3D soft fingers on prismatic joints along
the body's x axis (joint 0 = closed, contact faces touching at x = 0;
q > 0 opens; width = q_left + q_right; limits 0..0.055 m, the WSG 50-110
stroke). The finger base plane sits at y = 0.014052 and the tip plane at
y = 0.137367 in the model frame = y = 0.1865 on the `body` link, which is
where the directives put `wsg::tool` (UMI's TCP is its finger tip). The
default opening is set in the directives (`default_joint_positions`:
0.0453 per finger = 0.0906 m, the maximum `gripper_width` of
cup_in_the_wild; dynamic_tossing's is 0.0826). The benchmark never moves
the fingers (they are inactive plant positions held at that default by
`backend.full_q`), so the scorer and the methods check collisions at the
full opening; the replay sets them per frame from the episode's recorded
`gripper_width`.

## Finger mesh

`meshes/umi_finger_source.stl` is, byte for byte,
`3d_print_STL/soft_finger_TPU_95A/UMI-3D - soft_gripper_finger (1)-1.STL`
of https://github.com/Physical-Intelligence-Laboratory/UMI-3D-Hardware
(Apache-2.0), folder commit `b57c1e38f13c6c1b8c1336ccd88cad020131eaa1`
(2026-03-21), downloaded 2026-10-24.
sha256 `ca03cd67fa9185a9bf1d7a372abbbf975f837fb0f3c72275da32a23cb42dfd64`;
the sibling `(2)-1.STL` (`3668d9d8d8a294681288cdc7db5144ffd57778e0ae77c1d5b9474228778941d6`)
is its mirror image (the pair as assembled, contact faces 76.8 mm apart)
and is not shipped: the finger is symmetric across its width, so one
mesh serves both sides.

Measured with trimesh (binary STL, mm, assembly coordinates): 764
vertices, 1600 triangles, watertight, volume 47 027 mm^3, bounds x
[125.154, 166.808], y [167.294, 290.609], z [150.45, 176.25]; extents
41.654 (depth) x 123.315 (length) x 25.8 (width) mm. The length runs
along y (base at y_min, tip at y_max); the depth profile is 32 mm at the
base, 39 mm 12 mm up, tapering to 1.7 mm at the tip; the width is a
straight extrusion. The flat contact face is the large -x face (3339
mm^2, x = 125.2 at the tip, bulging to 128.5 at the base); the mesh has
no mounting foot (the holders are separate STLs in the release).

`meshes/umi_finger.obj` is that mesh in metres in the canonical finger
frame F, a proper rotation (det +1) plus translation:
x_F = (x - 125.154) / 1000 in [0, 0.0417] (contact face at 0, body
toward +x), y_F = (z - 163.35) / 1000 in +-0.0129 (width, centred),
z_F = -(y - 167.294) / 1000 in [-0.123315, 0] (tip at -0.123315, base
plane at 0). Single object, triangles only, with vertex normals (Drake's
VTK renderer needs them), no texture; the
convex hull has 92 vertices. The SDF rotates it onto each finger link
(left: (x, y, z) -> (-x, -z, -y); right: (x, y, z) -> (x, -z, y)) so the
contact face lies at the link origin's x and the tip points along +y.
Collision geometry is the hull (`drake:declare_convex`, what MuJoCo and
Pinocchio/coal use too); visual is the mesh.
