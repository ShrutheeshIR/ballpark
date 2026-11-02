import io
import json

import jax.numpy as jnp
import jaxls
import numpy as np
import trimesh
import yourdfpy

from ballpark import Robot, cuboidize
from ballpark._cuboid import cuboid_sdf, make_cuboid, transform_cuboid
from ballpark._refine import SphereVar, _sphere_cuboid_cost
from ballpark._spherize import Sphere

URDF = """<robot name="r">
  <link name="base"><collision><geometry><box size="0.4 0.4 0.1"/></geometry></collision></link>
  <link name="a"><collision><origin xyz="0 0 0.2"/><geometry><box size="0.1 0.1 0.4"/></geometry></collision></link>
  <link name="b"><collision><origin xyz="0 0 0.2"/><geometry><box size="0.1 0.1 0.4"/></geometry></collision></link>
  <link name="c"><collision><geometry><cylinder radius="0.05" length="0.3"/></geometry></collision></link>
  <joint name="j1" type="revolute"><parent link="base"/><child link="a"/><origin xyz="0.15 0 0"/><axis xyz="0 1 0"/><limit lower="-1" upper="1" effort="1" velocity="1"/></joint>
  <joint name="j2" type="revolute"><parent link="base"/><child link="b"/><origin xyz="-0.15 0 0"/><axis xyz="0 1 0"/><limit lower="-1" upper="1" effort="1" velocity="1"/></joint>
  <joint name="j3" type="revolute"><parent link="a"/><child link="c"/><origin xyz="0 0 0.4"/><axis xyz="0 1 0"/><limit lower="-1" upper="1" effort="1" velocity="1"/></joint>
</robot>"""


def _robot() -> Robot:
    urdf = yourdfpy.URDF.load(io.StringIO(URDF), load_collision_meshes=True)
    return Robot(urdf)


def _covered(points, cuboids):
    cov = np.zeros(len(points), dtype=bool)
    for c in cuboids:
        cov |= np.asarray(cuboid_sdf(points, c.center, c.half_extents, c.wxyz)) <= 1e-6
    return cov


def test_box_is_one_tight_cuboid():
    box = trimesh.creation.box(extents=[1.0, 0.4, 0.2])
    cuboids = cuboidize(box, 4)
    assert len(cuboids) == 1
    np.testing.assert_allclose(
        np.sort(np.asarray(cuboids[0].half_extents)), [0.1, 0.2, 0.5], rtol=0.05
    )


def test_l_shape_is_covered_with_budget():
    l_shape = trimesh.util.concatenate(
        [
            trimesh.creation.box(extents=[1, 0.2, 0.2]),
            trimesh.creation.box(extents=[0.2, 0.2, 1]).apply_translation([0.4, 0, 0.5]),
        ]
    )
    cuboids = cuboidize(l_shape, 6)
    assert 2 <= len(cuboids) <= 6
    assert _covered(l_shape.sample(3000), cuboids).all()


def test_transform_with_mirror_keeps_box_surface():
    c = make_cuboid(
        np.array([0.1, 0.2, 0.3]), np.array([0.3, 0.2, 0.1]), np.eye(3)
    )
    T = np.eye(4)
    T[1, 1] = -1.0
    T[:3, 3] = [1, 2, 3]
    c2 = transform_cuboid(c, T)
    corner = np.asarray(c.center) + np.asarray(c.half_extents)
    corner2 = (T @ np.append(corner, 1.0))[:3]
    sdf = float(cuboid_sdf(corner2[None], c2.center, c2.half_extents, c2.wxyz)[0])
    assert abs(sdf) < 1e-5


def test_sphere_cuboid_cost_is_penetration_depth():
    c = make_cuboid(np.zeros(3), np.ones(3), np.eye(3))
    var = SphereVar(0)
    vals = jaxls.VarValues.make(
        [var.with_value(Sphere(center=jnp.zeros(3), radius=jnp.array(0.8)))]
    )
    cost = _sphere_cuboid_cost(
        var,
        jnp.array([1.0, 0, 0, 0, 1.5, 0, 0]),  # sphere link: x = 1.5
        c.center,
        c.half_extents,
        c.wxyz,
        jnp.array([1.0, 0, 0, 0, 0, 0, 0]),
        1.0,
    )
    res = jaxls.LeastSquaresProblem([cost], [var]).analyze().compute_residual_vector(vals)
    np.testing.assert_allclose(np.asarray(res), [0.3], atol=1e-5)


def test_robot_mixed_modes_refine_and_json(tmp_path):
    robot = _robot()
    modes = {"base": "cuboid", "a": "cuboid", "c": "sphere"}
    res = robot.spherize(target_spheres=12, link_modes=modes)

    assert res.link_cuboids["base"] and not res.link_spheres["base"]
    assert res.link_spheres["c"] and "c" not in res.link_cuboids
    # b is identical to a (mirrored/translated): follows a's primitive type.
    assert res.link_cuboids.get("b") and not res.link_spheres["b"]

    refined = robot.refine(res)
    assert refined.num_cuboids == res.num_cuboids
    assert len(refined.link_spheres["c"]) == len(res.link_spheres["c"])

    path = tmp_path / "out.json"
    refined.save_json(path)
    data = json.loads(path.read_text())
    assert set(data["a"]["cuboids"]) == {"centers", "half_extents", "wxyz"}
    assert data["c"].keys() == {"centers", "radii"}  # old schema for sphere links


def test_auto_mode_prefers_cuboid_for_boxes_and_sphere_for_spheres():
    urdf = yourdfpy.URDF.load(
        io.StringIO(
            """<robot name="r">
  <link name="box"><collision><geometry><box size="0.5 0.1 0.1"/></geometry></collision></link>
  <link name="ball"><collision><geometry><sphere radius="0.2"/></geometry></collision></link>
  <joint name="j" type="fixed"><parent link="box"/><child link="ball"/></joint>
</robot>"""
        ),
        load_collision_meshes=True,
    )
    res = Robot(urdf).spherize(allocation={"box": 1, "ball": 1}, default_mode="auto")
    assert res.link_cuboids.get("box") and not res.link_spheres["box"]
    assert res.link_spheres["ball"] and "ball" not in res.link_cuboids
