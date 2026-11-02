"""Decompose a robot URDF into spheres e.g., for collision checking."""

from __future__ import annotations

from typing import Literal
from pathlib import Path

import tyro
import yourdfpy
from loguru import logger
from robot_descriptions.loaders.yourdfpy import load_robot_description

from ballpark import Robot


def main(
    robot_name: Literal["ur5", "panda", "yumi", "g1"] = "panda",
    target_spheres: int = 40,
    output_path: Path = Path("spheres.json"),
    refine: bool = False,
    default_mode: Literal["sphere", "cuboid", "auto"] = "sphere",
    cuboid_links: tuple[str, ...] = (),
    sphere_links: tuple[str, ...] = (),
) -> None:
    """Compute sphere decomposition for a robot URDF.

    Args:
        robot_name: Robot name from robot_descriptions (e.g., panda).
        target_spheres: Target sphere count across robot.
        output_path: Output JSON file path.
        refine: If True, optimize sphere positions and radii using gradient descent.
        default_mode: Primitive for links not listed below ("auto" picks the tighter
            of spheres/cuboids per link).
        cuboid_links: Links to approximate with cuboids instead of the default.
        sphere_links: Links to approximate with spheres instead of the default.

    Examples:
        python scripts/spherize_robot.py --robot-name panda
        python scripts/spherize_robot.py --robot-name panda --refine
        python scripts/spherize_robot.py --default-mode auto
        python scripts/spherize_robot.py --cuboid-links panda_link2 panda_link4
    """
    # Load URDF.
    # You could alternatively load from a file directly:
    # urdf_obj = yourdfpy.URDF.load(urdf, load_collision_meshes=True)
    logger.info("Loading URDF...")

    urdf_obj = load_robot_description(f"{robot_name}_description")
    urdf_obj = yourdfpy.URDF(
        robot=urdf_obj.robot,
        filename_handler=urdf_obj._filename_handler,
        load_collision_meshes=True,
    )

    # Create Robot and compute spheres
    logger.info("Analyzing robot structure...")
    robot = Robot(urdf_obj)

    logger.info(f"Computing spheres (target={target_spheres})...")
    link_modes = {name: "sphere" for name in sphere_links}
    link_modes.update({name: "cuboid" for name in cuboid_links})
    unknown = set(link_modes) - set(robot.collision_links)
    if unknown:
        raise ValueError(f"Unknown links: {sorted(unknown)}")
    result = robot.spherize(
        target_spheres=target_spheres,
        link_modes=link_modes,  # type: ignore[arg-type]
        default_mode=default_mode,
    )

    # Optionally refine spheres
    if refine:
        logger.info("Refining spheres...")
        result = robot.refine(result)

    # Export to JSON
    result.save_json(output_path)
    logger.info(
        f"Exported to {output_path} (spheres={result.num_spheres}, "
        f"cuboids={result.num_cuboids})"
    )


if __name__ == "__main__":
    tyro.cli(main)
