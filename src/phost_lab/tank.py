"""A glass tank of water with a boat in it: the lab's baseline scene for watching a native run and a custom one side by side.

What it is, plainly: MuJoCo (and MJWarp) have no fluid solver. The water here is a bath of small, frictionless,
lightly damped spheres. It sloshes, piles against the walls and carries a floating hull because the hull rests on the
spheres, not because anything computes buoyancy. It shows the right kind of motion; it is not hydrodynamics.

    tank      glass walls on a sprung slide joint, started 15 cm off-centre so it keeps shuttling and sets the water moving
    water     N small spheres, three slide joints each (so each is a point mass), no friction
    boat      an open hull (0.4 x 0.18 m, deck 5 cm up) on a free joint, with a mast and boom
    sail      a cloth (MuJoCo flex membrane) pinned along the mast and the boom (a flex grid is centred on its pos)
    crates    three rigid boxes on the deck, free to slide and tip as the boat rocks

The sizes are the lab's defaults for a laptop GPU; ``layers`` and ``radius`` change how much water there is.
"""

from __future__ import annotations

TANK = (0.80, 0.40, 0.40)          # inner length, width, height (m)
WALL = 0.02
SAIL_GRID = 8


def water_count(radius: float = 0.035, layers: int = 3) -> int:
    length, width, _ = TANK
    return layers * _columns(length, radius) * _columns(width, radius)


def _columns(span: float, radius: float) -> int:
    return int((span - 2.4 * radius) // (2.05 * radius)) + 1


def tank_xml(radius: float = 0.035, layers: int = 3, sail: bool = True, shuttle: float = 0.15) -> str:
    length, width, height = TANK
    nx, ny = _columns(length, radius), _columns(width, radius)
    bodies = []
    for iz in range(layers):
        for ix in range(nx):
            for iy in range(ny):
                x = -length / 2 + 1.2 * radius + ix * (length - 2.4 * radius) / (nx - 1)
                y = -width / 2 + 1.2 * radius + iy * (width - 2.4 * radius) / (ny - 1)
                z = radius + 0.001 + iz * 2.04 * radius + 0.001 * ((ix + iy) % 2)
                bodies.append(
                    f'<body pos="{x:.4f} {y:.4f} {z:.4f}">'
                    '<joint type="slide" axis="1 0 0"/><joint type="slide" axis="0 1 0"/><joint type="slide" axis="0 0 1"/>'
                    f'<geom class="water" size="{radius}"/></body>')
    surface = radius + 0.001 + (layers - 1) * 2.04 * radius + radius      # top of the highest layer of spheres
    boat_z = surface + 0.003
    t = WALL
    glass = 'rgba=".7 .85 .95 .18"'
    sail_xml = ""
    if sail:
        k = SAIL_GRID - 1
        sail_xml = f"""
      <body name="mast" pos="0 0 0.05">
        <geom type="capsule" fromto="0 0 0 0 0 0.30" size="0.006" rgba=".5 .35 .2 1" mass="0.01"/>
        <geom type="capsule" fromto="0 0 0.06 0.15 0 0.06" size="0.004" rgba=".5 .35 .2 1" mass="0.005"/>
        <flexcomp name="sail" type="grid" dim="2" count="{SAIL_GRID} {SAIL_GRID} 1" spacing="0.02 0.03 0.02"
                  pos="0.078 0 0.17" euler="90 0 0" mass="0.08" radius="0.002" rgba=".96 .96 .9 1">
          <pin gridrange="0 0 0 {k}"/><pin gridrange="0 0 {k} 0"/>
          <contact solref="0.01" condim="1" selfcollide="none"/>
          <elasticity elastic2d="both" poisson="0" thickness="1e-3" young="4e3" damping="0.05"/>
        </flexcomp>
      </body>"""
    hull = ".62 .27 .2 1"
    slick = "0.12 0.005 0.0001"            # low friction: the crates slide about on the deck as the boat rocks
    return f"""<mujoco model="boat_tank">
  <option timestep="0.002" integrator="Euler" cone="pyramidal"/>
  <default>
    <geom friction="0.5 0.005 0.0001" solref="0.005 1" solimp="0.9 0.95 0.001"/>
    <default class="water"><geom type="sphere" density="2000" friction="0 0 0" condim="1" solref="0.008 1" priority="1"
                                 rgba="0.15 0.45 0.9 0.6"/></default>
  </default>
  <worldbody>
    <light pos="0 0 2"/>
    <body name="tank" pos="0 0 0">
      <inertial pos="0 0 0.1" mass="6" diaginertia="0.2 0.4 0.5"/>
      <joint name="shuttle" type="slide" axis="1 0 0" stiffness="700" springref="{shuttle}" damping="1"/>
      <geom type="box" pos="0 0 {-t / 2}" size="{length / 2 + t} {width / 2 + t} {t / 2}" {glass}/>
      <geom type="box" pos="{-length / 2 - t / 2} 0 {height / 2}" size="{t / 2} {width / 2 + t} {height / 2}" {glass}/>
      <geom type="box" pos="{length / 2 + t / 2} 0 {height / 2}" size="{t / 2} {width / 2 + t} {height / 2}" {glass}/>
      <geom type="box" pos="0 {-width / 2 - t / 2} {height / 2}" size="{length / 2} {t / 2} {height / 2}" {glass}/>
      <geom type="box" pos="0 {width / 2 + t / 2} {height / 2}" size="{length / 2} {t / 2} {height / 2}" {glass}/>
    </body>
    {chr(10).join("    " + b for b in bodies).strip()}
    <body name="boat" pos="0 0 {boat_z:.4f}">
      <freejoint name="boat_free"/>
      <geom type="box" pos="0 0 0.025" size="0.2 0.09 0.025" mass="0.06" friction="{slick}" rgba="{hull}"/>
      <geom type="box" pos="0 0.09 0.075" size="0.19 0.005 0.025" mass="0.012" rgba="{hull}"/>
      <geom type="box" pos="0 -0.09 0.075" size="0.19 0.005 0.025" mass="0.012" rgba="{hull}"/>
      <geom type="box" pos="0.2 0 0.075" size="0.005 0.09 0.025" mass="0.012" rgba="{hull}"/>
      <geom type="box" pos="-0.2 0 0.075" size="0.005 0.09 0.025" mass="0.012" rgba="{hull}"/>{sail_xml}
    </body>
    <body name="crate_a" pos="-0.10 0.03 {boat_z + 0.077:.4f}"><freejoint/>
      <geom type="box" size="0.025 0.025 0.025" mass="0.03" friction="{slick}" rgba=".82 .62 .22 1"/></body>
    <body name="crate_b" pos="0.12 -0.04 {boat_z + 0.073:.4f}"><freejoint/>
      <geom type="box" size="0.022 0.03 0.02" mass="0.03" friction="{slick}" rgba=".7 .5 .2 1"/></body>
    <body name="crate_c" pos="-0.15 -0.04 {boat_z + 0.073:.4f}"><freejoint/>
      <geom type="box" size="0.02 0.02 0.02" mass="0.02" friction="{slick}" rgba=".76 .56 .26 1"/></body>
  </worldbody>
</mujoco>
"""
