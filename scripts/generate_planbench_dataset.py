#!/usr/bin/env python3
"""Generate simple PlanBench-style problems and plans for three domains."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import List

BLOCKSWORLD_DOMAIN = """(define (domain blocksworld)
  (:requirements :strips)
  (:predicates
    (on ?x ?y)
    (on-table ?x)
    (clear ?x)
    (holding ?x)
    (handempty)
  )

  (:action pick-up
    :parameters (?x)
    :precondition (and (clear ?x) (on-table ?x) (handempty))
    :effect (and (holding ?x) (not (on-table ?x)) (not (clear ?x)) (not (handempty)))
  )

  (:action put-down
    :parameters (?x)
    :precondition (holding ?x)
    :effect (and (on-table ?x) (clear ?x) (handempty) (not (holding ?x)))
  )

  (:action stack
    :parameters (?x ?y)
    :precondition (and (holding ?x) (clear ?y))
    :effect (and (on ?x ?y) (clear ?x) (handempty) (not (holding ?x)) (not (clear ?y)))
  )

  (:action unstack
    :parameters (?x ?y)
    :precondition (and (on ?x ?y) (clear ?x) (handempty))
    :effect (and (holding ?x) (clear ?y) (not (on ?x ?y)) (not (clear ?x)) (not (handempty)))
  )
)"""

MYSTERY_DOMAIN = """(define (domain mystery-bw)
  (:requirements :strips)
  (:predicates
    (rel ?x ?y)
    (grounded ?x)
    (free ?x)
    (holding ?x)
    (idle)
  )

  (:action lift
    :parameters (?x)
    :precondition (and (free ?x) (grounded ?x) (idle))
    :effect (and (holding ?x) (not (grounded ?x)) (not (free ?x)) (not (idle)))
  )

  (:action drop
    :parameters (?x)
    :precondition (holding ?x)
    :effect (and (grounded ?x) (free ?x) (idle) (not (holding ?x)))
  )

  (:action bind
    :parameters (?x ?y)
    :precondition (and (holding ?x) (free ?y))
    :effect (and (rel ?x ?y) (free ?x) (idle) (not (holding ?x)) (not (free ?y)))
  )
)"""

LOGISTICS_DOMAIN = """(define (domain logistics-mini)
  (:requirements :strips)
  (:predicates
    (at ?pkg ?loc)
    (at-vehicle ?veh ?loc)
    (in ?pkg ?veh)
    (airport ?loc)
  )

  (:action load-truck
    :parameters (?pkg ?trk ?loc)
    :precondition (and (at ?pkg ?loc) (at-vehicle ?trk ?loc))
    :effect (and (in ?pkg ?trk) (not (at ?pkg ?loc)))
  )

  (:action unload-truck
    :parameters (?pkg ?trk ?loc)
    :precondition (and (in ?pkg ?trk) (at-vehicle ?trk ?loc))
    :effect (and (at ?pkg ?loc) (not (in ?pkg ?trk)))
  )

  (:action drive
    :parameters (?trk ?from ?to)
    :precondition (at-vehicle ?trk ?from)
    :effect (and (at-vehicle ?trk ?to) (not (at-vehicle ?trk ?from)))
  )

  (:action load-plane
    :parameters (?pkg ?pln ?loc)
    :precondition (and (airport ?loc) (at ?pkg ?loc) (at-vehicle ?pln ?loc))
    :effect (and (in ?pkg ?pln) (not (at ?pkg ?loc)))
  )

  (:action unload-plane
    :parameters (?pkg ?pln ?loc)
    :precondition (and (in ?pkg ?pln) (at-vehicle ?pln ?loc))
    :effect (and (at ?pkg ?loc) (not (in ?pkg ?pln)))
  )

  (:action fly
    :parameters (?pln ?from ?to)
    :precondition (and (airport ?from) (airport ?to) (at-vehicle ?pln ?from))
    :effect (and (at-vehicle ?pln ?to) (not (at-vehicle ?pln ?from)))
  )
)"""


def write_file(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.strip() + "\n", encoding="utf-8")


def generate_blocksworld(root: Path, count: int) -> None:
    domain_dir = root / "blocksworld"
    shutil.rmtree(domain_dir, ignore_errors=True)
    (domain_dir / "problems").mkdir(parents=True, exist_ok=True)
    (domain_dir / "plans_valid").mkdir(parents=True, exist_ok=True)
    (domain_dir / "plans_corrupt").mkdir(parents=True, exist_ok=True)
    (domain_dir / "traces").mkdir(parents=True, exist_ok=True)
    write_file(domain_dir / "domain.pddl", BLOCKSWORLD_DOMAIN)

    for idx in range(1, count + 1):
        n_blocks = 4 + (idx % 5)  # 4..8 blocks
        blocks = [f"b{i}" for i in range(1, n_blocks + 1)]
        objects = " ".join(blocks)

        init_atoms = ["(handempty)"]
        init_atoms += [f"(on-table {b})" for b in blocks]
        init_atoms += [f"(clear {b})" for b in blocks]

        goal_atoms = [f"(on {blocks[i]} {blocks[i+1]})" for i in range(n_blocks - 1)]

        problem = f"""(define (problem bw-{idx:04d})
  (:domain blocksworld)
  (:objects {objects})
  (:init
    {'\n    '.join(init_atoms)}
  )
  (:goal
    (and
      {'\n      '.join(goal_atoms)}
    )
  )
)"""
        write_file(domain_dir / "problems" / f"p{idx:04d}.pddl", problem)

        plan_steps: List[str] = []
        for i in range(n_blocks - 2, -1, -1):
            block = blocks[i]
            target = blocks[i + 1]
            plan_steps.append(f"(pick-up {block})")
            plan_steps.append(f"(stack {block} {target})")
        write_file(domain_dir / "plans_valid" / f"p{idx:04d}.txt", "\n".join(plan_steps))


def generate_mystery(root: Path, count: int) -> None:
    domain_dir = root / "mystery_bw"
    shutil.rmtree(domain_dir, ignore_errors=True)
    (domain_dir / "problems").mkdir(parents=True, exist_ok=True)
    (domain_dir / "plans_valid").mkdir(parents=True, exist_ok=True)
    (domain_dir / "plans_corrupt").mkdir(parents=True, exist_ok=True)
    (domain_dir / "traces").mkdir(parents=True, exist_ok=True)
    write_file(domain_dir / "domain.pddl", MYSTERY_DOMAIN)

    for idx in range(1, count + 1):
        n_things = 4 + (idx % 5)
        things = [f"t{i}" for i in range(1, n_things + 1)]
        objects = " ".join(things)

        init_atoms = ["(idle)"]
        init_atoms += [f"(grounded {t})" for t in things]
        init_atoms += [f"(free {t})" for t in things]

        goal_atoms = [f"(rel {things[i]} {things[i+1]})" for i in range(n_things - 1)]

        problem = f"""(define (problem mystery-{idx:04d})
  (:domain mystery-bw)
  (:objects {objects})
  (:init
    {'\n    '.join(init_atoms)}
  )
  (:goal
    (and
      {'\n      '.join(goal_atoms)}
    )
  )
)"""
        write_file(domain_dir / "problems" / f"p{idx:04d}.pddl", problem)

        plan_steps: List[str] = []
        for i in range(n_things - 2, -1, -1):
            thing = things[i]
            target = things[i + 1]
            plan_steps.append(f"(lift {thing})")
            plan_steps.append(f"(bind {thing} {target})")
        write_file(domain_dir / "plans_valid" / f"p{idx:04d}.txt", "\n".join(plan_steps))


def generate_logistics(root: Path, count: int) -> None:
    domain_dir = root / "logistics"
    shutil.rmtree(domain_dir, ignore_errors=True)
    (domain_dir / "problems").mkdir(parents=True, exist_ok=True)
    (domain_dir / "plans_valid").mkdir(parents=True, exist_ok=True)
    (domain_dir / "plans_corrupt").mkdir(parents=True, exist_ok=True)
    (domain_dir / "traces").mkdir(parents=True, exist_ok=True)
    write_file(domain_dir / "domain.pddl", LOGISTICS_DOMAIN)

    for idx in range(1, count + 1):
        n_packages = 2 + (idx % 3)  # 2..4 packages
        packages = [f"pkg{i}" for i in range(1, n_packages + 1)]
        objects = " ".join(packages + ["truck1", "plane1", "loc0", "airportA", "airportB"])

        init_atoms = ["(airport airportA)", "(airport airportB)"]
        init_atoms.append("(at-vehicle truck1 loc0)")
        init_atoms.append("(at-vehicle plane1 airportA)")
        init_atoms += [f"(at {pkg} loc0)" for pkg in packages]

        goal_atoms = [f"(at {pkg} airportB)" for pkg in packages]

        problem = f"""(define (problem logistics-{idx:04d})
  (:domain logistics-mini)
  (:objects {objects})
  (:init
    {'\n    '.join(init_atoms)}
  )
  (:goal
    (and
      {'\n      '.join(goal_atoms)}
    )
  )
)"""
        write_file(domain_dir / "problems" / f"p{idx:04d}.pddl", problem)

        plan_steps: List[str] = []
        for pkg in packages:
            plan_steps.append(f"(load-truck {pkg} truck1 loc0)")
            plan_steps.append("(drive truck1 loc0 airportA)")
            plan_steps.append(f"(unload-truck {pkg} truck1 airportA)")
            plan_steps.append(f"(load-plane {pkg} plane1 airportA)")
            plan_steps.append("(fly plane1 airportA airportB)")
            plan_steps.append(f"(unload-plane {pkg} plane1 airportB)")
            plan_steps.append("(fly plane1 airportB airportA)")
            plan_steps.append("(drive truck1 airportA loc0)")
        write_file(domain_dir / "plans_valid" / f"p{idx:04d}.txt", "\n".join(plan_steps))


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate PlanBench-style dataset")
    parser.add_argument("--root", type=Path, default=Path("data/planbench_public"))
    parser.add_argument("--count", type=int, default=100)
    args = parser.parse_args()

    root = args.root
    root.mkdir(parents=True, exist_ok=True)

    print(f"Generating dataset in {root} (count={args.count})")
    generate_blocksworld(root, args.count)
    generate_mystery(root, args.count)
    generate_logistics(root, args.count)
    print("Done.")


if __name__ == "__main__":
    main()
