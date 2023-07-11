from torch_spex.spherical_expansions import VectorExpansion
from torch_spex.structures import structure_to_torch, build_neighborlist, InMemoryNeighborList, collate_nl
import torch
from torch.utils.data import DataLoader

import equistore

import json
import ase.io
import numpy as np

def test_in_memory_neighbor_list():

    with open("tests/data/expansion_coeffs-ethanol1_0-hypers.json", "r") as f:
        hypers = json.load(f)

    frames = ase.io.read('datasets/rmd17/ethanol1.extxyz', ':5')

    dataset = InMemoryNeighborList(frames, 4,
            get_energy=lambda frame: torch.tensor([frame.get_potential_energy()]),
            get_forces=lambda frame: torch.tensor(frame.get_forces()))
    loader = DataLoader(dataset, batch_size=5, collate_fn=collate_nl)
    batch = next(iter(loader))
    assert set(batch.keys()) == {'positions', 'species', 'cell', 'pbc', 'centers', 'pairs', 'cell_shifts', 'energy', 'forces'}
