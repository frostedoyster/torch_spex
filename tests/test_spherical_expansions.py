import json

import pytest

import torch

import equistore
import numpy as np
import ase.io

from torch_spex.spherical_expansions import VectorExpansion, SphericalExpansion
from torch_spex.structures import structure_to_torch, build_neighborlist 
from equistore import TensorMap, TensorBlock, Labels

class TestSphericalExpansion:
    device = "cpu"
    frame = ase.io.read('datasets/rmd17/ethanol1.extxyz', ':1')[0]
    all_species = np.unique(frame.numbers)
    with open("tests/data/expansion_coeffs-ethanol1_0-hypers.json", "r") as f:
        hypers = json.load(f)

    position, species, cell, pbc = structure_to_torch(frame)
    centers, pairs, cell_shifts = build_neighborlist(position, cell, pbc, hypers["cutoff radius"])

    def test_vector_expansion_coeffs(self):
        tm_ref = sort_tm(equistore.core.io.load_custom_array("tests/data/vector_expansion_coeffs-ethanol1_0-data.npz", equistore.core.io.create_torch_array))
        vector_expansion = VectorExpansion(self.hypers, self.all_species, device="cpu")
        with torch.no_grad():
            tm = sort_tm(vector_expansion.forward(self.position, self.species, self.cell, self.cell_shifts, self.centers, self.pairs))
        # default types are float32 so we set accuracy to 1e-7
        assert equistore.operations.allclose(tm_ref, tm, atol=1e-7, rtol=1e-7)

    def test_spherical_expansion_coeffs(self):
        tm_ref = equistore.core.io.load_custom_array("tests/data/spherical_expansion_coeffs-ethanol1_0-data.npz", equistore.core.io.create_torch_array)
        spherical_expansion_calculator = SphericalExpansion(self.hypers, self.all_species)
        with torch.no_grad():
            tm = spherical_expansion_calculator.forward(self.position, self.species, self.cell, self.cell_shifts, self.centers, self.pairs)
        # default types are float32 so we set accuracy to 1e-7
        assert equistore.operations.allclose(tm_ref, tm, atol=1e-7, rtol=1e-7)

    def test_spherical_expansion_coeffs_alchemical(self):
        with open("tests/data/expansion_coeffs-ethanol1_0-alchemical-hypers.json", "r") as f:
            hypers = json.load(f)
        centers, pairs, cell_shifts = build_neighborlist(self.position, self.cell, self.pbc, hypers["cutoff radius"])

        tm_ref = equistore.core.io.load_custom_array("tests/data/spherical_expansion_coeffs-ethanol1_0-alchemical-seed0-data.npz", equistore.core.io.create_torch_array)
        torch.manual_seed(0)
        spherical_expansion_calculator = SphericalExpansion(hypers, self.all_species)
        with torch.no_grad():
            tm = spherical_expansion_calculator.forward(self.position, self.species, self.cell, cell_shifts, centers, pairs)
        # default types are float32 so we set accuracy to 1e-7
        assert equistore.operations.allclose(tm_ref, tm, atol=1e-7, rtol=1e-7)


### these util functions will be removed once lab-cosmo/equistore/pull/281 is merged
def native_list_argsort(native_list):
    return sorted(range(len(native_list)), key=native_list.__getitem__)

def sort_tm(tm):
    blocks = []
    for _, block in tm.items():
        values = block.values

        samples_values = block.samples.values
        sorted_idx = native_list_argsort([tuple(row.tolist()) for row in block.samples.values])
        samples_values = samples_values[sorted_idx]
        values = values[sorted_idx]

        components_values = []
        for i, component in enumerate(block.components):
            component_values = component.values
            sorted_idx = native_list_argsort([tuple(row.tolist()) for row in component.values])
            components_values.append( component_values[sorted_idx] )
            values = np.take(values, sorted_idx, axis=i+1)

        properties_values = block.properties.values
        sorted_idx = native_list_argsort([tuple(row.tolist()) for row in block.properties.values])
        properties_values = properties_values[sorted_idx]
        values = values[..., sorted_idx]

        blocks.append(
            TensorBlock(
                values=values,
                samples=Labels(values=samples_values, names=block.samples.names),
                components=[Labels(values=components_values[i], names=component.names) for i, component in enumerate(block.components)],
                properties=Labels(values=properties_values, names=block.properties.names)
            )
        )
    return TensorMap(keys=tm.keys, blocks=blocks)
