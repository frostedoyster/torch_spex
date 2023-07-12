import numpy as np
import torch
from typing import Dict, List, Tuple, TypeVar, Callable
import ase

AtomicStructure = TypeVar('AtomicStructure')

class InMemoryNeighborList(torch.utils.data.Dataset):
    def __init__(self,
                 structures : List[AtomicStructure],
                 cutoff:float,
                 get_energy: Callable[AtomicStructure, torch.Tensor] = None,
                 get_forces: Callable[AtomicStructure, torch.Tensor] = None,
                 get_stress: Callable[AtomicStructure, torch.Tensor] = None,
                 get_properties: Callable[AtomicStructure, Dict[str, torch.Tensor]] = None,
                 store_contiguous=False):

        self._data = {}
        self._data['positions'] = []
        self._data['species'] = []
        self._data['cell'] = []
        self._data['pbc'] = []
        self._data['centers'] = []
        self._data['pairs'] = []
        self._data['cell_shifts'] = []

        self._store_contiguous = store_contiguous
        if get_energy is not None:
            self._data['energy'] = []
        if get_forces is not None:
            self._data['forces'] = []
        if get_stress is not None:
            self._data['stress'] = []

        for structure in structures:
            positions_i, species_i, cell_i, pbc_i = structure_to_torch(structure, device='cpu')
            centers_i, pairs_ij, cell_shifts_ij = build_neighborlist(positions_i, cell_i, pbc_i, cutoff)
            if get_energy is not None:
                self._data['energy'].append( get_energy(structure) )
            if get_forces is not None:
                self._data['forces'].append( get_forces(structure) )
                positions_i.requires_grad = True
            if get_stress is not None:
                self._data['stress'].append( get_stress(structure) )
                cell_i.requires_grad = True

            self._data['positions'].append( positions_i )
            self._data['species'].append( species_i )
            self._data['cell'].append( cell_i )
            self._data['pbc'].append( pbc_i )
            self._data['centers'].append( centers_i )
            self._data['pairs'].append( pairs_ij )
            self._data['cell_shifts'].append( cell_shifts_ij )

            if get_properties is not None:
                for key, item in get_properties(structure).values():
                    self._data[key].append(item)

        self.n_structures = len(structures)

        if self._store_contiguous:
            # stacking all lists together to one object and verify contiguity
            raise NotImplemented("Need to do it")

    def __getitem__(self, idx):
        return {key: self._data[key][idx] for key in self._data.keys()}

    def __len__(self):
        return self.n_structures

def collate_nl(data_list):
    return {key: torch.concatenate([data[key] for data in data_list], dim=0) for key in data_list[0].keys()}


def structure_to_torch(structure, device : torch.device = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    dtype is automatically referred from the type in the structure object
    """
    if isinstance(structure, ase.Atoms):
        positions = torch.tensor(structure.positions, device=device)
        species = torch.tensor(structure.numbers, device=device)
        cell = torch.tensor(structure.cell.array, device=device)
        pbc = torch.tensor(structure.pbc, device=device)
        return positions, species, cell, pbc
    else:
        raise ValueError("Unknown atom type. We only support ase.Atoms at the moment.")

def build_neighborlist(positions: torch.Tensor, cell: torch.Tensor, pbc: torch.Tensor, cutoff : float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

    assert positions.device == cell.device
    assert positions.device == pbc.device
    device = positions.device
    # will be replaced with something with GPU support
    pairs_i, pairs_j, cell_shifts = ase.neighborlist.primitive_neighbor_list(
        quantities="ijS",
        positions=positions.detach().cpu().numpy(),
        cell=cell.detach().cpu().numpy(),
        pbc=pbc.detach().cpu().numpy(),
        cutoff=cutoff,
        self_interaction=False,
        use_scaled_positions=False,
    )

    pairs_i = torch.tensor(pairs_i, device=device)
    pairs_j = torch.tensor(pairs_j, device=device)
    cell_shifts = torch.tensor(cell_shifts, device=device)

    pairs = torch.vstack([pairs_i, pairs_j]).T
    centers = torch.arange(len(positions))
    return centers, pairs, cell_shifts

