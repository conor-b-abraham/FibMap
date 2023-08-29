import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis import distances
import sys
import os

# CONTENTS
#       SystemInfo: Calculates and stores important system information for later use
#       angle180: Calculate 180 degree angle between two vectors
#       acute_angle: Calculate acute angle between two vectors
#       relative_path: Get relative path from current directory to a file

class SystemInfo:
    '''
    A class for storing system info and converting between full fibril information and per layer information.

    Attributes
    ----------
    structure : np.array(Str) shape: (n_layers, n_protofilaments)
        An array containing the fibril's segment IDs in the position in which they appear. Only includes non-omitted layers.
    segment_layers : dict(segmentID=layer_number)
        The layer in which each segment belongs. Only includes non-omitted layers.
    segment_protofilaments: dict(segmentID=protofilament_number)
        The protofilament in which each segment appears. Only includes non-omitted layers.
    segment_resids : numpy.ndarray (ag.residues.n_residues); dtype=int
        The resids of the residues of the first segment
    matched_resids : numpy.ndarray (ag.residues.n_residues, 2); dtype=str
        first column contains original resid (from the topology file), second column contains matched residue name and resid (e.g. ARG12).
    backbone_atom_names : np.array(Str)
        The names of the backbone atoms
    terminal_atom_names : np.array(Str)
        The names of the terminal atoms
    atom_info : np.array(Int) shape: (n_atoms, 4)
        Array containing identity information for atoms in the fibril. Each row contains information for the atom whose index matches the index of the row. The following information is included:[[<layer index>, <protofilament index>, <residue index (segment-based)>, <atom type (0=backbone, 1=sidechain, 2=terminus)>], ... ]
    
    Methods
    -------
    get_residue(resid) : Returns the matched name of the residue
    '''
    def __init__(self, ag, n_protofilaments, omit_layers):
        '''
        Parameters
        ----------
        ag : MDAnalysis.atomgroup
            Atom group of the fibril
        n_protofilaments : Int
            The number of protofilaments in the fibril
        omit_layers : Int
            The number of layers to omit from each end of the fibril
        '''
        # CONSTRUCT 'STRUCTURE' ATTRIBUTE
        # ag.segments.segids is sorted by the names of the segids rather than by the segment order. Here, we are gathering information about the order of the segments, so this is obviously a problem. Our solution is to gather the segment names from the residues, find the unique segids (automatically sorted) and then unsort them. 
        all_segids = [ag.residues.segids[i] for i in sorted(np.unique(ag.residues.segids, return_index=True)[1])]

        # According to the rule for segment order (see docs) each protofilament is made up of every n_protofilaments'th segment. 
        segids = [all_segids[pf::n_protofilaments] for pf in range(n_protofilaments)]

        # Force to be 2d so single protofilament case will behave like multiple protofilament case
        if n_protofilaments == 1: 
            segids = np.atleast_2d(segids)
        
        # Throw out omitted layers
        full_structure = np.array(segids).T
        if omit_layers == 0:
            self.structure = np.copy(full_structure)
        else:
            self.structure = np.copy(full_structure)[omit_layers:-omit_layers, :]

        # CONSTRUCT 'SEGMENT_LAYERS' & 'SEGMENT_PROTOFILAMENTS' ATTRIBUTES
        self.segment_layers, self.segment_protofilaments = {}, {}
        for li, layer in enumerate(self.structure):
            for pfi, segment in enumerate(layer):
                self.segment_layers[segment] = li+omit_layers+1 # The layer index (one-based)
                self.segment_protofilaments[segment] = pfi+1 # The protofilament index (one-based)
        
        
        # CONSTRUCT 'MATCHED_RESIDS' ATTRIBUTE
        # future: This needs to consider that the residues in each protofilament could be different (unlikely, but I think it's possible)
        self.segment_resids = ag.segments[0].residues.resids
        nres_per_seg = ag.segments[0].residues.n_residues
        self.matched_resids = np.zeros((ag.residues.n_residues, 2), dtype=int).astype(str)
        for i, res in enumerate(ag.residues):
            self.matched_resids[i, 0] = str(res.resid)
            self.matched_resids[i, 1] = f'{res.resname}{self.segment_resids[i%nres_per_seg]}'

        # CONSTRUCT 'BACKBONE_ATOM_NAMES' ATTRIBUTE
        # This will only include heavy atoms
        self.backbone_atom_names = np.unique(ag.select_atoms('backbone').names)

        # CONSTRUCT 'TERMINAL_ATOM_NAMES' ATTRIBUTE
        def _find_term_atoms(bbag):
            '''
            Find the terminal atoms
            '''
            # Find the atom connected to the next residue
            for atom in bbag:
                for bond in atom.bonds:
                    for bonded_atom in bond.atoms:
                        if bonded_atom.residue.resid != atom.residue.resid:
                            connected_atom = atom
            
            # Follow backbone to the term atom
            bbatoms = bbag.names
            checked_atoms = [connected_atom.name]
            keeplooking = True
            curatom = connected_atom
            while keeplooking:
                foundnext = False
                for bond in curatom.bonds:
                    for bonded_atom in bond.atoms:
                        if bonded_atom.name in bbatoms and bonded_atom.name not in checked_atoms and bonded_atom.name != curatom.name and bonded_atom.resid == curatom.resid:
                            checked_atoms.append(bonded_atom.name)
                            curatom = bonded_atom
                            foundnext = True
                            break
                    if foundnext:
                        break
                if not foundnext:
                    keeplooking = False
            terminal_bb_atom = curatom

            # Collect names of terminal atoms
            def _traverse(atom, collection, ignore_names):
                atomsel = f"(resname {atom.residue.resname} and name {atom.name})"
                if atomsel not in collection and atom.name not in ignore_names:
                    collection.append(atomsel)
                if atomsel in collection:
                    for bond in atom.bonds:
                        for nextatom in bond.atoms:
                            nextatomsel = f"(resname {nextatom.residue.resname} and name {nextatom.name})"
                            if nextatom.name != atom.name and nextatomsel not in collection and nextatom.name not in ignore_names:
                                collection = _traverse(nextatom, collection, ignore_names)
                return collection

            terminal_names = _traverse(terminal_bb_atom, [f"(resname {terminal_bb_atom.residue.resname} and name {terminal_bb_atom.name})"], bbatoms)
            return terminal_names

        terminal_atom_names = []
        for si in range(n_protofilaments):
            terminal_atom_names += _find_term_atoms(ag.segments[si].residues[0].atoms.select_atoms("backbone")) # N-Terminus
            terminal_atom_names += _find_term_atoms(ag.segments[si].residues[-1].atoms.select_atoms("backbone")) # C-Terminus
        self.terminal_atom_names = np.unique(terminal_atom_names)

        # CONSTRUCT 'ATOM INFO' ATTRIBUTE
        self.atom_info = np.zeros((ag.atoms.n_atoms, 4), dtype=int)
        for li, layer in enumerate(full_structure):
            for pfi, segid in enumerate(layer):
                self.atom_info[ag.atoms.segids==segid,0] = li+1 # Column 0 is the layer index
                self.atom_info[ag.atoms.segids==segid,1] = pfi+1 # Column 1 is the protofilament index

        # Column 2 is residue index within its segment(one-based)
        counter = 0
        for i, residue in enumerate(ag.residues):
            for _ in residue.atoms:
                self.atom_info[counter,2] = (i%nres_per_seg) + 1 
                counter += 1

        # Column 3 is atom type (0 for backbone, 1 for sidechain, 2 for terminus)
        term_residues = []
        for segment in ag.segments:
            term_residues.append(segment.residues[0].resid)
            term_residues.append(segment.residues[-1].resid)

        self.atom_info[:,3] = np.ones(ag.atoms.n_atoms) # Sidechain atoms are marked with a 1
        agatoms = [f"(resname {atom.residue.resname} and name {atom.name})" for atom in ag.atoms]
        self.atom_info[np.isin(agatoms, self.backbone_atom_names),3] = 0 # Backbone atoms are marked with a 0
        self.atom_info[np.isin(agatoms, self.terminal_atom_names) & np.isin(ag.atoms.resids, np.unique(term_residues)),3] = 2 # Terminus atoms are marked with a 2 (if residue is a terminus residue)

    def get_residue(self, resid):
        '''
        Return the matched name of residue with a given resid

        Parameters
        ----------
        resid : int
            Original resid of residue
        
        Returns
        -------
        resname : str
            Matched residue name and resid (e.g. ARG12)
        '''
        if type(resid) != str: # Convert integer to string
            resid = str(resid)

        return self.matched_resids[:, 1][self.matched_resids[:,0]==resid][0]
    
def angle180(a, b):
    '''
    Measure the 180 degree angle between two unit vectors

    Parameters
    ----------
    a, b : np.ndarray
        The two unit vectors to measure the angle between. Must be normalized.
    
    Returns
    -------
    angle : float
        The angle between the unit vectors a and b
    '''
    return np.rad2deg(np.arccos(np.clip(a.dot(b[:3]), -1.0, 1.0)))

def acute_angle(a, b):
    '''
    Measure the acute angle between two unit vectors
        
    Parameters
    ----------
    a, b : np.ndarray
        The two unit vectors to measure the angle between. Must be normalized.
    
    Returns
    -------
    angle : float
        The acute angle between the unit vectors a and b
    '''
    angle = angle180(a, b)
    if angle > 90:
        angle = 180 - angle
    return angle

def relative_path(path):
    '''
    Return relative path to file or directory with None handler and Windows separate paths handler

    Parameters
    ----------
    path : str or list
        the absolute path or a list of absolute paths you want to convert

    Returns
    -------
    path : str or list
        the relative path or list of relative paths
    '''
    if path is None:
        path = None
    if type(path) == list:
        paths = []
        for p in path:
            try:
                paths.append(os.path.relpath(p))
            except ValueError: 
                # On windows, relpath will raise ValueError if current directory and path are on separate drives
                # In the future, change this so that it won't catch every Value Error
                paths.append(p)
        path = paths
    else:
        try:
            path = os.path.relpath(path)
        except ValueError: 
            # On windows, relpath will raise ValueError if current directory and path are on separate drives
            # In the future, change this so that it won't catch every Value Error
            path = path
    return path
            
            


