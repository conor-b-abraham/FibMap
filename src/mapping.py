import MDAnalysis as mda
from MDAnalysis.analysis import distances, rms
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon, PathPatch, FancyBboxPatch, BoxStyle
from matplotlib.collections import PatchCollection
import matplotlib.path as mpath
import matplotlib as mpl
import os
import multiprocessing
from tqdm import tqdm

# CONTENTS
# Residue : Contains residue information for mapping
# Chain : Contains chain information for mapping
# FibrilMap : Class for constructing the fibril map

# ------------------------------------------------------------------------------
# RESIDUE & CHAIN OBJECTS
# ------------------------------------------------------------------------------
class Residue:
    '''
    To store positions and anchor points of each residue

    Attributes
    ----------
    mdaresidue : MDAnalysis.residue
        Residue from mdanalysis
    resname : str
        Name of residue with 3-letter code and residue number
    longlabel : str
        3-letter code of residue
    label : str
        1-letter code of residue
    resid : int
        Residue index in chain
    charge : float
        Charge of residue
    ca_position : np.array(Float, Float)
        Alpha carbon position on map's 2D plane
    sc_position : np.array(Float, Float)
        Center of mass of the sidechain on map's 2D plane
    chaincolor : str
        Color of chain the residue belong to. Used for outline.
    rescolor : str
        Color of residue. Used for shape fill
    labelcolor : str
        Color of label
    radius : float
        Radius of residue marker fill
    underradius : float
        Radius of residue marker outline
    vertices : np.array(float) [3x2]
        Vertices of triangle of marker for fill
    undervertices : np.array(float) [3x2]
        Vertices of triangle of marker for outline
    sc_anchor : np.array(float, float)
        Anchor point for sidechain contact
    ca_anchors : np.array(float) [3x2] or [4x2]
        Anchor points for backbone contact. 4x2 for glycine, 3x2 for all other residues.
    t_position : np.array(float) [1x2] or None
        A terminus position. None if this residue is not a terminus or if it hasn't been set with set_terminus().
    t_anchor : np.array(float) [1x2] or None
        The position on the residue bead that intersects with the terminus position. None if this residue is not a terminus or if it hasn't been set with set_terminus().

    Methods
    -------
    set_terminus() 
        Set the terminus position
    '''
    def __init__(self, mdaresidue, resname, CA_position, SC_position, radius, underradius, chain_color, 
                 residue_color, residue_label_color, legend_residue=False):
        '''
        Parameters
        ----------
        mdaresidue : MDAnalysis.residue
            Residue from mdanalysis
        resname : Str
            Name of residue with 3-letter code and residue number. For legend residue (legend_residue = True), this should be the residues label or an empty string.
        CA_position : np.array(Float, Float)
            Position of alpha carbon on map's 2D plane
        SC_position : np.array(Float, Float)
            Position of sidechain center of mass on map's 2D plane
        radius : float
            Radius of residue circle on map
        underradius : float
            Radius of residue background circle on map
        chaincolor : Str
            Color of chain to which the residue belongs
        residue_color : Str
            Color of residue marker fill
        residue_label_color : Str
            Color of residue label
        legend_residue : Bool
            To create a dummy residue for legend. Default is false. 
        '''
        # Residue 1 letter codes
        residue_codes = {'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLU': 'E',
                         'GLN': 'Q', 'GLY': 'G', 'HIS': 'H', 'HSD': 'H', 'HSP':'H', 'HSE':'H',
                         'ILE': 'I', 'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
                         'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'}
        
        # Residue Identity
        if not legend_residue:
            self.mdaresidue = mdaresidue
            self.resname = resname
            self.longlabel = resname[:3]
            self.label = residue_codes[self.longlabel]
            self.resid = int(resname[3:])
        else:
            self.label = resname

        # Residue Colors
        self.chaincolor = chain_color
        self.rescolor = residue_color
        if residue_label_color == "chain":
            self.labelcolor = chain_color
        else:
            self.labelcolor = residue_label_color

        # Residue Position
        self.ca_position = CA_position
        self.sc_position = SC_position

        # Vertices' Positions
        def _get_vertices(radius):
            resvector = self.sc_position - self.ca_position
            resvector = resvector / np.linalg.norm(resvector)
            vertices = [self.ca_position+(resvector*radius*2)] # The tip of the triangle
            R_1 = np.array([[np.cos(np.pi/3), -np.sin(np.pi/3)], [np.sin(np.pi/3), np.cos(np.pi/3)]])
            R_2 = np.array([[np.cos(-np.pi/3), -np.sin(-np.pi/3)], [np.sin(-np.pi/3), np.cos(-np.pi/3)]])
            vertices.append(self.ca_position+(R_1.dot(resvector)*radius))
            vertices.append(self.ca_position+(R_2.dot(resvector)*radius))
            return vertices
        
        self.radius = radius
        self.underradius = underradius
        self.vertices = _get_vertices(self.radius)
        self.undervertices = _get_vertices(self.underradius)

        # Anchors' Positions
        self.sc_anchor = self.undervertices[0]
        resvector = self.sc_anchor-self.ca_position
        resvector = (resvector/np.linalg.norm(resvector))*underradius
        self.ca_anchors = [self.undervertices[1:], self.ca_position-resvector]
        if not legend_residue and self.label == 'G': # if it's Glycine, also add a backbone anchor to the sidechain side of the bead
            self.ca_anchors.append(self.ca_position+resvector)
        self.ca_anchors = np.vstack(self.ca_anchors)
    
    def set_terminus(self, position):
        '''
        Tell this residue that it's a terminus

        Parameters
        ----------
        position : np.array(float) [1x2]
            The position of the terminus
        '''
        self.t_position = position

        anchor = self.t_position - self.ca_position
        self.t_anchor = self.ca_position + ((anchor/np.linalg.norm(anchor))*self.underradius)

class Chain:
    '''
    To store information about a chain on the map

    Attributes
    ----------
    residues : list(Residue) or None
        List of the residues in the chain or None to set later
    nt_position : np.array(Float, Float)
        Position of N terminus or None
    ct_position : np.array(Float, Float)
        Position of C terminus or None
    ca_positions : np.array(Float) [n_residues x 2]
        Position of all alpha carbons
    n_residues : int
        Number of residues in the chain
    color : str
        Color of chain or None

    Methods
    -------
    add_residue(Residue)
        Add a new residue to the chain
    add_nt_position(position)
        Add or change the chain's n-terminus position
    add_ct_position(position)
        Add or chaing the chain's c-terminus position
    '''
    def __init__(self, residues=[], nt_position=None, ct_position=None, color='black'):
        '''
        Parameters
        ----------
        residues : list(Residue) or list()
            List of the residues in the chain or empty list to add to later
        nt_position : np.array(Float, Float) or None
            Position of N terminus or None to set later
        ct_position : np.array(Float, Float) or None
            Position of C terminus or None or None to set later
        color : str [Default = 'black']
            Color of chain
        '''
        self.residues = residues
        self.nt_position = nt_position
        self.ct_position = ct_position
        self.color = color
        if self.residues is not None:
            self.n_residues = len(self.residues)
        else:
            self.n_residues = 0

        self.ca_positions = []
        if len(self.residues) != 0:
            for residue in self.residues:
                self.ca_positions.append(residue.ca_position)
            self.ca_positions = np.array(self.ca_positions)

    def add_residue(self, residue):
        '''
        Add a residue to the chain

        Parameters
        ----------
        residue : Residue
            residue to add
        '''
        self.residues.append(residue)

        if len(self.ca_positions) != 0:
            self.ca_positions = np.vstack((self.ca_positions, residue.ca_position))
        else:
            self.ca_positions = residue.ca_position
        
        self.n_residues += 1
    
    def add_nt_position(self, position):
        '''
        Set the N terminus position

        Parameters
        ----------
        position : np.array(Float, Float)
            Position of the n-terminus
        '''
        self.nt_position = position
        self.residues[0].set_terminus(position)
    
    def add_ct_position(self, position):
        '''
        Set the C terminus position

        Parameters
        ----------
        position : np.array(Float, Float)
            Position of the c-terminus
        '''
        self.ct_position = position
        self.residues[-1].set_terminus(position) 

# ------------------------------------------------------------------------------
# FIBRIL MAP
# ------------------------------------------------------------------------------
def _normal(ag):
    '''
    find normal vector to plane of best fit of atom group
    '''
    positions = ag.positions.T
    position_mean = np.mean(positions, axis=1, keepdims=True) # mean as column vector

    normal = np.linalg.svd(positions-position_mean)[0][:, -1]
    normal = -normal if normal[1] < 0 else normal # point normal up in the y-direction
    normal = normal/np.linalg.norm(normal)

    return normal

def _rotmat(a, b):
    '''
    Calculate rotation matrix to rotate vector a to vector b
    '''
    v = np.cross(a, b)
    s = np.linalg.norm(v)
    c = np.dot(a, b)
    I = np.identity(3)
    k = np.matrix([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    R = I + k + np.matmul(k,k) * ((1 -c)/(s**2))
    return np.asarray(R)

def _angle_180(v1, v2):
    '''
    Calculate angle from vector 1 to vector 2 in degrees (0 to 180)
    '''
    uv1 = v1 / np.linalg.norm(v1)
    uv2 = v2 / np.linalg.norm(v2)
    angle = np.arccos(np.clip(np.dot(uv1, uv2), -1.0, 1.0))
    return angle

def _align(mobile_fit, mobile_all, reference_fit, masses=None):
    """
    RMSD fit mobile_all using the mobile_fit atoms to reference_fit using the kabsch algorithm.
    """

    # Center reference coordinates
    reference_fit -= np.average(reference_fit, axis=0, weights=masses)

    # Center mobile coordinates
    mobile_com = np.average(mobile_fit, axis=0, weights=masses)
    mobile_fit -= mobile_com
    mobile_all -= mobile_com

    # Kabsch Algorithm
    h = mobile_fit.T @ reference_fit # Cross-Covariance Matrix
    (u, s, vt) = np.linalg.svd(h) # Singular Value Decomposition
    v = vt.T # Numpy's svd function returns the transpose of v, so transpose it back
    d = np.sign(np.linalg.det(v @ u.T)) # Correct rotation for right-handed coordinate system
    mat = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, d]])
    rot = v @ mat @ u.T # find rotation matrix
    mobile_all = mobile_all @ rot.T # apply rotation matrix
    mobile_fit = mobile_fit @ rot.T

    # Calculate final RMSD
    rmsd = rms.rmsd(mobile_fit, reference_fit, superposition=False)

    return mobile_all, rmsd

def _positions_mp(layer_indices, segids, TOP, TRAJ, reference_positions):
    '''
    Function to parallelize mapping of fibril onto 2D plane
    '''
    u = mda.Universe(TOP, TRAJ)
    CA_positions = np.zeros((segids.shape[1], u.select_atoms(f"segid {segids.flatten()[0]}").residues.n_residues, 2))
    SC_positions = np.zeros_like(CA_positions)
    Cterm_positions = np.zeros((segids.shape[1], 2))
    Nterm_positions = np.zeros_like(Cterm_positions)
    RMSDs = np.zeros((segids.shape[0], u.trajectory.n_frames))

    for li in layer_indices:
        sel = f"segid {' '.join(segids[li,:])}"
        layer = u.select_atoms(sel)
        layerCA = layer.select_atoms("name CA") # Just the alpha carbons
        # Collect Alpha Carbon, Side Chain, C-Terminus and N-Terminus Positions
        u.trajectory[0]
        for ts in tqdm(u.trajectory, leave=False):
            # RMSD fit layer to reference layer
            layer.positions, RMSDs[li, ts.frame] = _align(layerCA.positions, layer.positions, reference_positions)

            CAp, SCp, Ctermp, Ntermp = [], [], [], []
            for segment in layer.segments:
                CA = segment.atoms.select_atoms("name CA")
                CAp.append(CA.positions[:, :2])
                res_scpos = []
                for residue in segment.residues:
                    SC = residue.atoms.select_atoms('not name C CA N O HA HN HT1 HT2 HT3 OT1 OT2 HT2B') # Just the sidechains
                    if SC.n_atoms != 0:
                        res_scpos.append(SC.center_of_mass()[:2])
                    else:
                        res_scpos.append([0.0, 0.0])
                SCp.append(np.array(res_scpos))
                Ctermp.append(segment.residues[-1].atoms.select_atoms('name OT1 OT2').center_of_mass()[:2])
                Ntermp.append(segment.residues[0].atoms.select_atoms('name N HT1 HT2 HT3').center_of_mass()[:2])
            CA_positions += np.array(CAp)
            SC_positions += np.array(SCp)
            Cterm_positions += np.array(Ctermp)
            Nterm_positions += np.array(Ntermp)
    return {"CA":CA_positions, "SC":SC_positions, "CT":Cterm_positions, "NT":Nterm_positions, "RMSDs":RMSDs}

def _residue_marker(residue):
    '''
    Create residue marker object

    Parameters
    ----------
    residue : Residue
        residue to create a marker for
    
    Returns
    -------
    circle : matplotlib.patches.Circle
        The circular part of the marker fill
    triangle : matplotlib.patches.Polygon
        The triangular part of the marker fill
    undercircle : matplotlib.patches.Circle
        The circular part of the marker outline
    undertriangle : matplotlib.patches.Polygon
        The triangular part of the marker outline
    '''
    circle = Circle((residue.ca_position[0], residue.ca_position[1]), residue.radius, color=residue.rescolor)
    triangle = Polygon(residue.vertices, color=residue.rescolor)
    undercircle = Circle((residue.ca_position[0], residue.ca_position[1]), residue.underradius, color=residue.chaincolor)
    undertriangle = Polygon(residue.undervertices, color=residue.chaincolor)
    return circle, triangle, undercircle, undertriangle

class FibrilMap:
    '''
    Class for fibril map

    Attributes
    ----------
    fig : matplotlib.figure.Figure
        figure where map is being created. Populated by make_chain() method
    ax : matplotlib.axes.Axes
        axis where map is being created. Populated by make_chain() method
    chains : None or list[Chain]
        Populated by make_chain() method, otherwise None. Contains information about the residues making up each chain.

    Methods
    -------
    make_chain(OUTDIR)
        Creates chains of residues in a layer and generates additional internal
        attributes used for all other methods.
    shade_regions(resnames, region_file=None)
        Shades hydrophobic zippers and water channels according to contents of region_file.
        If region_file is None, user will be prompted to provide region information
    add_hydrogen_bonds(hbatoms, residues, cutoff)
        Add hydrogen bonds to map
    add_salt_bridges(saltbridges, residues, cutoff)
        Add salt bridges to map
    add_pipi_interactions(pipis, residues, cutoff)
        Add pipi interactions to map
    legend()
        Add legend to map
    save(OUTDIR, show=True)
        Save fibril map image and optionally, show the image
    '''
    def __init__(self, params, system_info):
        '''
        Parameters
        ----------
        params : io.Params
            Parameters for fibril mapping run
        system_info : utils.SystemInfo
            System Information
        '''
        self.__top = params.topology_file
        self.__traj = params.trajectory_file
        self.__sysinfo = system_info
        self.__u = mda.Universe(self.__top, self.__traj)
        if self.__u.trajectory.n_frames < params.nprocs:
            self.__nprocs = 1
        else:
            self.__nprocs = params.nprocs
        self.__figsize = (params.figure_width, params.figure_height)
        self.__dpi = params.figure_dpi
        self.__legend_items = {
            "main": {"water":False, "zipper":False},
            "hbond":{"section":False, "both":False, "same":False, "backbone":False},
            "sb":   {"section":False, "both":False},
            "pipi": {"section":False, "both":False, "same":False}
        }
        self.chains = None # Populated by make_chains() method

        # Keep params around for later
        self.__params = params

    # -------------------------------- RESIDUES AND CHAINS --------------------------------
    def _chain_positions(self):
        '''
        Get positions of alpha carbons and sidechain coms for each residue
        '''
        # SET REFERENCE LAYER
        refu = mda.Merge(self.__u.select_atoms("segid "+" ".join(self.__sysinfo.structure[int(np.floor(self.__sysinfo.structure.shape[0]/2)), :].tolist())))
        reflayer = refu.atoms
        original_positions = reflayer.positions

        # Rotate Reference Layer to be aligned with the xy plane
        refCA = reflayer.select_atoms("name CA")
        normal = _normal(refCA)
        centered_positions = original_positions - np.mean(refCA.positions, axis=0)
        reflayer.positions = _rotmat(normal, np.array([0, 0, 1])).dot(centered_positions.T).T

        # Rotate Reference Layer so that longest distance is aligned with the x axis
        # Could replace this with SVD
        layer_dists = distances.self_distance_array(refCA.positions)
        s_layer_dists = np.zeros((refCA.n_atoms, refCA.n_atoms))
        counter = 0
        for i in range(refCA.n_atoms):
            for j in range(i+1, refCA.n_atoms):
                s_layer_dists[i, j] = layer_dists[counter]
                counter += 1
        atom1, atom2 = np.where(s_layer_dists==np.max(s_layer_dists))
        dist_vector = refCA.positions[atom1[0]]-refCA.positions[atom2[0]]
        dist_vector = dist_vector/np.linalg.norm(dist_vector)
        centered_positions = reflayer.positions - np.mean(refCA.positions, axis=0)
        reflayer.positions = _rotmat(dist_vector, np.array([1, 0, 0])).dot(centered_positions.T).T
        reference_positions = refCA.positions
        
        # DETERMINE POSITIONS
        CA_positions = np.zeros((self.__sysinfo.structure.shape[1], self.__u.select_atoms(f"segid {self.__sysinfo.structure.flatten()[0]}").residues.n_residues, 2))
        SC_positions = np.zeros_like(CA_positions)
        Cterm_positions = np.zeros((self.__sysinfo.structure.shape[1], 2))
        Nterm_positions = np.zeros_like(Cterm_positions)
        RMSDs = np.zeros((self.__sysinfo.structure.shape[0], self.__u.trajectory.n_frames))
        
        if self.__nprocs != 1:
            if self.__nprocs > self.__sysinfo.structure.shape[0]:
                self.__nprocs = self.__sysinfo.structure.shape[0]
            split_layers = np.array_split(np.arange(self.__sysinfo.structure.shape[0]), self.__nprocs)
            with multiprocessing.Pool(self.__nprocs) as p:
                mp_args = [(i, self.__sysinfo.structure, self.__top, self.__traj, reference_positions) for i in split_layers]
                results = p.starmap(_positions_mp, mp_args)
            p.close()
            p.join()
            for r in results:
                CA_positions += r["CA"]
                SC_positions += r["SC"]
                Cterm_positions += r["CT"]
                Nterm_positions += r["NT"]
                RMSDs += r["RMSDs"]
        else:
            layer_inds = list(range(self.__sysinfo.structure.shape[0]))
            results = _positions_mp(layer_inds, self.__sysinfo.structure, self.__top, self.__traj, reference_positions)
            CA_positions = results['CA']
            SC_positions = results['SC']
            Cterm_positions = results['CT']
            Nterm_positions = results['NT']
            RMSDs = results['RMSDs']
        
        print(f"  Mean RMSD of fitted layers: {np.round(np.mean(RMSDs),3)}\u212B")

        # Get averages by dividing by the number of frames times the number of layers
        CA_positions /= (self.__u.trajectory.n_frames*self.__sysinfo.structure.shape[0])
        SC_positions  /= (self.__u.trajectory.n_frames*self.__sysinfo.structure.shape[0])
        Cterm_positions /= (self.__u.trajectory.n_frames*self.__sysinfo.structure.shape[0])
        Nterm_positions /= (self.__u.trajectory.n_frames*self.__sysinfo.structure.shape[0])

        return CA_positions, SC_positions, Cterm_positions, Nterm_positions
    
    def _get_residue_colors(self, residue_name, residue_charge):
        '''
        Get Residue and Label Color

        Parameters
        ----------
        residue_name : Str
            3-letter code for name of residue
        residue_charge : Int or Float
            Charge of residue
        
        Returns
        -------
        rescolor : Str
            Color of residue marker fill
        labelcolor : Str
            Color of residue label

        Raises
        ------
        SystemExit
            If the residue type cannot be determined
        '''
        # Residue Type Definitions
        residue_types = {'ALA': 'NP', 'ARG': 'P', 'ASN': 'P', 'ASP': 'P', 'CYS': 'P', 'GLU': 'P',
                         'GLN': 'P', 'GLY': 'NP', 'HIS': 'P', 'HSP':'P', 'HSD': 'NP', 'HSE':'NP',
                         'ILE': 'NP', 'LEU': 'NP', 'LYS': 'P', 'MET': 'NP', 'PHE': 'NP', 'PRO': 'NP',
                         'SER': 'P', 'THR': 'P', 'TRP': 'NP', 'TYR': 'P', 'VAL': 'NP'}
        # Probably should implement a pH dependent way of determining polarity (particularly for histidine)
        if residue_name not in residue_types.keys():
            raise SystemExit(f"Error: unknown residue {residue_name}: undefined residue detected, known residues are: {' '.join(list(residue_types.keys()))}")
        
        # Find Residue Colors
        if residue_charge == -1.0: # Negatively Charged (Acidic)
            rescolor = self.__params.acidic_color
            labelcolor = self.__params.acidic_label_color
        elif residue_charge == 1.0: # Positively Charged (Basic)
            rescolor = self.__params.basic_color
            labelcolor = self.__params.basic_label_color
        elif residue_types[residue_name] == 'P': # Polar
            rescolor = self.__params.polar_color
            labelcolor = self.__params.polar_label_color
        elif residue_types[residue_name] == 'NP': # Non-polar
            rescolor = self.__params.nonpolar_color
            labelcolor = self.__params.nonpolar_label_color
        else:
            raise SystemExit(f"Error: unknown residue {residue_name}: could not determine residue type")
        
        return rescolor, labelcolor
    
    def _init_figure(self, ca_positions, sc_positions, ct_positions, nt_positions):
        '''
        Initialize figure of correct size
        '''
        # First, we need to find the bead radius: Bead radius will be 1/3 the minimum alpha-Carbon alpha-Carbon "bonded" distance
        min_bonded_dist = np.round(np.min([np.sqrt((np.diff(p, axis=0)**2).sum(axis=1)) for p in ca_positions]))
        temp_underradius = 5*min_bonded_dist/12

        # Second, we find the x and y span covered by the residues
        maxx = np.ceil(np.max((ca_positions[:,:,0].max(), sc_positions[:,:,0].max(), nt_positions[:,0].max(), ct_positions[:,0].max())))+(2*temp_underradius)
        minx = np.floor(np.min((ca_positions[:,:,0].min(), sc_positions[:,:,0].min(), nt_positions[:,0].min(), ct_positions[:,0].min())))-(2*temp_underradius)
        maxy = np.ceil(np.max((ca_positions[:,:,1].max(), sc_positions[:,:,1].max(), nt_positions[:,1].max(), ct_positions[:,1].max())))+(2*temp_underradius)
        miny = np.floor(np.min((ca_positions[:,:,1].min(), sc_positions[:,:,1].min(), nt_positions[:,1].min(), ct_positions[:,1].min())))-(2*temp_underradius)
        xspan = maxx - minx
        yspan = maxy - miny
        
        # Find conversion factor to convert from Angstroms to Points
        if self.__params.legend:
            conversion_factor = np.min([((self.__figsize[0]-1.5)*72)/xspan, (self.__figsize[1]*72)/yspan])
        else:
            conversion_factor = np.min([((self.__figsize[0])*72)/xspan, (self.__figsize[1]*72)/yspan])

        # Convert the positions' units
        ca_positions = ca_positions * conversion_factor
        sc_positions = sc_positions * conversion_factor
        nt_positions = nt_positions * conversion_factor
        ct_positions = ct_positions * conversion_factor

        self.fig, self.ax = plt.subplots(figsize=self.__figsize, dpi=self.__dpi)

        if self.__params.legend:
            self.ax.set_xlim(minx*conversion_factor, (maxx*conversion_factor)+(72*1.5))
            self.ax.set_ylim(miny*conversion_factor, maxy*conversion_factor)
            self.__legend_bottomleft = [maxx*conversion_factor, miny*conversion_factor]
            self.__legend_dims = [72*1.5, yspan*conversion_factor]
            self.__legend_1nm = 10*conversion_factor
        else:
            self.ax.set_xlim(minx*conversion_factor, maxx*conversion_factor)
            self.ax.set_ylim(miny*conversion_factor, maxy*conversion_factor)

        # set the radius and underradius of the beads
        self.__underradius = temp_underradius*conversion_factor
        self.__radius = self.__underradius-1

        # Set fontsize
        mpl.rcParams['font.sans-serif'] = "Arial"
        mpl.rcParams['font.family'] = "sans-serif"
        mpl.rcParams['font.size'] = np.round(1.5*self.__radius)

        return ca_positions, sc_positions, ct_positions, nt_positions
    
    def make_chains(self):
        '''
        Create the chains of one layer and add it to the plot

        Populated Attributes
        --------------------
        self.chains : Chain
            contains chain and residue information
        self.fig : matplotlib.figure.Figure
            the figure
        self.ax : matplotlib.axes.Axes
            the axes
        '''
        # ------------------------------------------------------------------------------
        def _termini_marker_vector(lastbb, termpos, underradius):
            # determine vector for termini markers
            vec = termpos-lastbb
            vec = vec/np.linalg.norm(vec)
            vec = vec*underradius*2
            xcoords = np.array([lastbb[0], lastbb[0]+vec[0]])
            ycoords = np.array([lastbb[1], lastbb[1]+vec[1]])
            return xcoords, ycoords

        def _C_termini_marker_direction(lastbb, ctermpos):
            # Determine direction of the C-termini marker
            vec = ctermpos-lastbb
            vec = vec/np.linalg.norm(vec)
            angle = (np.arctan2(vec[1], vec[0]) * 180 / np.pi) - 90
            return angle
        
        def _residue_charge(residue, bb_and_sc):
            if int(np.round(np.sum(residue.atoms.charges))) == 0:
                charge = 0
            else:
                charge = int(np.round(np.sum(bb_and_sc.charges)))
            return charge
        
        def _term_color(self, residue):
            termatoms = residue.mdaresidue.atoms.select_atoms(f"not backbone and name {' '.join(self.__sysinfo.terminal_atom_names)}")
            if int(np.round(np.sum(residue.mdaresidue.atoms.charges))) != 0:
                charge = int(np.round(np.sum(termatoms.charges)))
                if charge == 1.0:
                    color = self.__params.basic_color
                elif charge == -1.0:
                    color = self.__params.acidic_color
                else:
                    color = residue.chaincolor
            else:
                color = residue.chaincolor
            return color

        # ------------------------------------------------------------------------------
        # radius = 24
        # underradius = 28

        if self.__params.map_positions_file is None:
            CA_pos, SC_pos, Cterm_pos, Nterm_pos = self._chain_positions()
        else:
            positions = np.load(self.__params.map_positions_file)
            CA_pos = np.copy(positions["CA"])
            SC_pos = np.copy(positions["SC"])
            Cterm_pos = np.copy(positions["CT"])
            Nterm_pos = np.copy(positions["NT"])
            positions.close()

        # Adjust figure size and convert position units
        CA_positions, SC_positions, Cterm_positions, Nterm_positions = self._init_figure(CA_pos, SC_pos, Cterm_pos, Nterm_pos)
        self.chains = []
        # first_resid = self.__u.residues.resids[0]
        for ci in range(self.__sysinfo.structure.shape[1]):
            newchain = Chain(residues=[], 
                             nt_position=Nterm_positions[ci], 
                             ct_position=Cterm_positions[ci], 
                             color=self.__params.backbone_color[ci%len(self.__params.backbone_color)])
            for ri, residue in enumerate(self.__u.segments[ci].residues):
                matched_residue_name = self.__sysinfo.get_residue(residue.resid)
                residue_color, residue_label_color = self._get_residue_colors(matched_residue_name[:3], _residue_charge(residue, residue.atoms.select_atoms(f"backbone or not name {' '.join(self.__sysinfo.terminal_atom_names)}")))
                newchain.add_residue(Residue(residue,
                                             matched_residue_name, 
                                             CA_positions[ci][ri], 
                                             SC_positions[ci][ri], 
                                             self.__radius, 
                                             self.__underradius, 
                                             newchain.color,
                                             residue_color,
                                             residue_label_color)) # Remove this arg
            self.chains.append(newchain)

        for chain in self.chains:
            # ADD LINE FOR EACH CHAIN
            self.ax.plot(chain.ca_positions[:,0], chain.ca_positions[:,1], color=chain.color, linewidth=2, zorder=6)

            # ADD C & N TERMINI MARKERS
            Cmarkerx, Cmarkery = _termini_marker_vector(chain.ca_positions[-1, :], chain.ct_position, self.__underradius)
            Nmarkerx, Nmarkery = _termini_marker_vector(chain.ca_positions[0, :], chain.nt_position, self.__underradius)

            self.ax.plot(Cmarkerx, Cmarkery, color=chain.color, linewidth=2, solid_capstyle="butt", zorder=6) # Line to C-Terminal Marker
            self.ax.plot(Nmarkerx, Nmarkery, color=chain.color, linewidth=2, solid_capstyle="butt", zorder=6) # Line to N-Terminal Marker

            chain.add_ct_position(np.array([Cmarkerx[1], Cmarkery[1]])) # Update C-terminal position to be position of marker
            chain.add_nt_position(np.array([Nmarkerx[1], Nmarkery[1]])) # Update N-terminal position to be position of marker
            self.ax.plot(chain.ct_position[0], chain.ct_position[1], markeredgecolor=chain.color, markerfacecolor=_term_color(self, chain.residues[-1]), 
                         markersize=4, markeredgewidth=0.5, marker=(3,0,_C_termini_marker_direction(chain.ca_positions[-1, :], chain.ct_position)), zorder=6)
            self.ax.plot(chain.nt_position[0], chain.nt_position[1], markeredgecolor=chain.color, markerfacecolor=_term_color(self, chain.residues[0]), 
                         markersize=4, markeredgewidth=0.5, marker='o', zorder=6)

            # ADD THE RESIDUES
            beadpatches, underbeadpatches = [], []
            for ri, residue in enumerate(chain.residues):
                c, t, uc, ut = _residue_marker(residue)
                beadpatches.append(c)
                underbeadpatches.append(uc)
                if residue.label != "G":
                    beadpatches.append(t)
                    underbeadpatches.append(ut)
                if self.__params.numbered_residues:
                    reslabel = residue.resid
                else:
                    reslabel = residue.label
                self.ax.text(residue.ca_position[0], residue.ca_position[1], reslabel, c=residue.labelcolor, va='center_baseline', ha='center', fontweight='bold', zorder=16)
            self.ax.add_collection(PatchCollection(beadpatches, match_original=True, zorder=14))
            self.ax.add_collection(PatchCollection(underbeadpatches, match_original=True, zorder=10))
        return CA_pos, SC_pos, Cterm_pos, Nterm_pos, mpl.rcParams['font.size']
    
    # -------------------------------- SHADED REGIONS --------------------------------
    def _make_region_path(self, region):
        path, pathstring = [], []
        for (pf, (f, l)) in region:
            fi = np.where(self.__sysinfo.segment_resids==f)[0][0]
            li = np.where(self.__sysinfo.segment_resids==l)[0][0]
            pathstring.append(f'{pf}-{self.chains[pf-1].residues[fi].resname}')
            if f != l:
                pathstring.append(f'{pf}-{self.chains[pf-1].residues[li].resname}')

            path.append(self.chains[pf-1].ca_positions[fi:li+1,:])
        
        pathstring.append(pathstring[0])
        pathstring = " --> ".join(pathstring)
        return np.vstack(path), pathstring

    def add_regions(self):
        '''
        Shade hydrophobic zippers and water channels of fibril.
        '''
        region_patches, region_strings = [], []
        if self.__params.zipper_region is not None:
            for region in self.__params.zipper_region:
                path, pathstring = self._make_region_path(region)
                region_patches.append(Polygon(path, 
                                              edgecolor=None, facecolor=self.__params.zipper_color, alpha=self.__params.zipper_opacity))
                region_strings.append(f"{'Hydrophobic Zipper:':>21} "+pathstring)
                self.__legend_items["main"]["zipper"] = True
                
                
        if self.__params.water_region is not None:
            for region in self.__params.water_region:
                path, pathstring = self._make_region_path(region)
                region_patches.append(Polygon(path, 
                                              edgecolor=None, facecolor=self.__params.water_color, alpha=self.__params.water_opacity))
                region_strings.append(f"{'Water Channel:':>21} "+pathstring)
                self.__legend_items["main"]["water"] = True
                
        self.ax.add_collection(PatchCollection(region_patches, match_original=True, zorder=0))
        return "\n".join(region_strings)

    # -------------------------------- HYDROGEN BONDS --------------------------------
    def _hydrogen_bond(self, dpos=None, dca=None, apos=None, aca=None, interlayer=False, intralayer=False):
        '''
        Drawing utility for hydrogen bonds

        To draw same position hbond: Only provide dpos. Setting interlayer to true is not necessary.
        To draw regular two position hbond: Provide all positions and set interlayer and intralayer kwargs to set style
        To draw adjacent residue backbone-backbone interlayer hbond: Provide dca and aca.

        Parameters
            ----------
            dpos : np.array(float) [1x2]
                Position of donor anchor
            dca : np.array(float) [1x2] or None
                Position of donor alpha carbon. 
            apos : np.array(float) [1x2] or None
                Position of acceptor anchor. 
            aca : np.array(float) [1x2] or None
                Position of acceptor alpha carbon.
            interlayer : Bool
                If true, draw in interlayer style
            intralayer : Bool
                If true, draw in intralayer style
        '''
        # ------------------------------------------------------------------------------
        def _arrow_curve_direction(dpos, dca, apos, aca):
            '''
            Determine if arrow should go clockwise or counter clockwise.

            Parameters
            ----------
            dpos : np.array(float) [1x2]
                Position of donor anchor
            dca : np.array(float) [1x2]
                Position of donor alpha carbon
            apos : np.array(float) [1x2]
                Position of acceptor anchor
            aca : np.array(float) [1x2]
                Position of acceptor alpha carbon
            
            Returns
            -------
            cdir : int
                Either 1 or -1 to inform matplotlib of direction of the curve
            cmag : float
                The magnitude of the curve
            '''
            dvec = dpos-dca
            dvec = dvec/np.linalg.norm(dvec)
            avec = apos-aca
            avec = avec/np.linalg.norm(avec)
            cdir = np.sign(np.cross(dvec, avec))

            if np.all(dca != aca):
                cmag = 0.3*((np.pi-_angle_180(dpos-dca, apos-aca))/np.pi)
            else:
                cmag = -1
            return cdir, cmag
        # ------------------------------------------------------------------------------

        if dca is None and apos is None and aca is None and dpos is not None: # Same position marker
            self.ax.plot(dpos[0], dpos[1], marker='o', markersize=3, linestyle=None, 
                         markerfacecolor=self.__params.hbond_color_2, markeredgecolor=self.__params.hbond_color_1, 
                         markeredgewidth=0.5, zorder=16)
            self.__legend_items["hbond"]["same"] = True 

        elif dca is not None and dpos is not None and aca is not None and apos is not None: # Regular
            cdir, cmag = _arrow_curve_direction(dpos, dca, apos, aca)
            if intralayer: # Anytime there's an intralayer Hbond we draw this arrow
                self.ax.annotate('', xy=dpos, xytext=apos, 
                                 arrowprops=dict(arrowstyle='-', fc='none', ec=self.__params.hbond_color_1, shrinkA=3, shrinkB=0, 
                                                 linestyle="solid", linewidth=1, connectionstyle=f"arc3, rad={cdir*cmag}"), 
                                 zorder=12) # LINE
                if interlayer: # Also interlayer
                    self.ax.annotate('', xy=dpos, xytext=apos, 
                                     arrowprops=dict(arrowstyle='-', fc='none', ec=self.__params.hbond_color_2, shrinkA=3, shrinkB=0, 
                                                     linestyle="dashed", linewidth=0.5, connectionstyle=f"arc3, rad={cdir*cmag}"), 
                                 zorder=12) # DASHED LINE
                    self.__legend_items["hbond"]["both"] = True 
                self.ax.annotate('', xy=dpos, xytext=apos, 
                                 arrowprops=dict(arrowstyle='<|-', fc=self.__params.hbond_color_1, ec=self.__params.hbond_color_1, shrinkA=0, shrinkB=0, 
                                                 linestyle='solid', mutation_scale=9, linewidth=0, connectionstyle=f"arc3, rad={cdir*cmag}"), 
                                 zorder=12) # ARROWHEAD

            elif interlayer: # Intralayer only style (dashed)
                self.ax.annotate('', xy=dpos, xytext=apos, 
                                 arrowprops=dict(arrowstyle='-', fc='none', ec=self.__params.hbond_color_1, shrinkA=3, shrinkB=0, 
                                                 linestyle="dashed", linewidth=1, connectionstyle=f"arc3, rad={cdir*cmag}"),
                                 zorder=12) # LINE
                self.ax.annotate('', xy=dpos, xytext=apos, 
                                 arrowprops=dict(arrowstyle='<|-', fc=self.__params.hbond_color_1, ec=self.__params.hbond_color_1, shrinkA=0, shrinkB=0, 
                                                 linestyle='solid', mutation_scale=9, linewidth=0, connectionstyle=f"arc3, rad={cdir*cmag}"), 
                                 zorder=12) # ARROWHEAD

        elif dpos is None and apos is None and dca is not None and aca is not None:
            self.ax.annotate('', xy=dca, xytext=aca, 
                             arrowprops=dict(arrowstyle='-', fc='none', ec=self.__params.hbond_color_2, shrinkA=0, shrinkB=0, 
                                             linestyle="dotted", connectionstyle=f"arc3, rad=0"), 
                             zorder=6)
            self.__legend_items["hbond"]["backbone"] = True
        self.__legend_items["hbond"]["section"] = True
        
    def add_hydrogen_bonds(self):
        '''
        Add Hydrogen Bonds to Map

        Returns
        -------
        hbond_string : Str
            List of HBonds added to map
        '''
        hbinfo = np.load(self.__params.hb_processed_file)
        # Use either count or probability cutoff
        if self.__params.use_hbond_n_cutoff:
            hbinfo = hbinfo[:,(0,1,2,3,4,5,6,8)]
        else:
            hbinfo = hbinfo[:,(0,1,2,3,4,5,7,9)]
        # Only loop through hbonds that won't be cutoff
        hbinfo = hbinfo[np.any(hbinfo[:,-2:]>self.__params.hbond_cutoff, axis=1),:]
        # Switch to zero-based indexing of protofilaments and residues
        hbinfo[:, (0,1,3,4)] = hbinfo[:, (0,1,3,4)] - 1

        types = ["BB", "SC", "T"]
        hbond_strings = []
        for (dpi, dri, dsi, api, ari, asi), (intraL, interL) in zip(hbinfo[:,:-2].astype(int), hbinfo[:,-2:]):
            # NEED TO ADD ERROR HANDLING FOR CASES IN WHICH IMPROPER SITE IDS WERE USED
            # Get the involved residues
            donor = self.chains[dpi].residues[dri]
            acceptor = self.chains[api].residues[ari]
            dID = f"{dpi+1}-{donor.resname}-{types[dsi]}"
            aID = f"{api+1}-{acceptor.resname}-{types[asi]}"
            intraL_str = f"{np.round(intraL, 3):.3f}"
            interL_str = f"{np.round(interL, 3):.3f}"
            hbondID = f"{dID:>12}{aID:>12}{intraL_str:>10}{interL_str:>12}"
            # Get all of the positions you'll ever need
            if dsi == 0:
                dpos = donor.ca_anchors
            elif dsi == 1:
                dpos = donor.sc_anchor
            elif dsi == 2:
                dpos = donor.t_position
            if asi == 0:
                apos = acceptor.ca_anchors
            elif asi == 1:
                apos = acceptor.sc_anchor
            elif asi == 2:
                apos = acceptor.t_position
            dca = donor.ca_position
            aca = acceptor.ca_position

            if dpi == api and dri == ari and interL > self.__params.hbond_cutoff: # interlayer hbond to same residue
                if dsi == asi: # Interlayer same residue, same position
                    if dsi == 0: # Backbone-Backbone
                        self._hydrogen_bond(dpos=dpos[-1,:])
                    elif dsi == 1 or dsi == 2: # Sidechain-Sidechain (1) or Terminus-Terminus (2)
                        self._hydrogen_bond(dpos=dpos)
                else: # Interlayer same residue, different positions
                    if dsi == 0: # Donor is backbone
                        dpos = dpos[1,:] # Grab anchor position of index 1 (could improve selection here)
                    if asi == 0: # Donor is backbone
                        apos = apos[1,:] # Grab anchor position of index 1 (could improve selection here)
                    self._hydrogen_bond(dpos, dca, apos, aca, interlayer=True)
            elif dpi == api and np.abs(dri-ari) == 1 and interL > self.__params.hbond_cutoff:
                self._hydrogen_bond(dpos=None, dca=dca, apos=None, aca=aca)
            elif dpi != api or dri != ari: # Hbond between different residues (non-adjacent interlayer)
                if dsi == 0 and asi == 0:
                    d_da = np.zeros((dpos.shape[0], apos.shape[0]))
                    for i, d in enumerate(dpos):
                        for j, a in enumerate(apos):
                            d_da[i,j] = np.linalg.norm(a-d)
                    selda = np.where(d_da==d_da.min())
                    dpos = dpos[selda[0][0],:]
                    apos = apos[selda[1][0],:]
                elif dsi == 0:
                    d_da = np.array([np.linalg.norm(apos-d) for d in dpos])
                    dpos = dpos[np.argmin(d_da),:]
                elif asi == 0:
                    d_da = np.array([np.linalg.norm(a-dpos) for a in apos])
                    apos = apos[np.argmin(d_da),:]
                self._hydrogen_bond(dpos, dca, apos, aca, interlayer=interL>self.__params.hbond_cutoff, intralayer=intraL>self.__params.hbond_cutoff)

            else:
                hbondID = f"------------------------------------------------------------------\nWarning: Unknown H-Bond Type:\n{hbondID}\n------------------------------------------------------------------"
            
            hbond_strings.append(hbondID)
        
        return "\n".join(hbond_strings)

    # -------------------------------- SALT BRIDGES --------------------------------
    def _salt_bridge(self, aCA, aCAa1, aCAa2, aSC, cCA, cCAa1, cCAa2, cSC, interlayer, intralayer, I=None, arcscale=3):
        '''
        Create Salt Bridge Shape: Starts at side of one alpha carbon, bezier curve to intersection point between the two sidechains, bezier curve to the opposite side of the other alpha carbon, line across to other side of second alpha carbon, bezier curve back to intersection point, bezier curve to opposite side of first alpha carbon, and line across to the starting point.

        Parameters
        ----------
        aCA : np.array(Float, Float)
            Anion alpha-Carbon Position
        aCAa1 : np.array(Float, Float)
            Anion alpha-Carbon Anchor 1
        aCAa2 : np.array(Float, Float)
            Anion alpha-Carbon Anchor 2
        aSC : np.array(Float, Float)
            Anion Sidechain Position
        cCA : np.array(Float, Float)
            Cation alpha-Carbon Position
        cCAa1 : np.array(Float, Float)
            Cation alpha-Carbon Anchor 1
        cCAa2 : np.array(Float, Float)
            Cation alpha-Carbon Anchor 2
        cSC : np.array(Float, Float)
            Cation Sidechain Position
        interlayer : Bool
            If true, use interlayer style, default is False
        intralayer : Bool
            If true, use intralayer style, default is False
        I : np.array(Float, Float) or None
            The intersection point. If None, optimal intersection point will be calculated. Default is None
        arcscale : Float
            Factor by which to scale the arc. Default=3
        '''
        # Find intersection or mid point
        if I is None:
            mSC = (aSC+cSC)/2 # Midpoint Between Sidechains
            mCA = (aCA+cCA)/2 # Midpoint Between Alpha Carbons
            mV = mSC - mCA # Vector from CA midpoint to SC midpoint
            if np.linalg.norm(mV) == 0: # If they are pointed directly at each other make intersection the sidechains midpoint
                I = mSC
            else:
                scale = arcscale*np.linalg.norm(aSC-aCA)*((np.pi-_angle_180(aSC-aCA, cSC-cCA))/np.pi) # Scaling Factor (how far you want intersection point to be from midpoint of alpha carbons). Here, the greater the angle between the two residues CA to SC vectors, the greater the distance to the intersection point.
                I = mCA+(mV/np.linalg.norm(mV))*scale # Apply scaling factor

        # self.ax.scatter(mSC[0], mSC[1], color="red", zorder=20)
        # self.ax.scatter(mCA[0], mCA[1], color="blue", zorder=20)

        # Create Path
        path_data = [
        (mpath.Path.MOVETO, aCAa1.tolist()), # Start from one side of anionic residue CA circle
        (mpath.Path.CURVE3, I.tolist()), # Curve to intersection point
        (mpath.Path.CURVE3, cCAa1.tolist()), # Curve to a side of the cationic residue
        (mpath.Path.LINETO, cCAa2.tolist()), # straight to other side of cationic residue
        (mpath.Path.CURVE3, I.tolist()), # Curve to intersection point
        (mpath.Path.CURVE3, aCAa2.tolist()), # Curve to other side of anionic residue
        (mpath.Path.CLOSEPOLY, aCAa1.tolist())] # Straight line to return to starting point

        codes, verts = zip(*path_data)
        path = mpath.Path(verts, codes)

        patches = []
        if intralayer:
            patches.append(PathPatch(path, facecolor=self.__params.saltbridge_color_1, edgecolor=self.__params.saltbridge_color_2, linewidth=1, linestyle="solid"))
            if interlayer:
                patches.append(PathPatch(path, facecolor="none", edgecolor=self.__params.saltbridge_color_3, linewidth=0.5, linestyle="dashed"))
                self.__legend_items["sb"]["both"] = True
        elif interlayer:
            patches.append(PathPatch(path, facecolor=self.__params.saltbridge_color_1, edgecolor=self.__params.saltbridge_color_1, linewidth=1, linestyle="solid"))
            patches.append(PathPatch(path, facecolor="none", edgecolor=self.__params.saltbridge_color_2, linewidth=1, linestyle="dashed"))

        self.ax.add_collection(PatchCollection(patches, match_original=True, zorder=4)) #6
        self.__legend_items["sb"]["section"] = True

    
    def add_salt_bridges(self):
        '''
        Add Salt Bridges to Map

        Returns
        -------
        sb_string : Str
            List of salt bridges added to map
        '''
        sbinfo = np.load(self.__params.sb_processed_file)
        # Only loop through salt bridges that won't be cutoff
        sbinfo = sbinfo[np.any(sbinfo[:,-2:]>self.__params.saltbridge_cutoff, axis=1),:]
        # Switch to zero-based indexing of protofilaments and residues
        sbinfo[:, (0,1,3,4)] = sbinfo[:, (0,1,3,4)] - 1
        deg30rotmat = np.array([[np.cos(np.pi/6), -np.sin(np.pi/6)], [np.sin(np.pi/6), np.cos(np.pi/6)]])
        deg330rotmat = np.array([[np.cos(11*np.pi/6), -np.sin(11*np.pi/6)], [np.sin(11*np.pi/6), np.cos(11*np.pi/6)]])
        sb_strings = []
        types = ["BB", "SC", "T"]
        for (api, ari, asi, cpi, cri, csi), (intraL, interL) in zip(sbinfo[:,:-2].astype(int), sbinfo[:,-2:]):
            anion = self.chains[api].residues[ari]
            cation = self.chains[cpi].residues[cri]
            
            # add to map
            if asi == 1 and csi == 1: # Sidechain to sidechain
                self._salt_bridge(anion.ca_position, anion.ca_anchors[0], anion.ca_anchors[1], anion.sc_anchor,
                                  cation.ca_position, cation.ca_anchors[0], cation.ca_anchors[1], cation.sc_anchor,
                                  interlayer=interL>self.__params.saltbridge_cutoff,
                                  intralayer=intraL>self.__params.saltbridge_cutoff)
            elif asi == 2 and csi == 1: # Terminus to sidechain
                tvec = anion.t_anchor-anion.ca_position
                ca_anchor1 = deg30rotmat.dot(tvec) + anion.ca_position
                ca_anchor2 = deg330rotmat.dot(tvec) + anion.ca_position
                self._salt_bridge(anion.ca_position, ca_anchor1, ca_anchor2, anion.t_anchor,
                                  cation.ca_position, cation.ca_anchors[0], cation.ca_anchors[1], cation.sc_anchor,
                                  interlayer=interL>self.__params.saltbridge_cutoff,
                                  intralayer=intraL>self.__params.saltbridge_cutoff, I=anion.t_position)
            elif asi == 1 and csi == 2: # Sidechain to terminus
                tvec = cation.t_anchor-cation.ca_position
                ca_anchor1 = deg30rotmat.dot(tvec) + cation.ca_position
                ca_anchor2 = deg330rotmat.dot(tvec) + cation.ca_position
                self._salt_bridge(anion.ca_position, anion.ca_anchors[0], anion.ca_anchors[1], anion.sc_anchor,
                                  cation.ca_position, ca_anchor1, ca_anchor2, cation.t_anchor,
                                  interlayer=interL>self.__params.saltbridge_cutoff,
                                  intralayer=intraL>self.__params.saltbridge_cutoff, I=cation.t_position)
            
            # For printing
            if asi in [1,2] and csi in [1,2]:
                aID = f"{api+1}-{anion.resname}-{types[asi]}"
                cID = f"{cpi+1}-{cation.resname}-{types[csi]}"
                intraL_str = f"{np.round(intraL, 3):.3f}"
                interL_str = f"{np.round(interL, 3):.3f}"
                sb_strings.append(f"{aID:>12}{cID:>12}{intraL_str:>10}{interL_str:>12}")
            else:
                sb_strings.append(f"------------------------------------------------------------------\nWarning: Unknown Salt Bridge Type:\n{aID:>12}{cID:>12}{intraL_str:>10}{interL_str:>12}\n------------------------------------------------------------------")

        return "\n".join(sb_strings)

    # -------------------------------- PI STACKING --------------------------------
    def _pi_stack(self, sc1, sc2=None, interlayer=False, intralayer=False):
        '''
        Drawing utility for pi stacking interaction

        For same residue different layer: only specify sc1. Setting interlayer to True is not necessary. 
        For pair of residues: Set sc1 and sc2 and specify interlayer and intralayer

        Parameters
        ----------
        sc1 : np.array(Float, Float)
            Sidechain anchor position for residue 1
        sc2 : np.array(Float, Float) or None
            Sidechain anchor position for residue 2. If not specified, a same residue marker will be placed at sc1.
        interlayer : Bool
            If true, use interlayer style
        intralayer : Bool
            If true, use intralayer style
        '''
        if sc1 is not None and sc2 is None: # Place same residue marker
            self.ax.plot(sc1[0], sc1[1], linestyle=None, color=self.__params.pistacking_color_2, zorder=8, marker='o', 
                         markerfacecolor=self.__params.pistacking_color_2, markerfacecoloralt=self.__params.pistacking_color_2, 
                         markeredgecolor=self.__params.pistacking_color_1, fillstyle='full', markersize=4, markeredgewidth=0.5)
            self.__legend_items["pipi"]["same"] = True
        elif intralayer:
            self.ax.annotate('', xy=sc1, xytext=sc2, 
                             arrowprops=dict(arrowstyle='-', fc='none', ec=self.__params.pistacking_color_1, shrinkA=0, shrinkB=0, 
                                             linestyle="solid", linewidth=1, connectionstyle=f"arc3, rad=0"), 
                             zorder=6)
            if interlayer:
                self.ax.annotate('', xy=sc1, xytext=sc2, 
                                 arrowprops=dict(arrowstyle='-', fc='none', ec=self.__params.pistacking_color_2, shrinkA=0, shrinkB=0, 
                                                 linestyle="dotted", linewidth=0.5, connectionstyle=f"arc3, rad=0"), 
                                 zorder=6)
                self.__legend_items["pipi"]["both"] = True
        elif interlayer:
            self.ax.annotate('', xy=sc1, xytext=sc2, 
                             arrowprops=dict(arrowstyle='-', fc='none', ec=self.__params.pistacking_color_1, shrinkA=0, shrinkB=0, 
                                             linestyle="dotted", linewidth=1, connectionstyle=f"arc3, rad=0"), 
                             zorder=6)
        self.__legend_items["pipi"]["section"] = True

    def add_pi_stacking(self):
        '''
        Add pi stacking interactions to map.

        Returns
        -------
        pi_string : Str
            Formatted string with details of pi stacking interactions that were added to the map. 
        '''
        psinfo = np.load(self.__params.pi_processed_file)
        # Only loop through pi interactions that won't be cutoff
        psinfo = psinfo[np.any(np.column_stack((psinfo[:,4]>self.__params.pistacking_cutoff, psinfo[:,9]>self.__params.pistacking_cutoff)), axis=1),:]
        # Switch to zero-based indexing of protofilaments and residues
        psinfo[:, :4] = psinfo[:, :4] - 1

        pi_strings = []
        for (p1i, r1i, p2i, r2i), probs in zip(psinfo[:,:4].astype(int), psinfo[:,4:]):
            intraL = probs[0] # Intralayer Total
            interL = probs[5] # Interlayer Total
            residue1 = self.chains[p1i].residues[r1i]
            residue2 = self.chains[p2i].residues[r2i]
            ID1 = f"{p1i+1}-{residue1.resname}"
            ID2 = f"{p2i+1}-{residue2.resname}"
            if p1i == p2i and r1i == r2i and interL > self.__params.pistacking_cutoff: # If they are the same residue different layers
                self._pi_stack(residue1.sc_anchor)
                ID2 = "SAME"
            else:
                self._pi_stack(residue1.sc_anchor, residue2.sc_anchor, interlayer=(interL>self.__params.pistacking_cutoff), intralayer=(intraL>self.__params.pistacking_cutoff))
            
            # Add to string
            intraL_To = f"{np.round(intraL, 3):.3f}"
            intraL_T = f"{np.round(probs[1],3):.3f}" # Intralayer T-Shaped
            intraL_I = f"{np.round(probs[2],3):.3f}" # Intralayer Intermediate
            intraL_S = f"{np.round(probs[3],3):.3f}" # Intralayer Sandwich
            intraL_D = f"{np.round(probs[4],3):.3f}" # Intralayer Parallel Displaced
            interL_To = f"{np.round(interL, 3):.3f}"
            interL_T = f"{np.round(probs[6],3):.3f}" # Interlayer T-Shaped
            interL_I = f"{np.round(probs[7],3):.3f}" # Interlayer Intermediate
            interL_S = f"{np.round(probs[8],3):.3f}" # Interlayer Sandwich
            interL_D = f"{np.round(probs[9],3):.3f}" # Interlayer Parallel Displaced
            pi_strings.append(f"{ID1:>11}{ID2:>12}{intraL_To:>10}{intraL_T:>7}{intraL_I:>7}{intraL_S:>7}{intraL_D:>7}{interL_To:>10}{interL_T:>7}{interL_I:>7}{interL_S:>7}{interL_D:>7}")

        return "\n".join(pi_strings)

    def make_legend(self):
        '''
        Add legend to the figure
        '''
        def _legend_residue(ax, residue):
            overpatches, underpatches = [], []
            overpatches.append(Circle((residue.ca_position[0], residue.ca_position[1]), residue.radius, color=residue.rescolor))
            overpatches.append(Polygon(residue.vertices, color=residue.rescolor))
            underpatches.append(Circle((residue.ca_position[0], residue.ca_position[1]), residue.underradius, color=residue.chaincolor))
            underpatches.append(Polygon(residue.undervertices, color=residue.chaincolor))
            ax.add_collection(PatchCollection(underpatches, match_original=True, zorder=10))
            ax.add_collection(PatchCollection(overpatches, match_original=True, zorder=12))
            ax.annotate(residue.label, (residue.ca_position[0], residue.ca_position[1]), ha="center", va="center", color=residue.labelcolor, fontsize=4, fontweight="bold", zorder=14)

        # for sk, sv in self.__legend_items.items():
        #     for k, v in sv.items():
        #         self.__legend_items[sk][k] = True

        # First, get starting positions
        legheight = 45 # Going to be at least 50pts for top+bottom buffer and main
        if self.__legend_items["main"]["water"] or self.__legend_items["main"]["zipper"]: legheight += 10 # if Regions, add 10
        if self.__legend_items["hbond"]["section"]:
            legheight += 25 # having the section adds at least 25
            if self.__legend_items["hbond"]["both"] or self.__legend_items["hbond"]["backbone"]: # adding a row
                legheight += 10
            if self.__legend_items["hbond"]["same"]:
                legheight += 12
        if self.__legend_items["sb"]["section"]:
            legheight += 25 # having section adds at least 25
            if self.__legend_items["sb"]["both"]:
                legheight += 12
        if self.__legend_items["pipi"]["section"]:
            legheight += 25 # having section adds at least 25
            if self.__legend_items["pipi"]["both"] or self.__legend_items["pipi"]["same"]:
                legheight += 10
        
        self.__legend_bottomleft[1] = self.__legend_bottomleft[1] + (self.__legend_dims[1]-legheight)/2
        self.__legend_dims[1] = legheight

        currentx = self.__legend_bottomleft[0]+10
        currenty = self.__legend_bottomleft[1]+self.__legend_dims[1]-10
        starty = self.__legend_bottomleft[1]+self.__legend_dims[1]
        safe_width = self.__legend_dims[0]-20

        # Second, add the main components
        # Distance Indicator
        marker_pos = [np.linspace(currentx+(safe_width/2)-self.__legend_1nm/2, currentx+(safe_width/2)+self.__legend_1nm/2, 11), [currenty]*11]
        self.ax.plot(*marker_pos, ls="-", color="black", solid_capstyle='butt', linewidth=1, markevery=10, zorder=2)
        for i, mx in enumerate(marker_pos[0]):
            if i == 0 or i == 10: # End Caps
                self.ax.plot([mx, mx], [currenty-2.5, currenty+2.5], ls="-", color="black", solid_capstyle='butt', linewidth=1, zorder=2)
            elif i ==5: # Midpoint
                self.ax.plot([mx, mx], [currenty-2.5, currenty], ls="-", color="black", solid_capstyle='butt', linewidth=0.5, zorder=2)
            else:
                self.ax.plot([mx, mx], [currenty-1.5, currenty], ls="-", color="black", solid_capstyle='butt', linewidth=0.5, zorder=2)
        self.ax.annotate(r"10 $\mathrm{\AA}$", (currentx+(safe_width/2), currenty+1.5), ha="center", va="bottom", fontweight="normal", fontsize=4, zorder=2)
        currenty -= 12

        # Backbone Representation
        x1 = currentx+5
        x2 = currentx+safe_width-5
        self.ax.plot([x1, x2], [currenty]*2, color=self.__params.backbone_color[0], linewidth=2, solid_capstyle="butt", zorder=2)
        self.ax.plot(x1, currenty, markeredgecolor=self.__params.backbone_color[0], markerfacecolor=self.__params.backbone_color[0], 
                         markersize=4, markeredgewidth=0.5, marker='o', zorder=2)
        self.ax.plot(x2, currenty, markeredgecolor=self.__params.backbone_color[0], markerfacecolor=self.__params.backbone_color[0], 
                         markersize=4, markeredgewidth=0.5, marker=(3,0,-90), zorder=2)
        self.ax.annotate("N", (currentx, currenty), ha="right", va="center_baseline", fontweight="normal", fontsize=4, zorder=2)
        self.ax.annotate("C", (currentx+safe_width, currenty), ha="left", va="center_baseline", fontweight="normal", fontsize=4, zorder=2)

        # Residue Markers
        for restype, rescolor, reslabelcolor, shiftx, reslabel in zip(["Acidic", "Basic", "Polar", "Nonpolar"], 
                                     [self.__params.acidic_color, self.__params.basic_color, self.__params.polar_color, self.__params.nonpolar_color],
                                     [self.__params.acidic_label_color, self.__params.basic_label_color, self.__params.polar_label_color, self.__params.nonpolar_label_color],
                                     [safe_width/5, 2*safe_width/5, 3*safe_width/5, 4*safe_width/5],
                                     [r"$\ominus$", r"$\oplus$", r"$\rightarrow$", r"$\nrightarrow$"]):
            residue = Residue(None, reslabel, np.array([currentx+shiftx, currenty]), np.array([currentx+shiftx, currenty+3]), 2, 2.75, self.__params.backbone_color[0], rescolor, reslabelcolor, legend_residue=True)
            _legend_residue(self.ax, residue)
            self.ax.annotate(restype, (currentx+shiftx, currenty-6), ha="center", va="center_baseline", fontweight="normal", fontsize=4, zorder=2)
        currenty -= 13

        # Water Channel and Hydrophobic Zipper
        if self.__legend_items["main"]["zipper"] and self.__legend_items["main"]["water"]:
            zipperx = currentx+(3*safe_width/4)
            waterx = currentx+(safe_width/4)
        elif self.__legend_items["main"]["zipper"]:
            zipperx = currentx+(safe_width/2)
        elif self.__legend_items["main"]["water"]:
            waterx = currentx+(safe_width/2)

        if self.__legend_items["main"]["zipper"]:
            self.ax.annotate("Hydrophobic Zipper", [zipperx, currenty], fontsize=4, ha='center', va='center_baseline', 
                             bbox=dict(boxstyle='square, pad=0.5', alpha=self.__params.zipper_opacity, fc=self.__params.zipper_color, ec="none"), zorder=2)
        if self.__legend_items["main"]["water"]:
            self.ax.annotate("Water Channel", [waterx, currenty], fontsize=4, ha='center', va='center_baseline', 
                             bbox=dict(boxstyle='square, pad=0.5', alpha=self.__params.water_opacity, fc=self.__params.water_color, ec="none"), zorder=2)
        
        if self.__legend_items["main"]["zipper"] or self.__legend_items["main"]["water"]:
            currenty -= 10

        # Third, Add Hydrogen Bond information
        if self.__legend_items["hbond"]["section"]:
            currenty -= 5
            self.ax.annotate("Hydrogen Bonds", [currentx+(safe_width/2), currenty], fontsize=4, fontweight="bold", ha='center', va='center_baseline', zorder=2)
            currenty -= 10

            # Interlayer/Intralayer
            res1 = Residue(None, "", np.array([currentx, currenty]), np.array([currentx+3, currenty-2]), 2, 2.75, "black", "white", "white", legend_residue=True)
            res2 = Residue(None, "", np.array([currentx+(safe_width/2)-10, currenty]), np.array([currentx+(safe_width/2)-13, currenty-2]), 2, 2.75, "black", "white", "white", legend_residue=True)
            _legend_residue(self.ax, res1)
            _legend_residue(self.ax, res2)
            self.ax.annotate("Intralayer", [currentx+((safe_width/2)-10)/2, currenty], fontsize=4, ha='center', va='center_baseline', zorder=2)
            self.ax.annotate('', xy=res1.sc_anchor, xytext=res2.sc_anchor, 
                                 arrowprops=dict(arrowstyle='-', fc='none', ec=self.__params.hbond_color_1, shrinkA=3, shrinkB=0, 
                                                 linestyle="solid", linewidth=1, connectionstyle=f"arc3, rad=-0.15"), zorder=2) # LINE
            self.ax.annotate('', xy=res1.sc_anchor, xytext=res2.sc_anchor, 
                                arrowprops=dict(arrowstyle='<|-', fc=self.__params.hbond_color_1, ec=self.__params.hbond_color_1, shrinkA=0, shrinkB=0, 
                                                linestyle='solid', mutation_scale=9, linewidth=0, connectionstyle=f"arc3, rad=-0.15"), zorder=2) # ARROWHEAD
            
            res1 = Residue(None, "", np.array([currentx+(safe_width/2)+10, currenty]), np.array([currentx+(safe_width/2)+13, currenty-2]), 2, 2.75, "black", "white", "white", legend_residue=True)
            res2 = Residue(None, "", np.array([currentx+safe_width, currenty]), np.array([currentx+safe_width-3, currenty-2]), 2, 2.75, "black", "white", "white", legend_residue=True)
            _legend_residue(self.ax, res1)
            _legend_residue(self.ax, res2)
            self.ax.annotate("Interlayer", [currentx+((3*safe_width)/4)+5, currenty], fontsize=4, ha='center', va='center_baseline', zorder=2)
            self.ax.annotate('', xy=res1.sc_anchor, xytext=res2.sc_anchor, 
                                 arrowprops=dict(arrowstyle='-', fc='none', ec=self.__params.hbond_color_1, shrinkA=3, shrinkB=0, 
                                                 linestyle="dashed", linewidth=1, connectionstyle=f"arc3, rad=-0.15"), zorder=2) # LINE
            self.ax.annotate('', xy=res1.sc_anchor, xytext=res2.sc_anchor, 
                                arrowprops=dict(arrowstyle='<|-', fc=self.__params.hbond_color_1, ec=self.__params.hbond_color_1, shrinkA=0, shrinkB=0, 
                                                linestyle='solid', mutation_scale=9, linewidth=0, connectionstyle=f"arc3, rad=-0.15"), zorder=2) # ARROWHEAD
            currenty -= 10

            if self.__legend_items["hbond"]["both"] and self.__legend_items["hbond"]["backbone"]:
                bothx = currentx
                backbonex = currentx + safe_width/2 + 10
            elif self.__legend_items["hbond"]["both"]:
                bothx = currentx + safe_width/4 + 5
            elif self.__legend_items["hbond"]["backbone"]:
                backbonex = currentx + safe_width/4 + 5
            
            if self.__legend_items["hbond"]["both"]:
                res1 = Residue(None, "", np.array([bothx, currenty]), np.array([bothx+3, currenty-2]), 2, 2.75, "black", "white", "white", legend_residue=True)
                res2 = Residue(None, "", np.array([bothx+(safe_width/2)-10, currenty]), np.array([bothx+(safe_width/2)-13, currenty-2]), 2, 2.75, "black", "white", "white", legend_residue=True)
                _legend_residue(self.ax, res1)
                _legend_residue(self.ax, res2)
                self.ax.annotate("Intra & Inter", [bothx+((safe_width/2)-10)/2, currenty], fontsize=4, ha='center', va='center_baseline', zorder=2)
                self.ax.annotate('', xy=res1.sc_anchor, xytext=res2.sc_anchor, 
                                    arrowprops=dict(arrowstyle='-', fc='none', ec=self.__params.hbond_color_1, shrinkA=3, shrinkB=0, 
                                                    linestyle="solid", linewidth=1, connectionstyle=f"arc3, rad=-0.15"), zorder=2) # LINE
                self.ax.annotate('', xy=res1.sc_anchor, xytext=res2.sc_anchor, 
                                 arrowprops=dict(arrowstyle='-', fc='none', ec=self.__params.hbond_color_2, shrinkA=3, shrinkB=0, 
                                            linestyle="dashed", linewidth=0.5, connectionstyle=f"arc3, rad=-0.15"), zorder=2) # DASHED LINE
                self.ax.annotate('', xy=res1.sc_anchor, xytext=res2.sc_anchor, 
                                    arrowprops=dict(arrowstyle='<|-', fc=self.__params.hbond_color_1, ec=self.__params.hbond_color_1, shrinkA=0, shrinkB=0, 
                                                    linestyle='solid', mutation_scale=9, linewidth=0, connectionstyle=f"arc3, rad=-0.15"), zorder=2) # ARROWHEAD
                
            if self.__legend_items["hbond"]["backbone"]:
                res1 = Residue(None, "", np.array([backbonex, currenty]), np.array([backbonex, currenty-2]), 2, 2.75, "black", "white", "white", legend_residue=True)
                res2 = Residue(None, "", np.array([backbonex+(safe_width/2)-10, currenty]), np.array([backbonex+(safe_width/2)-10, currenty-2]), 2, 2.75, "black", "white", "white", legend_residue=True)
                self.ax.plot([res1.ca_position, res2.ca_position], [currenty]*2, color="black", linewidth=2, zorder=2)
                self.ax.annotate('', xy=res1.ca_position, xytext=res2.ca_position, 
                             arrowprops=dict(arrowstyle='-', fc='none', ec=self.__params.hbond_color_2, shrinkA=0, shrinkB=0, 
                                             linestyle="dashed", connectionstyle=f"arc3, rad=0"), zorder=2)
                _legend_residue(self.ax, res1)
                _legend_residue(self.ax, res2)
                self.ax.annotate(r"$\mathrm{\beta}$-strand", [backbonex+((safe_width/2)-10)/2, currenty-5], fontsize=4, ha='center', va='center_baseline', zorder=2)

            if self.__legend_items["hbond"]["both"] or self.__legend_items["hbond"]["backbone"]:
                currenty -= 10

            # Finally, add same
            if self.__legend_items["hbond"]["same"]:
                currenty -= 2
                samex = currentx + 2*safe_width/3 
                residue = Residue(None, "", np.array([samex, currenty]), np.array([samex-3, currenty]), 2, 2.75, "black", "white", "white", legend_residue=True)
                _legend_residue(self.ax, residue)
                self.ax.plot(residue.sc_anchor[0], residue.sc_anchor[1], marker='o', markersize=3, linestyle=None, 
                             markerfacecolor=self.__params.hbond_color_2, markeredgecolor=self.__params.hbond_color_1, 
                             markeredgewidth=0.5, zorder=16)
                self.ax.annotate("Interlayer\n(Same Position)", [residue.sc_anchor[0]-3, currenty], fontsize=4, ha='right', va='center_baseline', zorder=2)
                currenty -= 10
            
        # Fourth, Salt Bridges
        if self.__legend_items["sb"]["section"]:
            currenty -= 5
            self.ax.annotate("Salt Bridges", [currentx+(safe_width/2), currenty], fontsize=4, fontweight="bold", ha='center', va='center_baseline', zorder=2)
            currenty -= 10

            res1 = Residue(None, "", np.array([currentx, currenty]), np.array([currentx+3, currenty-2]), 2, 2.75, "black", "white", "white", legend_residue=True)
            res2 = Residue(None, "", np.array([currentx+(safe_width/2)-10, currenty]), np.array([currentx+(safe_width/2)-13, currenty-2]), 2, 2.75, "black", "white", "white", legend_residue=True)
            _legend_residue(self.ax, res1)
            _legend_residue(self.ax, res2)
            self._salt_bridge(res1.ca_position, res1.ca_anchors[0], res1.ca_anchors[1], res1.sc_anchor,
                              res2.ca_position, res2.ca_anchors[0], res2.ca_anchors[1], res2.sc_anchor,
                              interlayer=False,
                              intralayer=True,
                              arcscale=6)
            self.ax.annotate("Intralayer", [currentx+((safe_width/2)-10)/2, currenty], fontsize=4, ha='center', va='center_baseline', zorder=2)

            res1 = Residue(None, "", np.array([currentx+(safe_width/2)+10, currenty]), np.array([currentx+(safe_width/2)+13, currenty-2]), 2, 2.75, "black", "white", "white", legend_residue=True)
            res2 = Residue(None, "", np.array([currentx+safe_width, currenty]), np.array([currentx+safe_width-3, currenty-2]), 2, 2.75, "black", "white", "white", legend_residue=True)
            _legend_residue(self.ax, res1)
            _legend_residue(self.ax, res2)
            self.ax.annotate("Interlayer", [currentx+((3*safe_width)/4)+5, currenty], fontsize=4, ha='center', va='center_baseline', zorder=2)
            self._salt_bridge(res1.ca_position, res1.ca_anchors[0], res1.ca_anchors[1], res1.sc_anchor,
                              res2.ca_position, res2.ca_anchors[0], res2.ca_anchors[1], res2.sc_anchor,
                              interlayer=True,
                              intralayer=False,
                              arcscale=6)
            currenty -= 10

            if self.__legend_items["sb"]["both"]:
                currenty -= 2
                res1 = Residue(None, "", np.array([currentx+(safe_width/4)+5, currenty]), np.array([currentx+(safe_width/4)+8, currenty-2]), 2, 2.75, "black", "white", "white", legend_residue=True)
                res2 = Residue(None, "", np.array([currentx+(3*safe_width/4)-5, currenty]), np.array([currentx+(3*safe_width/4)-8, currenty-2]), 2, 2.75, "black", "white", "white", legend_residue=True)
                _legend_residue(self.ax, res1)
                _legend_residue(self.ax, res2)
                self.ax.annotate("Intra & Inter", [currentx+(safe_width/2), currenty+1.5], fontsize=4, ha='center', va='center_baseline', zorder=2)
                self._salt_bridge(res1.ca_position, res1.ca_anchors[0], res1.ca_anchors[1], res1.sc_anchor,
                                  res2.ca_position, res2.ca_anchors[0], res2.ca_anchors[1], res2.sc_anchor,
                                  interlayer=True,
                                  intralayer=True,
                                  arcscale=6)
                currenty -= 10
        
        # Fifth, Pi Stacking Interactions
        if self.__legend_items["pipi"]["section"]:
            currenty -= 5
            self.ax.annotate("Pi Stacking", [currentx+(safe_width/2), currenty], fontsize=4, fontweight="bold", ha='center', va='center_baseline', zorder=2)
            currenty -= 10

            res1 = Residue(None, "", np.array([currentx, currenty]), np.array([currentx+3, currenty]), 2, 2.75, "black", "white", "white", legend_residue=True)
            res2 = Residue(None, "", np.array([currentx+(safe_width/2)-10, currenty]), np.array([currentx+(safe_width/2)-13, currenty]), 2, 2.75, "black", "white", "white", legend_residue=True)
            _legend_residue(self.ax, res1)
            _legend_residue(self.ax, res2)
            self._pi_stack(res1.sc_anchor, res2.sc_anchor, interlayer=False, intralayer=True)
            self.ax.annotate("Intralayer", [currentx+((safe_width/2)-10)/2, currenty+1.5], fontsize=4, ha='center', va='bottom', zorder=2)

            res1 = Residue(None, "", np.array([currentx+(safe_width/2)+10, currenty]), np.array([currentx+(safe_width/2)+13, currenty]), 2, 2.75, "black", "white", "white", legend_residue=True)
            res2 = Residue(None, "", np.array([currentx+safe_width, currenty]), np.array([currentx+safe_width-3, currenty]), 2, 2.75, "black", "white", "white", legend_residue=True)
            _legend_residue(self.ax, res1)
            _legend_residue(self.ax, res2)
            self._pi_stack(res1.sc_anchor, res2.sc_anchor, interlayer=True, intralayer=False)
            self.ax.annotate("Interlayer", [currentx+((3*safe_width)/4)+5, currenty+1.5], fontsize=4, ha='center', va='bottom', zorder=2)
            
            currenty -= 10

            if self.__legend_items["pipi"]["both"] and self.__legend_items["pipi"]["same"]:
                bothx = currentx
                samex = currentx + safe_width
            elif self.__legend_items["pipi"]["both"]:
                bothx = currentx + safe_width/4 + 5
            elif self.__legend_items["pipi"]["same"]:
                samex = currentx + safe_width/2
            
            if self.__legend_items["pipi"]["both"]:
                res1 = Residue(None, "", np.array([bothx, currenty]), np.array([bothx+3, currenty]), 2, 2.75, "black", "white", "white", legend_residue=True)
                res2 = Residue(None, "", np.array([bothx+(safe_width/2)-10, currenty]), np.array([bothx+(safe_width/2)-13, currenty]), 2, 2.75, "black", "white", "white", legend_residue=True)
                _legend_residue(self.ax, res1)
                _legend_residue(self.ax, res2)
                self._pi_stack(res1.sc_anchor, res2.sc_anchor, interlayer=True, intralayer=True)
                self.ax.annotate("Intra & Inter", [bothx+((safe_width/4)+10)/2, currenty+1.5], fontsize=4, ha='center', va='bottom', zorder=2)
            
            if self.__legend_items["pipi"]["same"]:
                residue = Residue(None, "", np.array([samex, currenty]), np.array([samex-3, currenty]), 2, 2.75, "black", "white", "white", legend_residue=True)
                _legend_residue(self.ax, residue)
                self._pi_stack(residue.sc_anchor, interlayer=True)
                self.ax.annotate("Interlayer\n(Same Position)", [residue.sc_anchor[0]-3, currenty], fontsize=4, ha='right', va='center', zorder=2)
            
            if self.__legend_items["pipi"]["both"] or self.__legend_items["pipi"]["same"]:
                currenty -= 10
        # Finally, draw a box around the key
        bordercol = [FancyBboxPatch((self.__legend_bottomleft[0], currenty), self.__legend_dims[0], starty-currenty, fc="white", ec="black", boxstyle=BoxStyle("Round", pad=0.02), zorder=0)]
        self.ax.add_collection(PatchCollection(bordercol, match_original=True, zorder=0))

    def save(self):
        '''
        Save and show fibril map
        '''

        # Finish Axis Formatting
        self.ax.axis('off')
        plt.axis('equal')

        # Save Figure
        self.fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        plt.savefig(self.__params.figure_file, dpi=self.__dpi, transparent=self.__params.transparent_background)
        # Show Figure
        if self.__params.showfig:
            plt.show()
