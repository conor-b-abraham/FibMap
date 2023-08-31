import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis.hydrogenbonds.hbond_analysis import HydrogenBondAnalysis as HBA
from MDAnalysis.analysis import distances
from tqdm import tqdm
import multiprocessing
from functools import partial
from scipy.spatial.distance import pdist
import warnings

from src import utils

warnings.filterwarnings("ignore", category=UserWarning) # We've already checked the files for information we need, so silence MDAnalysis UserWarnings

# CONTENTS:
# Contains calculators for FibMap.py calc
#       HydrogenBondCalculator: Calculates and Parses Hydrogen Bonds
#       SaltBridgeCalculator: Calculates and Parses Salt Bridges
#       PiStackingCalculator: Calculates and Parses Pi Stacking Interactions

# ------------------------------------------------------------------------------
# GENERAL PROCESSING
# ------------------------------------------------------------------------------
def _process_interactions(interactions, n_layers, n_frames, combine_intraL_interL=True):
    '''
    Process an array of all detection interactions over a trajectory and return each type of reaction and its occurence probability

    Parameters
    ----------
    interactions : nd.array [N, 9] or [N,7]
        An array with each row specifying each interaction. For interactions in which the sidechain and backbone bead should be distiguished between (ie H-Bonds), must be Nx9; Otherwise must be Nx7. The first column is always the frame in which each interaction occurs. The subsequent set of columns must include information about the first interaction site (Donor for H-Bonds), and the final set of columns must include information about the second interaction site (Acceptor for H-Bonds). For Nx9 shaped array: A set of columns must include (in order) LayerID, ProtofilamentID, ResidueID, SiteTypeID. For Nx7 shaped array: A set of columns must include (in order) LayerID, ProtofilamentID, ResidueID.
    n_layers : Int
        The number of layers in the fibril that are included in the analysis
    n_frames : Int
        The number of frames in the trajectory
    combine_intraL_interL : Bool
        Optional argumnet. If true (default), interlayer and intralayer results for each interaction type will be combined into individual rows. Otherwise, they will be kept separate.
    '''
    if interactions.shape[1] == 9:
        include_sitetype=True
    elif interactions.shape[1] == 7:
        include_sitetype=False
    else:
        raise SystemExit("Error: Improper Interaction Array Provided to Processing")
    
    # Construct new array with the frame, layers, protofilaments, residues, interaction sites (BB=0, SC=1 or T=2), and layer difference. Only include 1 of each interaction layer-and-site-wise (i.e. only one for each Layer-Protofilament-Residue-Site).
    if include_sitetype:
        alsel = 5
        dlsel = 1
    else:
        alsel = 4
        dlsel = 1
    residues = np.unique(np.column_stack((
        interactions, # the frame and all site information
        interactions[:,alsel]-interactions[:,dlsel] # Donor Layer - Acceptor Layer
    )), axis=0) 
    residues[residues[:,-1]!=0,-1] = 1 # Interlayer (1) or Intralayer (0)
    # residues for include_sitetype:: 0:Frame, 1:D_l, 2: D_p, 3: D_r, 4: D_s, 5: A_l, 6: A_p, 7: A_r, 8: A_s, 9: InterLvIntraL
    # residues for not include_sitetype:: # 0:Frame, 1:D_l, 2: D_p, 3: D_r, 4: A_l, 5: A_p, 6: A_r, 7: InterLvIntraL
    
    # Construct new array with the frame, protofilaments, residues, interaction sites, interlayer v intralayer status (1=interlayer, 0=intralayer), and absolute value of layer difference (i.e. layer span). Only include 1 of each interaction type layerspan-and-site-wise. Count how many of each layerdiff-and-site-wise type there is and put in last column.
    residues_unique = np.column_stack(np.unique(np.column_stack((
        residues[:,0], # Frame
        residues[:,dlsel+1:alsel], # Donor info (Protofilament, Residue, Site)
        residues[:,alsel+1:], # Acceptor info (Protofilament, Residue, Site), IntraLvInterL
        residues[:,alsel]-residues[:,dlsel] # Layer Difference
    )), return_counts=True, axis=0))
    residues_unique[:,-2] = np.abs(residues_unique[:,-2]) # Convert layer difference to layerspan
    # residues_unique for include_sitetype:: 0:Frame, 1:D_p, 2: D_r, 3: D_s, 4: A_p, 5: A_r, 6: A_s, 7: InterLvIntraL, 8: LayerSpan, 9:Count
    # residues_unique for not include_sitetype:: 0:Frame, 1:D_p, 2: D_r, 3: A_p, 4: A_r, 5: InterLvIntraL, 6: LayerSpan, 7:Count
    
    # First, for each frame, consider each type of interaction, sitewise (i.e. layers are ignored). For each site-wise type, how many layerdiff-and-site-wise types are there? If there is only one, calculate its pHB the fast way:
    details_unique, details_indices, details_inverse, details_counts = np.unique(residues_unique[:,:-2], axis=0, return_index=True, return_inverse=True, return_counts=True)
    # details_unique for include_sitetype:: 0:Frame, 1:D_p, 2: D_r, 3: D_s, 4: A_p, 5: A_r, 6: A_s, 7: InterLvIntraL
    # details_unique for not include_sitetype:: 0:Frame, 1:D_p, 2: D_r, 3: A_p, 4: A_r, 5: InterLvIntraL
    selsingle = (details_counts == 1)
    details_single = details_unique[selsingle,:]
    single_indices = details_indices[selsingle]
    frames_single = residues_unique[single_indices,:][:,-1]/(n_layers-residues_unique[single_indices,:][:,-2])  # N_hbonds/N_possible where N_possible = N_layers-LayerSpan
    # Second, for those site-wise types with more than one layerdiff-and-site-type, we need to calculate its prob. more manually:
    if include_sitetype:
        unisel = (0,2,3,4,6,7,8,9)
        resdim = 3
    else:
        unisel = (0,2,3,5,6,7)
        resdim = 2
    
    search_residues_unique = residues_unique[details_counts[details_inverse]>1]
    details_multi= details_unique[details_counts>1]
    uts, uts_counts = np.unique(details_multi[:,0], return_counts=True)
    frames_multi = []
    for tsi, ts in enumerate(tqdm(uts, desc="P", disable=uts.shape[0]==1)):
        # frames_multi = np.zeros(details_multi.shape[0])
        search_residues_unique_ts = search_residues_unique[search_residues_unique[:,0]==ts]
        search_residues_ts = residues[residues[:,0]==ts]
        frames_multi_ts = np.zeros(uts_counts[tsi])
        for i, unique_row in enumerate(tqdm(details_multi[details_multi[:,0]==ts], desc='P', leave=False, disable=uts.shape[0]>1)):
            selresun = np.all(search_residues_unique_ts[:,:-2]==unique_row, axis=1)
            # for each layerdiff-and-site-wise type we need to know its layerspan and how many of them are present
            types = search_residues_unique_ts[selresun,-2] # The layerdiff-and-site-wise types' layerspans
            typecount = search_residues_unique_ts[selresun,-1] # The layerdiff-and-site-wise types' counts
            max_type_p = np.max((typecount)/(n_layers-types)) # What is the maximum prob. you can get from an individual type?
            selres = np.all(search_residues_ts[:,unisel]==unique_row, axis=1)
            combined_count = np.unique(search_residues_ts[selres,1:resdim+1], axis=0).shape[0]+np.unique(search_residues_ts[selres,resdim+1:-1], axis=0).shape[0] # We get the count constructing an array where each row is a unique instance of an interacting layer-site-wise site and then counting the rows. In other words, we're counting the number of interacting residues.
            combined_p = combined_count/(2*(n_layers-np.min(types))) # What is the prob. you get from a combination of all of the types: N_interacting_sites/(2*N_possible_interactions) where N_possible_interactions = N_layers-LayerSpan
            frames_multi_ts[i] = np.max([max_type_p, combined_p]) # Is the maximum prob. from an individual type greater or is the maximum prob. from the combined types greater?
        frames_multi.append(frames_multi_ts)
    if len(frames_multi) > 1:
        frames_multi = np.hstack(frames_multi)
    else:
        frames_multi = np.array(frames_multi)
        print(frames_multi.shape)
    
    # Third, combine single and multi results
    details_unique=np.vstack((details_multi, details_single))
    if frames_multi.ndim > 1 and frames_multi.size == 1: # THIS IS A BANDAID. NOT A GOOD SOLUTION.
        frames_multi = frames_multi.flatten()
    frames = np.hstack((frames_multi, frames_single))
    
    # Fourth, average over the trajectory
    unique = np.unique(details_unique[:,1:], axis=0)
    probs = np.zeros(unique.shape[0])
    for i, unique_row in enumerate(unique):
        probs[i] = np.sum(frames[np.all(details_unique[:,1:]==unique_row, axis=1)])/n_frames
    
    if combine_intraL_interL:
        # COMBINE INTERLAYER & INTRALAYER
        final_unique = np.unique(unique[:,:-1], axis=0) # Cut off layer diff
        final_values = np.zeros((final_unique.shape[0], 2)) # Intralayer P, Interlayer P
        for i, unique_row in enumerate(final_unique):
            sel = np.all(unique[:,:-1]==unique_row, axis=1)
            layerdiffs = unique[sel, -1].astype(int)

            for layerdiff, pval in zip(layerdiffs, probs[sel]):
                final_values[i, layerdiff] = pval
                
        # RESULTS
        probability_results = np.hstack((
            final_unique.astype(float),
            final_values
        )) # Donor Protofilament, Donor Residue, Donor Type, Acceptor Protofilament, Acceptor Residue, Acceptor Type, Intralayer P, Interlayer P

    else:
        # RESULTS
        probability_results = np.column_stack((
            unique.astype(float),
            probs
        )) # Donor Protofilament, Donor Residue, Donor Type, Acceptor Protofilament, Acceptor Residue, Acceptor Type, Intralayer P, Interlayer P

    return probability_results

# ------------------------------------------------------------------------------
# HYDROGEN BONDING
# ------------------------------------------------------------------------------
def _calc_hbonds(frame_chunk, u, sel, angle_cutoff=150, distance_cutoff=3.5):
            '''
            Calculate hydrogen bonds for a chunk of frames

            Parameters
            ----------
            frame_chunk : tuple
                first and last frame of chunk: (first_frame, last_frame)
            u : MDAnalysis.universe
                universe in which hydrogen bonds are to be computed
            sel : str
                extra selection string to restrict which atoms are considered
            angle_cutoff : float (Default: 150)
                the D-H-A angle cutoff
            distance_cutoff : float (Default: 3.5)
                the D-A distance cutoff
            
            Returns
            -------
            MDAnalysis.analysis.hydrogenbonds.hbond_analysis.results.hbonds
                H-bond results for chunk of frames
            '''
            # Function for multiprocessing hbond calculation
            hbonds = HBA(u, d_a_cutoff=distance_cutoff, d_h_a_angle_cutoff=angle_cutoff)
            hbonds.hydrogens_sel = f"({hbonds.guess_hydrogens('protein')}) and {sel}"
            hbonds.acceptors_sel = f"({hbonds.guess_acceptors('protein')}) and {sel}"
            hbonds.run(start=frame_chunk[0], stop=frame_chunk[1], verbose=True)
            print('\033[1A', end='\x1b[2K') # Clear progress bar
            return hbonds.results.hbonds

class HydrogenBondCalculator:
    '''
    Class for computing and sorting hydrogen bonds

    Attributes
    ----------
    raw_results : numpy.ndarray
        Raw hydrogen bond results from mdanalysis.analysis.hydrogenbonds.hbond_analysis.HydrogenBondAnalysis. Each row contains (in order) the frame, donor index, hydrogen index, acceptor index, distance, and angle of a hydrogen bond.
        See https://docs.mdanalysis.org/stable/documentation_pages/analysis/hydrogenbonds.html
    processed_results : numpy.ndarray or None
        Processed hydrogen bond results with atom information. Each row contains (in order) the donor protofilament index (one-based), the donor residue index (one-based), the donor type (0=backbone, 1=sidechain, 2=terminus), the acceptor protofilament index (one-based), the acceptor residue index (one-based), the acceptor type (0=backone, 1=sidechain, 2=terminus), Intralayer <N_H-Bonds>, Intralayer P(H-Bond), Interlayer <N_H-Bonds>, and Interlayer P(H-Bond). Populated after HydrogenBondCalculator.process() is called. 
    
    Methods
    -------
    process()
        Process raw_results. Populates intramolecular, interlayer, and interprotofilament attributes.
    save(maindirectory=None, rawdirectory=None)
        If maindirectory is not None, save processed results to it. If rawdirectory is not None, save raw results to it.
    show()
        Print summary of processed results to terminal.

    Usage
    -----
    1) Perform initial calculation
    hbonds = HydrogenBondCalculator(topology_file, trajectory_file, n_procs, system_info)

    # You can now save the raw results
    hbonds.save(unprocessed_file='PATH/TO/RESULTS/unprocessed_hydrogen_bonds.npy')

    2) Process the raw results
    hbonds.process()

    # You can now save the processed results
    hbonds.save(processed_file='PATH/TO/RESULTS/unprocessed_hydrogen_bonds.npy')

    # You can now show the results
    hbonds.show(LOG)

    '''
    def __init__(self, topology_file, trajectory_file, nprocs, system_info, unprocessed_filename=None, angle_cutoff=150, distance_cutoff=3.5):
        '''
        Parameters
        ----------
        topology_file : str
            Name of topology file
        trajectory_file : str or [str]
            Name(s) of trajectory file(s)
        nprocs : int
            Number of processors to use
        system_info : utils.SystemInfo
            System Information
        unprocessed_filename : Str or None
            Name of file containing previously calculated unprocessed results. This filename should come only from the Params class. If None, hydrogen bonds will be calculated.
        angle_cutoff : float (Default: 150)
            the D-H-A angle cutoff
        distance_cutoff : float (Default: 3.5)
            the D-A distance cutoff
        '''
        # Input Parameters
        self.__top = topology_file
        self.__traj = trajectory_file
        self.__nprocs = nprocs
        self.__sysinfo = system_info

        # Initialize Empty Objects for Parsed Results
        self.processed_results = None

        # Calculate 
        u = mda.Universe(self.__top, self.__traj)

        if unprocessed_filename is not None:
            self.raw_results = np.load(unprocessed_filename)
        else:
            sel = f"segid {' '.join(self.__sysinfo.structure.flatten())}"
            # Can't use more processors than there are frames
            if self.__nprocs > u.trajectory.n_frames: 
                self.__nprocs = u.trajectory.n_frames

            if self.__nprocs != 1: # Using multiple processors
                # Perform Calculation
                frame_chunks = [(chunk[0], chunk[-1]+1) for chunk in np.array_split(np.arange(u.trajectory.n_frames), self.__nprocs)]
                run_chunks = partial(_calc_hbonds, u=u, sel=sel, angle_cutoff=angle_cutoff, distance_cutoff=distance_cutoff)
                with multiprocessing.Pool(self.__nprocs) as worker_pool:
                    results = worker_pool.map(run_chunks, frame_chunks)
                self.raw_results = np.vstack(results)
            else: # Not using multiple processors
                self.raw_results = _calc_hbonds((None, None), u, sel, angle_cutoff=angle_cutoff, distance_cutoff=distance_cutoff)

    def process(self):
        '''
        Process raw hydrogen bond array to gather information about how to represent the hydrogen bonds on the map. Calling this method populates the processed_results attribute. 
        '''
        u = mda.Universe(self.__top, self.__traj)
        p = u.select_atoms("protein") # Protein atomgroup

        # Eliminate consideration of omitted layers (if any are present)
        sel = np.all(np.isin(p.atoms.segids[self.raw_results[:, (1,3)].astype(int)], self.__sysinfo.structure.flatten()), axis=1)
        hb = self.raw_results[sel, :][:, (0, 1, 3)].astype(int) # Just need frame, donor index, and acceptor index

        # FIND AVERAGE NUMBER OF HBONDS
        # Construct new array with the frame, protofilaments, residues, atom types (BB=0, SC=1 or T=2), and layer difference
        hb_details = np.column_stack((
            hb[:,0], # the frame
            self.__sysinfo.atom_info[hb[:,1],1:], # Donor info (Protofilament, Residue, Type)
            self.__sysinfo.atom_info[hb[:,2],1:], # Acceptor Info (Protofilament, Residue, Type)
            self.__sysinfo.atom_info[hb[:,1],0]-self.__sysinfo.atom_info[hb[:,2],0] # Donor Layer - Acceptor Layer
        )) # 0:Frame, 1: D_p, 2: D_r, 3: D_s, 4: A_p, 5: A_r, 6: A_s, 7: Layer Difference

        # hb details but only one of each layerdiff-and-site-wise type of hbond per frame. Layer diff is changed to layer span after unique types are found. Also includes IntraLvInterL and counts.
        hb_details_unique = np.column_stack(np.unique(np.column_stack((
            hb_details,
            hb_details[:,-1]
        )), axis=0, return_counts=True))
        hb_details_unique[hb_details_unique[:,-3]!=0,-3] = 1 # Interlayer (1) or Intralayer (0)
        hb_details_unique[:,-2] = np.absolute(hb_details_unique[:,-2]) # LayerSpan
        # 0:Frame, 1: D_p, 2: D_r, 3: D_s, 4: A_p, 5: A_r, 6: A_s, 7: IntraLvInterL, 8: LayerSpan, 9: Count

        # First, for each frame, we compute Nhb for each hbond if there is only one type of it (do it the quick way)
        hbn_details_unique, hbn_details_indices, hbn_details_inverse, hbn_details_counts = np.unique(hb_details_unique[:,:-2], axis=0, return_index=True, return_inverse=True, return_counts=True) # hbn_details_unique includes: 0:Frame, 1: D_p, 2: D_r, 3: D_s, 4: A_p, 5: A_r, 6: A_s, 7: IntraLvInterL
        selsingle = (hbn_details_counts == 1)
        hbn_details_single = hbn_details_unique[selsingle,:]
        hbn_single_indices = hbn_details_indices[selsingle]
        nhb_frames_single = hb_details_unique[hbn_single_indices,:][:,-1]/(self.__sysinfo.structure.shape[0]-hb_details_unique[hbn_single_indices,:][:,-2])

        # Second, for each frame, we compute Nhb for each hbond if there is more than one type of it (have to do it the slow way)
        search_array = hb_details_unique[hbn_details_counts[hbn_details_inverse]>1]
        hbn_details_multi= hbn_details_unique[hbn_details_counts>1]
        nhb_frames_multi = np.zeros(hbn_details_multi.shape[0])
        for i, unique_row in enumerate(tqdm(hbn_details_multi, leave=False, desc="N")):
            sel = np.all(search_array[:,:-2]==unique_row, axis=1)
            types = search_array[sel,-2] # LayerSpans
            nhb_frames_multi[i] = np.sum(search_array[sel,-1])*types.shape[0]/np.sum(self.__sysinfo.structure.shape[0]-types) # <N(HB)> = 2*(Number of Interactions)/(Number of Weighted Sites) where Number of Weighted Sites = sum_types^N_types(2*(N_layers-LayerSpan_type)/N_types)

        # Third, combine the results from first step and second step
        hbn_details_unique=np.vstack((hbn_details_multi, hbn_details_single))
        nhb_frames = np.hstack((nhb_frames_multi, nhb_frames_single))

        # Fourth, average over the trajectory
        hbn_unique = np.unique(hbn_details_unique[:,1:], axis=0)
        nhb = np.zeros(hbn_unique.shape[0])
        for i, unique_row in enumerate(hbn_unique):
            nhb[i] = np.sum(nhb_frames[np.all(hbn_details_unique[:,1:]==unique_row, axis=1)])/u.trajectory.n_frames
        
        # Fifth, combine interlayer and intralayer results
        processed_count_details = np.unique(hb_details[:, 1:-1], axis=0) # Cut off layer diff and frame
        processed_count_values = np.zeros((processed_count_details.shape[0], 2)) # Intralayer <N>, Interlayer <N>
        for i, unique_row in enumerate(processed_count_details):
            sel = np.all(hbn_unique[:,:-1]==unique_row, axis=1)
            layerdiffs = hbn_unique[sel, -1] # Either 0 (intralayer) or 1 (interlayer)
            
            for layerdiff, nval in zip(layerdiffs, nhb[sel]):
                processed_count_values[i, layerdiff] = nval
                
        # FIND PROBABILITIES
        processed_probability_results = _process_interactions(np.column_stack((
            hb[:,0], # Frames
            self.__sysinfo.atom_info[hb[:,1],:], # Donor
            self.__sysinfo.atom_info[hb[:,2],:] # Acceptor
        )), n_layers=self.__sysinfo.structure.shape[0], n_frames=u.trajectory.n_frames)

        # Combine Counts and Probabilities
        self.processed_results = np.column_stack((processed_count_details, 
                                                  processed_count_values[:,0],
                                                  processed_probability_results[:,-2],
                                                  processed_count_values[:,1],
                                                  processed_probability_results[:,-1]
        ))

    def save(self, processed_file=None, unprocessed_file=None):
        '''
        Save hydrogen bonds

        Parameters
        ----------
        processed_file : str or None
            name of file to save processed results. If None, processed results are not saved.
        unprocessed_file : str or None
            name of file to save unprocessed results. If None, unprocessed results are not saved.
        '''
        # Save Processed Results
        if processed_file is not None:
            np.save(processed_file, self.processed_results)
        
        # Save Raw Results
        if unprocessed_file is not None:
            np.save(unprocessed_file, self.raw_results)

    def show(self, LOG):
        '''
        Print hydrogen bonds

        Parameters
        ----------
        LOG : io.Logger
            Custom logger for output
        '''
        # if self.processed_results.size == 0:
        types = ["BB", "SC", "T"]
        if np.sum(self.processed_results[:,(7,9)]>=0.5) == 0:
            LOG.output(f"NONE")
        else:
            LOG.output(f"{'INTRALAYER':>44}{'INTERLAYER':>26}")
            LOG.output(f"{'DONOR':>10}{'ACCEPTOR':>13}{'<N H-BONDS>':>16}{'P(H-BOND)':>11}{'<N H-BONDS>':>15}{'P(H-BOND)':>11}")
            for hb in self.processed_results:
                if hb[7] > 0.5 or hb[9] > 0.5:
                    pintra = f"{np.round(hb[7], 3):.3f}"
                    pinter = f"{np.round(hb[9], 3):.3f}"
                    nintra = f"{np.round(hb[6], 3):.3f}"
                    ninter = f"{np.round(hb[8], 3):.3f}"
                    d = f'{int(hb[0])}-{self.__sysinfo.get_residue(self.__sysinfo.segment_resids[int(hb[1])-1])}-{types[int(hb[2])]}'
                    a = f'{int(hb[3])}-{self.__sysinfo.get_residue(self.__sysinfo.segment_resids[int(hb[4])-1])}-{types[int(hb[5])]}'
                    LOG.output(f"{d:>12}{a:>12}{nintra:>12}{pintra:>12}{ninter:>14}{pinter:>12}")

# ------------------------------------------------------------------------------
# SALT BRIDGES
# ------------------------------------------------------------------------------
def auto_sb_finder(ag, charge_cutoff, findtype, sysinfo):
    '''
    Automatically find cation or anion atom groups for salt bridges

    Parameters
    ----------
    ag : MDAnalysis.atomgroup
        An atom group containing the residues to search
    charge_cutoff : float > 0
        The charge cutoff for a potential salt bridge forming region
    findtype : str ("anion" or "cation")
        The type of group you are trying to find
    sysinfo : utils.SystemInfo
        Object containing system information

    Returns
    -------
    sel : str
        An MDAnalysis selection command to select the potential salt bridge formers of this type
    autosb_string : str
        Output for auto sb finder
    '''
    needed_rescharge = {"anion":-1.0, "cation":1.0}[findtype.lower()]
    cutsign = {"anion":"<", "cation":">"}[findtype.lower()]
    checked_resids, sel_collection, autosb_string = [], [], []
    for residue in ag.residues:
        if residue.resid not in checked_resids:
            rescharge = int(np.round(np.sum(residue.atoms.charges)))
            if rescharge == needed_rescharge:
                ressc = residue.atoms.select_atoms(f"(not backbone or {' or '.join(sysinfo.terminal_atom_names)}) and not name H*") # the sidechain atoms of this residue
                groupatomnames = []
                groupchargedict = {}
                for heavyatom in ressc:
                    is_backbone = False
                    for bond in heavyatom.bonds: # check to make sure this atom isn't actually a backbone atom.
                        for otheratom in bond.atoms:
                            if otheratom.index != heavyatom.index:
                                if otheratom.resid != heavyatom.resid:
                                    is_backbone = True
                                    break
                        if is_backbone:
                            break
                    if not is_backbone:
                        groupcharge = heavyatom.charge
                        for bond in heavyatom.bonds:
                            for bonded_atom in bond.atoms:
                                if bonded_atom.name[0] == "H":
                                    groupcharge += bonded_atom.charge
                        groupchargedict[heavyatom.name] = groupcharge
                        if (findtype.lower() == "cation" and groupcharge > charge_cutoff) or (findtype.lower() == "anion" and groupcharge < charge_cutoff):
                            groupatomnames.append(heavyatom.name)
                if len(groupatomnames) == 0:
                    if findtype == "cation":
                        maxgroupcharge = np.max(groupchargedict.values())
                        maxgroupchargename = list(groupchargedict.keys())[np.argmax(groupchargedict.values())]
                    elif findtype == "anion":
                        maxgroupcharge = np.min(groupchargedict.values())
                        maxgroupchargename = list(groupchargedict.keys())[np.argmin(groupchargedict.values())]
                    autosb_string.append(f"WARNING: Could not find charged atom group for {findtype} {residue.resname}{residue.resid} (total charge: {rescharge}) with a charge within the cutoff ({cutsign}{charge_cutoff}). Using the group closest to the cutoff: {maxgroupchargename} (group charge = {maxgroupcharge}). You might want to adjust the charge cutoff.\n")
                    groupatomnames.append(maxgroupchargename)
                sel_collection.append(f"(resid {residue.resid} and name {' '.join(groupatomnames)})")
            checked_resids.append(residue.resid)
    if len(sel_collection) == 0:
        autosb_string.append(f"WARNING: Could not find any {findtype}s. Salt bridges will not be computed because none could form. If this is a mistake, you could try manually defining the potential salt bridge participants.\n")
        return None, autosb_string
    else:
        sel = " or ".join(sel_collection)
        found_resnames = []
        autosb_string.append(f"{findtype.upper():>6}S: RESNAME   ATOM NAMES")
        for s in sel_collection:
            sag = ag.select_atoms(s)
            sresname = sag[0].residue.resname
            if sresname not in found_resnames:
                atomnames = sag.residues[0].atoms.select_atoms(s).names
                autosb_string.append(f"{sresname:>14}   {' '.join(atomnames)}")
            found_resnames.append(sresname)

        return sel, "\n".join(autosb_string)

class SaltBridgeCalculator:
    '''
    Class for computing and sorting salt bridges

    Attributes
    ----------
    raw_results : numpy.ndarray
        Raw salt bridge results. Each row contains (in order) the frame, anionic layer index (one-based), anionic protofilament index (one-based), anionic residue index (one-based), anionic atom type (1=sidechain, 2=terminus), cationic layer index (one-based), cationic protofilament index (one-based), cationic residue index (one-based), cationic atom type (1=sidechain, 2=terminus), and the distance between the charged groups.
    processed_results : numpy.ndarray or None
        Processed salt bridge results. Each row contains (in order) the anion protofilament index (one-based), the anion residue (one-based), the anion atom type (1=sidechain, 2=terminus), the cation protofilament index (one-based), the cation residue index (one-based), the cation atom type (1=sidechain, 2=terminus), intralayer P(SaltBridge),  and interlayer P(SaltBridge). This attribute will not be populated until SaltBridgeCalculator.process() is called.
    warned : bool
        If a warning was thrown this will be set to true
    
    Methods
    -------
    process()
        Process raw_results. Populates intramolecular, interlayer, and interprotofilament attributes.
    save(maindirectory=None, rawdirectory=None)
        If maindirectory is not None, save processed results to it. If rawdirectory is not None, save raw results to it.
    show()
        Print summary of processed results to terminal.

    Usage
    -----
    1) Perform initial calculation
    saltbridges = SaltBridgeCalculator(topology_file, trajectory_file, system_info)

    # You can now save the raw results
    saltbridges.save(unprocessed_file='PATH/TO/RESULTS/unprocessed_salt_bridges.npy')

    2) Process the raw results
    saltbridges.process()

    # You can now save the processed results
    saltbridges.save(processed_file='PATH/TO/RESULTS/processed_salt_bridges.npy')

    # You can now show the results
    saltbridges.show()

    '''
    def __init__(self, topology_file, trajectory_file, system_info, anion_sel, cation_sel, unprocessed_filename=None, cutoff_distance=4.0):
        '''
        Parameters
        ----------
        topology_file : str
            Name of topology file
        trajectory_file : str or [str]
            Name(s) of trajectory file(s)
        system_info : utils.SystemInfo
            System Information
        anion_sel : str
            MDAnalysis selection string for salt bridge anion groups
        cation_sel : str
            MDAnalysis selection string for salt bridge cation groups
        unprocessed_filename : Str or None (Default: None)
            Name of file containing previously calculated unprocessed results. This filename should come only from the Params class. If None, salt bridges will be calculated.
        cutoff_distance : float
            Maximum distance for a saltbridge
        '''
        # Input Parameters
        self.__top = topology_file
        self.__traj = trajectory_file
        self.__sysinfo = system_info

        # Initialize Empty Objects for Parsed Results
        self.intramolecular = None
        self.interprotofilament = None
        self.interlayer = None

        u = mda.Universe(self.__top, self.__traj)
        self.__n_frames = u.trajectory.n_frames 
        
        if unprocessed_filename is not None:
            self.raw_results = np.load(unprocessed_filename)
        else:
            sel = f"segid {' '.join(self.__sysinfo.structure.flatten())}"
            anions = u.select_atoms(f'{anion_sel} and {sel}')
            cations = u.select_atoms(f'{cation_sel} and {sel}')
            
            if anions is not None and cations is not None:
                anionic_layers = np.tile(self.__sysinfo.atom_info[anions.indices,0], (cations.n_atoms, 1)).T
                anionic_protofilaments = np.tile(self.__sysinfo.atom_info[anions.indices,1], (cations.n_atoms, 1)).T
                anionic_residues = np.tile(self.__sysinfo.atom_info[anions.indices,2], (cations.n_atoms, 1)).T
                anionic_types = np.tile(self.__sysinfo.atom_info[anions.indices,3], (cations.n_atoms, 1)).T

                cationic_layers = np.tile(self.__sysinfo.atom_info[cations.indices,0], (anions.n_atoms, 1))
                cationic_protofilaments = np.tile(self.__sysinfo.atom_info[cations.indices,1], (anions.n_atoms, 1))
                cationic_residues = np.tile(self.__sysinfo.atom_info[cations.indices,2], (anions.n_atoms, 1))
                cationic_types = np.tile(self.__sysinfo.atom_info[cations.indices,3], (anions.n_atoms, 1))

                if self.__n_frames == 1:
                    dists = distances.distance_array(anions.positions, cations.positions)
                    sel = dists <= cutoff_distance
                    sbs = np.column_stack([np.full(np.sum(sel), 0),
                                        anionic_layers[sel], anionic_protofilaments[sel], anionic_residues[sel], anionic_types[sel],
                                        cationic_layers[sel], cationic_protofilaments[sel], cationic_residues[sel], cationic_types[sel], dists[sel]])
                    self.raw_results = np.unique(sbs, axis=0)
                else:
                    self.raw_results = []
                    for ts in tqdm(u.trajectory, leave=False):
                        dists = distances.distance_array(anions.positions, cations.positions)
                        sel = dists <= cutoff_distance
                        sbs = np.column_stack([np.full(np.sum(sel), ts.frame),
                                            anionic_layers[sel], anionic_protofilaments[sel], anionic_residues[sel], anionic_types[sel],
                                            cationic_layers[sel], cationic_protofilaments[sel], cationic_residues[sel], cationic_types[sel], dists[sel]])
                        self.raw_results.append(np.unique(sbs, axis=0))

                    self.raw_results = np.vstack(self.raw_results)

    def process(self):
        '''
        Process raw salt bridge array to gather information about how to represent the salt bridges on the map. Calling this method populates the processed_results attribute. 
        '''
        # Calculate Probabilities
        self.processed_results = _process_interactions(self.raw_results[:,:-1], n_layers=self.__sysinfo.structure.shape[0], n_frames=self.__n_frames)
        # self.processed_results:: 0:A_p, 1: A_r, 2: A_t, 3: C_p, 4: C_r, 5: C_t, 6: Intralayer <P_SB>, 7: Interlayer <P_SB>

    def save(self, processed_file=None, unprocessed_file=None):
        '''
        Save salt bridges

        Parameters
        ----------
        processed_file : str or None
            name of file to save processed results. If None, processed results are not saved.
        unprocessed_file : str or None
            name of file to save unprocessed results. If None, unprocessed results are not saved.
        '''
        # Save Processed Results
        if processed_file is not None:
            np.save(processed_file, self.processed_results)
        
        # Save Raw Results
        if unprocessed_file is not None:
            np.save(unprocessed_file, self.raw_results)
    
    def show(self, LOG):
        '''
        Print salt bridges
        '''
        types = ["BB", "SC", "T"]
        if np.sum(self.processed_results[:,(6,7)]>=0.5) == 0:
        # if self.processed_results.size == 0:
            LOG.output("NONE")
        else:
            LOG.output(f"{'ANION':>10}{'CATION':>13}{'Intralayer P(SB)':>19}{'Interlayer P(SB)':>19}")
            for sb in self.processed_results:
                if sb[-2] > 0.1 or sb[-1] > 0.1:
                    a = f'{int(sb[0])}-{self.__sysinfo.get_residue(self.__sysinfo.segment_resids[int(sb[1])-1])}-{types[int(sb[2])]}'
                    c = f'{int(sb[3])}-{self.__sysinfo.get_residue(self.__sysinfo.segment_resids[int(sb[4])-1])}-{types[int(sb[5])]}'
                    pintra = f"{np.round(sb[6], 3):.3f}"
                    pinter = f"{np.round(sb[7], 3):.3f}"
                    LOG.output(f"{a:>12}{c:>12}{pintra:>12}{pinter:>19}")

# ------------------------------------------------------------------------------
# PI-PI STACKING
# ------------------------------------------------------------------------------
def _centroids_and_normals(PHE, TYR, HIS, TRP):
    '''
    Calculate ring centroids & normals
    '''
    PHEpo = PHE.positions.reshape(PHE.residues.n_residues, 6, 3)
    TYRpo = TYR.positions.reshape(TYR.residues.n_residues, 6, 3)
    HISpo = HIS.positions.reshape(HIS.residues.n_residues, 5, 3)
    TRPpo = TRP.positions.reshape(TRP.residues.n_residues, 5, 3)
    PHEcentroids = np.mean(PHEpo, axis=1)
    TYRcentroids = np.mean(TYRpo, axis=1)
    HIScentroids = np.mean(HISpo, axis=1)
    TRPcentroids = np.mean(TRPpo, axis=1)
    PHEnormals = np.linalg.svd(np.transpose(PHEpo-PHEcentroids.reshape(PHEcentroids.shape[0], 1, PHEcentroids.shape[1]), axes=(0, 2, 1)))[0][:,:,-1]
    TYRnormals = np.linalg.svd(np.transpose(TYRpo-TYRcentroids.reshape(TYRcentroids.shape[0], 1, TYRcentroids.shape[1]), axes=(0, 2, 1)))[0][:,:,-1]
    HISnormals = np.linalg.svd(np.transpose(HISpo-HIScentroids.reshape(HIScentroids.shape[0], 1, HIScentroids.shape[1]), axes=(0, 2, 1)))[0][:,:,-1]
    TRPnormals = np.linalg.svd(np.transpose(TRPpo-TRPcentroids.reshape(TRPcentroids.shape[0], 1, TRPcentroids.shape[1]), axes=(0, 2, 1)))[0][:,:,-1]

    return np.vstack((PHEcentroids, TYRcentroids, HIScentroids, TRPcentroids)), np.vstack((PHEnormals, TYRnormals, HISnormals, TRPnormals))

def _gamma(a, b):
    '''
    gamma: acute angle between aromatic ring planes of ring 1 and ring 2
    '''
    return utils.acute_angle(a, b)

def _delta(a, b):
    '''
    delta: acute angle between centroid-centroid vector and the aromatic ring plane of ring 2
    '''
    v = b[:3]-a[:3] # Centroid-Centroid Vector
    v = v/np.linalg.norm(v) # Centroid-Centroid Unit Vector
    ccn_angle = utils.acute_angle(v, b[3:]) # Angle between centroid-centroid vector and normal of ring 2
    return 90 - ccn_angle # Angle between centroid-centroid vector and plane of ring 2
    # 
def _theta(a, b):
    '''
    theta: acute angle between centroid-centroid vector and the aromatic ring plane of ring 1
    '''
    v = b[:3]-a[:3] # Centroid-Centroid Vector
    v = v/np.linalg.norm(v) # Centroid-Centroid Unit Vector
    ccn_angle = utils.acute_angle(v, a[3:]) # Angle between centroid-centroid vector and normal of ring 1
    return 90 - ccn_angle # Angle between centroid-centroid vector and plane of ring 1

def _calc_pipi(frame_chunk, u, sel, pairinfo, phe_sel="(resname PHE and name CG CD2 CE2 CZ CE1 CD1)", tyr_sel="(resname TYR and name CG CD2 CE2 CZ CE1 CD1)", his_sel="(resname HSD HSE HSP and name CG CD2 NE2 CE1 ND1)", trp_sel="(resname TRP and name CG CD1 NE1 CE2 CD2)"):
    '''
    Calculate pi stacking interactions for a chunk of frames

    Parameters
    ----------
    frame_chunk : tuple
        first and last frame of chunk: (first_frame, last_frame)
    u : MDAnalysis.universe
        universe in which interactions are to be computed
    sel : str
        extra selection string to restrict which atoms are considered
    pairinfo : np.ndarray
        Information about residue pairs
    phe_sel : str (Default: resname PHE and name CG CD2 CE2 CZ CE1 CD1)
        The MDAnalysis selection command for phenylalanine rings
    tyr_sel : str (Default: resname TYR and name CG CD2 CE2 CZ CE1 CD1)
        The MDAnalysis selection command for tyrosine rings
    his_sel : str (Default: resname HSD HSE HSP and name CG CD2 NE2 CE1 ND1)
        The MDAnalysis selection command for histidine rings
    trp_sel : str (Default: resname TRP and name CG CD1 NE1 CE2 CD2)
        The MDAnalysis selection command for tryptophan rings

    Returns
    -------
    results : numpy.ndarray
        identifiers for each pipi interaction. Each row includes (in order) frame, layer 1, protofilament 1, resid 1, layer 2, protofilament 2, residue 2, centroid-centroid distance, gamma angle, delta angle, and theta angle.
    '''
    PHE = u.select_atoms(f"{phe_sel} and {sel}") # The 6 membered ring
    TYR = u.select_atoms(f"{tyr_sel} and {sel}") # The 6 membered ring
    HIS = u.select_atoms(f"{his_sel} and {sel}") # The 5 membered ring
    TRP = u.select_atoms(f"{trp_sel} and {sel}") # The 5 membered ring
    
    results = []
    for ts in tqdm(u.trajectory[frame_chunk[0]:frame_chunk[1]], leave=False):
        centroids, normals = _centroids_and_normals(PHE, TYR, HIS, TRP)
        c_and_n = np.hstack((centroids, normals))

        # Rcen: Centroid-Centroid Distance
        Rcen = pdist(centroids, 'euclidean')
        # gamma: acute angle between aromatic ring planes of ring 1 and ring 2
        gamma = pdist(normals, _gamma)
        # delta: acute angle between centroid-centroid vector and the aromatic ring plane of ring 2
        delta = pdist(c_and_n, _delta)
        # theta: acute angle between centroid-centroid vector and the aromatic ring plane of ring 1
        theta = pdist(c_and_n, _theta)

        # toss out any non-pipi interactions
        # pipi interactions will have a centroid distance of 7.2 Angstroms or less.
        # Also, if delta < 30 and theta < 30 it is a spurious interaction (not pi-pi)
        # therefore, for a pipi interaction either delta or theta must be >= 30
        sel_pipi = (Rcen < 7.2) & ((delta >= 30) | (theta >= 30))
        results.append(np.column_stack((np.full(sum(sel_pipi), ts.frame), pairinfo[sel_pipi],
                                                 Rcen[sel_pipi], gamma[sel_pipi], delta[sel_pipi], theta[sel_pipi])))
    if len(results) == 1 and type(results[0]) == np.ndarray:
        results = results[0]
    else:
        results = np.vstack(results)
    
    return results

def _classify_pipi(pipi_results):
    '''
    Classify pipi interactions using their metrics as Sandwich (S), T-Shaped (T),
    Parallel Displaced (D), or Intermediate (I).

    Parameters
    ----------
    pipi_results : numpy.ndarray
        results from _calc_pipi()

    Returns
    -------
    pipi_results : numpy.ndarray
        pipiresults with an additional column detailing type of interaction: 0=T-Shaped, 1=Intermediate, 2=Sandwich, 3=Parallel Displaced
    '''
    classes = np.full(pipi_results.shape[0], 4)

    # T-Shaped : gamma > 50 degrees
    T_sel = pipi_results[:, 8] > 50
    classes[T_sel] = 0

    # Intermediate : 30 degrees <= gamma <= 50 degrees
    I_sel = (pipi_results[:, 8] >= 30) & (pipi_results[:, 8] <= 50)
    classes[I_sel] = 1

    # Sandwich : gamma < 30 degrees, and theta > 80 degrees or delta > 80 degrees
    S_sel = (pipi_results[:, 8] <= 30) & ((pipi_results[:, 10] > 80) | (pipi_results[:, 9] > 80))
    classes[S_sel] = 2

    # Parallel Displaced : gamma < 30 degrees, and theta <= 80 degrees, and delta <= 80 degrees
    D_sel = (pipi_results[:, 8] <= 30) & (pipi_results[:, 10] <= 80) & (pipi_results[:, 9] <= 80)
    classes[D_sel] = 3

    # Check to make sure all were classified
    if np.any(classes==4):
        SystemExit("Error: Pi Stacking Classifier: Could not determine types of all pi stacking interactions.")

    pipi_results = np.column_stack((pipi_results, classes))
    return pipi_results

class PiStackingCalculator:
    '''
    Class for computing and sorting pi pi stacking interactions (Sandwich, T-Shaped, Parallel Displaced, & Intermediate)
    within a fibril

    ** Only set up to work with CHARMM forcefield

    This scheme follows the scheme presented here:
    Zhao, Y., Li, J., Gu, H. et al. Interdiscip Sci Comput Life Sci
    7, 211–220 (2015). https://doi.org/10.1007/s12539-015-0263-z

    Attributes
    ----------
    raw_results : numpy.ndarray
        Unprocessed pipi interactions. Each row includes (in order): the frame, layer A, protofilament A, residue A, layer B, protofilament B, residue B, centroid-centroid distance, gamma angle, delta angle, theta angle, and type (0=Sandwich, 1=parallel displaced, 2=Intermediate, 3=T-shaped)
    intramolecular, interlayer, interprotofilament : numpy.ndarray or None
        Processed pi stacking results with residue information broken into three arrays for intramolecular, interlayer, and interprotofilament. Each row contains (in order): Protofilament A, Residue A, Protofilament B, Residue B, Intralayer P(Total), Intralayer P(T-Shaped), Intralayer P(Intermediate), Intralayer P(Sandwich), Intralayer P(Parallel Displaced), Interlayer P(Total), Interlayer P(T-Shaped), Interlayer P(Intermediate), Interlayer P(Sandwich), Interlayer P(Parallel Displaced)
    
    Methods
    -------
    process()
        Process raw_results. Populates intramolecular, interlayer, and interprotofilament attributes.
    save(maindirectory=None, rawdirectory=None)
        If maindirectory is not None, save processed results to it. If rawdirectory is not None, save raw results to it.
    show()
        Print summary of processed results to terminal.

    Usage
    -----
    1) Perform initial calculation
    pistacking = PiStackingCalculator(topology_file, trajectory_file, nprocs, system_info)

    # You can now save the raw results
    pistacking.save(unprocessed_file='PATH/TO/RESULTS/unprocessed_pistacking_interactions.npy')

    2) Process the raw results
    pistacking.process()

    # You can now save the processed results
    pistacking.save(processed_file='PATH/TO/RESULTS/processed_pistacking_interactions.npy')

    # You can now show the results
    pistacking.show()

    '''
    def __init__(self, topology_file, trajectory_file, nprocs, system_info, unprocessed_filename=None, phe_sel="(resname PHE and name CG CD2 CE2 CZ CE1 CD1)", tyr_sel="(resname TYR and name CG CD2 CE2 CZ CE1 CD1)", his_sel="(resname HSD HSE HSP and name CG CD2 NE2 CE1 ND1)", trp_sel="(resname TRP and name CG CD1 NE1 CE2 CD2)"):
        '''
        Parameters
        ----------
        topology_file : str
            Name of topology file
        trajectory_file : str or [str]
            Name(s) of trajectory file(s)
        nprocs : int
            Number of processors to use
        system_info : utils.SystemInfo
            System information
        unprocessed_filename : Str or None
            Name of file containing previously calculated unprocessed results. This filename should come only from the Params class. If None, salt bridges will be calculated.
        phe_sel : str (Default: resname PHE and name CG CD2 CE2 CZ CE1 CD1)
            The MDAnalysis selection command for phenylalanine rings
        tyr_sel : str (Default: resname TYR and name CG CD2 CE2 CZ CE1 CD1)
            The MDAnalysis selection command for tyrosine rings
        his_sel : str (Default: resname HSD HSE HSP and name CG CD2 NE2 CE1 ND1)
            The MDAnalysis selection command for histidine rings
        trp_sel : str (Default: resname TRP and name CG CD1 NE1 CE2 CD2)
            The MDAnalysis selection command for tryptophan rings
        
        '''
        self.__top = topology_file
        self.__traj = trajectory_file
        self.__sysinfo = system_info

        u = mda.Universe(self.__top, self.__traj)
        self.__n_frames = u.trajectory.n_frames

        if unprocessed_filename is not None:
            self.raw_results = np.load(unprocessed_filename)
        else:
            sel = f"segid {' '.join(self.__sysinfo.structure.flatten())}"
            PHE = u.select_atoms(f"{phe_sel} and {sel}") # The 6 membered ring
            TYR = u.select_atoms(f"{tyr_sel} and {sel}") # The 6 membered ring
            HIS = u.select_atoms(f"{his_sel} and {sel}") # The 5 membered ring
            TRP = u.select_atoms(f"{trp_sel} and {sel}") # The 5 membered ring

            PHEatomsel = [residue.atoms[0].index for residue in PHE.residues]
            TYRatomsel = [residue.atoms[0].index for residue in TYR.residues]
            HISatomsel = [residue.atoms[0].index for residue in HIS.residues]
            TRPatomsel = [residue.atoms[0].index for residue in TRP.residues]
            resinfo = np.vstack((self.__sysinfo.atom_info[PHEatomsel,:3],
                                 self.__sysinfo.atom_info[TYRatomsel,:3],
                                 self.__sysinfo.atom_info[HISatomsel,:3],
                                 self.__sysinfo.atom_info[TRPatomsel,:3]))

            pairinfo = []
            for i, (Li, Pi, Ri) in enumerate(resinfo):
                for Lj, Pj, Rj in resinfo[i+1:]:
                    pairinfo.append(np.array([Li, Pi, Ri, Lj, Pj, Rj]))
            pairinfo = np.array(pairinfo)

            # Can't use more processors than there are frames
            if nprocs > u.trajectory.n_frames:
                use_nprocs = u.trajectory.n_frames
            else:
                use_nprocs = nprocs

            if use_nprocs != 1: # Multiprocess
                # Perform Calculation
                frame_chunks = [(chunk[0], chunk[-1]+1) for chunk in np.array_split(np.arange(u.trajectory.n_frames), use_nprocs)]
                run_chunks = partial(_calc_pipi, u=u, sel=sel, pairinfo=pairinfo, phe_sel=phe_sel, tyr_sel=tyr_sel, his_sel=his_sel, trp_sel=trp_sel)
                with multiprocessing.Pool(use_nprocs) as worker_pool:
                    results = worker_pool.map(run_chunks, frame_chunks)
                results = np.vstack(results)
            else: # Not multiprocessed
                results = _calc_pipi((None, None), u=u, sel=sel, pairinfo=pairinfo, phe_sel=phe_sel, tyr_sel=tyr_sel, his_sel=his_sel, trp_sel=trp_sel)

            self.raw_results = _classify_pipi(results)

    def process(self):
        '''
        Process raw pi stacking array to gather information about how to represent the pi stacking interactions on the map. Calling this method populates the processed_results attribute.
        '''
        # Calculate Total Probabilities
        total_probabilities = _process_interactions(self.raw_results[:,:7], n_layers=self.__sysinfo.structure.shape[0], n_frames=self.__n_frames, combine_intraL_interL=False)
        # total_probabilities:: 0:D_p, 1: D_r, 2: A_p, 3: A_r, 4: IntraLvInterL, 5: <P_SB>

        # Calculate Type Probabilities
        # These probabilities tell you the probability of each kind of pi stacking interaction when it forms
        search_array = np.column_stack((
            self.raw_results[:,(2,3,5,6)],
            self.raw_results[:,4]-self.raw_results[:,1],
            self.raw_results[:,-1] # Type
        ))
        search_array[search_array[:,-2]!=0,-2] = 1 # Interlayer (1) or Intralayer (0)
        # search_array:: 0:D_p, 1: D_r, 2: A_p, 3: A_r, 4: IntraLvInterL, 5: Type

        type_p_array = np.zeros((total_probabilities.shape[0],4))
        for i, pii in enumerate(total_probabilities):
            selp = np.all(search_array[:,:-1]==pii[:-1], axis=1) # Search for matching residue pairs and distinguish between intraL and interL
            found_types = search_array[selp,-1].astype(int) # Get the types
            unique_types = np.unique(found_types)
            n_total = found_types.shape[0]
            for ti in unique_types: # Calculate prob. of each type if it is found at all (otherwise it's zero): 0=T-Shaped, 1=Intermediate, 2=Sandwich, 3=Parallel Displaced
                type_p_array[i, ti] = np.sum(found_types==ti)/n_total
            search_array = search_array[np.invert(selp)] # Don't search the same values again
        # type_p_array :: 0:T-Shaped, 1:Intermediate, 2:Sandwich, 3:Parallel Displaced

        # Combine intralayer and interlayer
        unique_pairs = np.unique(total_probabilities[:,:-2], axis=0)
        all_probabilities = np.zeros((unique_pairs.shape[0],10))
        for i, unique_row in enumerate(unique_pairs):
            sel = np.all(total_probabilities[:,:-2]==unique_row, axis=1)
            for layerdiff, pvals in zip(total_probabilities[sel,4], np.column_stack((total_probabilities[sel,5], type_p_array[sel,:]))):
                if layerdiff == 0: # Intralayer
                    all_probabilities[i, :5] = pvals
                else: # Interlayer
                    all_probabilities[i, 5:] = pvals
        
        self.processed_results = np.column_stack((
            unique_pairs,
            all_probabilities
        ))
        # 0: Protofilament A, 1: Residue A, 2: Protofilament B, 3: Residue B, 4: Intralayer P(Total), 5: Intralayer P(T-Shaped), 6: Intralayer P(Intermediate), 7: Intralayer P(Sandwich), 8: Intralayer P(Parallel Displaced), 9: Interlayer P(Total), 10: Interlayer P(T-Shaped), 11: Interlayer P(Intermediate), 12: Interlayer P(Sandwich), 13: Interlayer P(Parallel Displaced)

    def save(self, processed_file=None, unprocessed_file=None):
        '''
        Save pi stacking interactions

        Parameters
        ----------
        processed_file : str or None
            name of file to save processed results. If none, processed results are not saved. NPZ file
        unprocessed_file : str or None
            name of file to save raw results. If none, raw results are not saved. NPZ file
        '''
        # Save Processed Results
        if processed_file is not None:
            np.save(processed_file, self.processed_results)
        
        # Save Raw Results
        if unprocessed_file is not None:
            np.save(unprocessed_file, self.raw_results)

    def show(self, LOG):
        '''
        Log pi stacking interactons
        '''
        if np.sum(self.processed_results[:,(4,9)]>=0) == 0:
            LOG.output("NONE")
        else:
            LOG.output(f"{'Intralayer Probabilities':>56}{'Interlayer Probabilities':>39}")
            LOG.output(f"{'RESIDUE-A':>12}{'RESIDUE-B':>12}{'Total':>9}{'T':>5}{'I':>7}{'S':>7}{'D':>7}{'Total':>12}{'T':>5}{'I':>7}{'S':>7}{'D':>7}")
            for pipi in self.processed_results:
                if pipi[4] >= 0 or pipi[9] >= 0:
                    a = f'{int(pipi[0])}-{self.__sysinfo.get_residue(self.__sysinfo.segment_resids[int(pipi[1])-1])}'
                    b = f'{int(pipi[2])}-{self.__sysinfo.get_residue(self.__sysinfo.segment_resids[int(pipi[3])-1])}'
                    pintra_tot = f"{np.round(pipi[4], 3):.3f}"
                    pintra_T = f"{np.round(pipi[5], 3):.3f}"
                    pintra_I = f"{np.round(pipi[6], 3):.3f}"
                    pintra_S = f"{np.round(pipi[7], 3):.3f}"
                    pintra_D = f"{np.round(pipi[8], 3):.3f}"
                    pinter_tot = f"{np.round(pipi[9], 3):.3f}"
                    pinter_T = f"{np.round(pipi[10], 3):.3f}"
                    pinter_I = f"{np.round(pipi[11], 3):.3f}"
                    pinter_S = f"{np.round(pipi[12], 3):.3f}"
                    pinter_D = f"{np.round(pipi[13], 3):.3f}"
                    LOG.output(f"{a:>11}{b:>12}{pintra_tot:>10}{pintra_T:>7}{pintra_I:>7}{pintra_S:>7}{pintra_D:>7}{pinter_tot:>10}{pinter_T:>7}{pinter_I:>7}{pinter_S:>7}{pinter_D:>7}")

