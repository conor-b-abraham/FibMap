import numpy as np
import MDAnalysis as mda
import sys
import argparse
import os
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# ARGUMENTS
parser = argparse.ArgumentParser(description='Rename Segments in PDB')
parser.add_argument('-i', '--inputfile', required=True, help='Input PDB File (From CHARMM-GUI pdbreader)')
parser.add_argument('-c', '--charge_inputfile', default=None, help='(Optional) A second input file containing charge information (e.g. PSF file). This will prompt the program to also write a psf file for the reordered system.')
parser.add_argument('-o', '--output_namestem', required=True, help='Name stem for output files')
parser.add_argument('-m', '--makefig', action="store_true", help="Create a figure to help with segid assignment")
args = parser.parse_args()
IN = args.inputfile
CIN = args.charge_inputfile
OUT = args.output_namestem
MAKEFIG = args.makefig

if IN[-4:] != '.pdb':
    print(f'INPUT ERROR: file specified by --inputfile/-i must be a pdb, so {IN} is not a valid input file.\n')
    exit=True
elif not os.path.isfile(IN):
    print(f'INPUT ERROR: {IN} does not exist.\n')
    exit=True
else:
    exit=False

if CIN is not None and not os.path.isfile(CIN):
    print(f'INPUT ERROR: {CIN} does not exist.\n')
    exit=True
else:
    exit=False

if exit:
    parser.print_help()
    sys.exit('\nPlease try again with valid input files')
else:
    if CIN is None:
        print(f'\nRenaming residues in {IN}')
    if CIN is not None:
        print(f'\nRenaming residues in {IN} and {CIN}')

# MAIN
if CIN is None:
    u = mda.Universe(IN)
else:
    u = mda.Universe(CIN, IN)

if MAKEFIG:
    axisnames = ["X", "Y", "Z"]
    dimgrab = [[1,2],[0,1],[0,2]]
    fig, axes = plt.subplots(1, 3)
    for i, (ax, dims) in enumerate(zip(axes, dimgrab)):
        xcoms, ycoms = [], []
        for segment in u.select_atoms("protein").segments:
            COM = segment.atoms.center_of_mass()
            xcoms.append(COM[dims[0]])
            ycoms.append(COM[dims[1]])

            ax.plot(COM[dims[0]], COM[dims[1]], color="white")
            ax.annotate(segment.segid, (COM[dims[0]], COM[dims[1]]), ha='center', va='center_baseline')
        
        # Adjust axis labels
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        ax.set_title(f"{axisnames[dims[0]]}{axisnames[dims[1]]}-plane")

        # Adjust axis lims
        minx = np.min(xcoms)
        maxx = np.max(xcoms)
        xspanfactor = (maxx-minx)*0.2
        ax.set_xlim(minx-xspanfactor, maxx+xspanfactor)
        miny = np.min(ycoms)
        maxy = np.max(ycoms)
        yspanfactor = (maxy-miny)*0.05
        ax.set_ylim(miny-yspanfactor, maxy+yspanfactor)
    
    plt.savefig(f"{OUT}_segment_locs.png", dpi=300)
    print(f"\nSegment order image saved to {OUT}_segment_locs.png")
    plt.show()

instruct = '\nAdd Segment IDs of a protofibril or terminate and write results:\n'+\
           '    To terminate and write results hit [ENTER]\n'+\
           '    OR add new protofilament by typing segids in order along the principle axis of the fibril\n'+\
           '       and separate the segids by a space. i.e. "PROA PROC PROE"\n> '
prompt = 'start'
segnames = []
while prompt != '':
    prompt = input(instruct)
    if prompt != '':
        segids = prompt.split()
        tryagain = False
        for segid in segids:
            try:
                seg = u.select_atoms(f'segid {segid}')
                if seg.atoms.n_atoms == 0:
                    print(f'Segment {segid} does not exist in input PDB. Try again')
                    tryagain = True
            except:
                print(f'Segment {segid} does not exist in input PDB. Try again')
                tryagain = True
        if not tryagain:
            segnames.append(segids)
    else:
        print()

pflens = np.array([len(i) for i in segnames])
if not np.all(pflens==pflens[0]):
    sys.exit('ERROR: Protofilaments lengths do not match. Try running the program again.')

exit = False
for i in segnames[0]:
    if len(segnames) > 1 and i in segnames[1]:
        print(f'ERROR: segment {i} was included twice.')
        exit = True
if exit:
    sys.exit('Segments cannot be included more than once')

new_segorder, new_segnames = [], []
for layi in range(pflens[0]):
    for pfi, pfnames in enumerate(segnames):
        segname = pfnames[layi]
        new_segnames.append(f'P{pfi+1}{layi+1}')
        new_segorder.append(u.select_atoms(f'segid {segname}'))

newu = mda.Merge(*new_segorder)
newu.segments.segids = new_segnames

if CIN is not None:
    new_charges = []
    for segment in new_segorder:
        new_charges += segment.atoms.charges.tolist()
    newu.add_TopologyAttr('formalcharges')
    newu.atoms.formalcharges = new_charges
    newu.atoms.charges = new_charges


if newu.atoms.n_atoms != u.atoms.n_atoms:
    sys.exit('Some atoms are missing based on your selection. Try again.')

newu.atoms.write(OUT)
print(f"Corrected PDB written to: {OUT}.pdb")
if CIN is not None:
    with open(f"{OUT}.psf", "w+") as file:
        file.write("PSF EXT CMAP XPLOR\n\n")
        file.write("         3 !NTITLE\n")
        file.write("* GENERATED FOR USE BY FIBMAP\n")
        file.write("* GENERATED BY correct_pdb.py\n")
        file.write("* THIS FILE ONLY CONTAINS ATOM AND BOND INFORMATION\n")
        file.write(f"\n{str(newu.atoms.n_atoms):>10} !NATOM\n")
        for a in newu.atoms:
            charge = f"{np.round(a.charge,6):.6f}"
            mass = f"{np.round(a.mass,6):.6f}"
            file.write(f"{(a.index+1):>10} {a.segment.segid:<9}{a.residue.resid:<9}{a.residue.resname:<9}{a.name:<9}{a.type:<9}{charge:<15}{mass:<18}0\n")
        file.write(f"\n{len(newu.atoms.bonds):>10} !NBOND: bonds\n")
        counter = 0
        for b in newu.atoms.bonds:
            for a in b.atoms:
                file.write(f"{(a.index+1):>10}")
            counter += 1
            if counter == 4:
                file.write("\n")
                counter = 0
        file.write("\n")
    print(f"Corrected PSF written to: {OUT}.psf")

