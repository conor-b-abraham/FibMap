import numpy as np
import MDAnalysis as mda
import os
import sys
import argparse
import multiprocessing
import textwrap
import matplotlib as mpl

from src import io
from src import calc
from src import utils
from src import mapping


def main():
    github_link = "LINK"

    # Initialize logger
    LOG = io.Logger()

    # ------------------------------------------------------------------------------
    # INPUT
    # ------------------------------------------------------------------------------
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=LOG.get_title()+LOG.output_formatter(f"\n\nThis program was written to compute and visualize the hydrogen bonds, salt bridges, and pistacking interactions within an amyloid fibril from either a PDB file or a molecular dynamics trajectory. To use it, first compute the interactions using FibMap.py calc and then create the visualization with FibMap.py map. For more thorough usage and tutorials please check the github repo: {github_link}")
        )
    parser._optionals.title = 'HELP'
    subparsers = parser.add_subparsers(
                    title="SUBCOMMANDS",
                    description=LOG.output_formatter("The functionality of this program is broken up into two subcommands, calc and map. Use calc to compute interactions within the fibril structure. Then, use map to create the visualization."),
                    help="Valid subcommands are calc and map.", 
                    dest="command", 
                    required=True
                    )

    calc_parser = subparsers.add_parser("calc",
                    formatter_class=argparse.RawDescriptionHelpFormatter, 
                    help=LOG.output_formatter("Calculates interactions within an amyloid fibril. For additional help use: FibMap.py calc -h or FibMap.py calc --help"),
                    description=LOG.get_title()+LOG.output_formatter(f"\n\nUse this subcommand first to calculate the hydrogen bonds, salt bridges, and pi stacking interactions within the fibril. For more thorough usage and tutorials please check the github repo: {github_link}"),
                    epilog=LOG.output_formatter(textwrap.dedent("""
                        ADDITIONAL HELP
                        I. INPUT FILE:
                        The parameters can be provided in an input file (specified with -i/--input_file) for the user's convenience. Only one parameter is allowed per line, separate the parameters name and the chosen value(s) with an equal-to sign (=). For flags (i.e. verbose, [no]saveraw, [no]log, [no]backup), set the value to either 'True' or 'False'. Lines may be commented out with a pound/hash/number sign (#). If a parameter can take multiple values (e.g. trajectory_file), you can either use separate entries for each value (e.g. put trajectory_file = filename1, trajectory_file = filename2, etc. on separate lines) or you can separate the filenames with a space in a single entry (e.g. trajectory_file = filename1 filename2 ...). An input file template is available in README.md.
                        
                        II. IMPORTANT NOTE ABOUT TOPOLOGY FILES (the order of the fibril's segids matters):
                        This program does not determine the relative location of each segment automatically, and therefore, the order of the segments must be predictable (proper ordering described below). If you did not originally set up your system in the described order you can try to reorder it using MDAnalysis.

                        The proper order should be layer by layer and protofilament by protofilament. For example, if you have a fibril with 3 layers and 2 protofilaments, the segments should be in the following order:

                        1) Layer 1, Protofilament 1    PROTOFILAMENT 1    PROTOFILAMENT 2
                        2) Layer 1, Protofilament 2    ███████████████    ███████████████
                        3) Layer 2, Protofilament 1    ██████ 5 ██████    ██████ 6 ██████   LAYER 3
                        4) Layer 2, Protofilament 2    ███████████████    ███████████████
                        5) Layer 3, Protofilament 1
                        6) Layer 3, Protofilament 2    ███████████████    ███████████████
                                                       ██████ 3 ██████    ██████ 4 ██████   LAYER 2
                                                       ███████████████    ███████████████

                                                       ███████████████    ███████████████
                                                       ██████ 1 ██████    ██████ 2 ██████   LAYER 1
                                                       ███████████████    ███████████████

                        III. A TIP TO SPEED UP CALCULATION:
                        The calculation of hydrogen bonds, salt bridges, and pi stacking interactions can take a long time. To speed this up, you can compute each interaction type separately. Also, note that the calculation of hydrogen bonds and salt bridges can be parallelized, but the calculation of pi stacking interactions can only be completed on a single processor (because the pi stacking calculation is relatively fast). These separate calculations can be run using --calctype and --nprocs. 
                    """), smarttabs=True)
                )
    calc_parser._optionals.title = 'HELP'
    calc_inputs = calc_parser.add_argument_group("INPUT FILES")
    calc_inputs.add_argument("-i", "--input_file",
                            type=io.ap_valid_file,
                            default=None, 
                            help='(OPTIONAL, Default=None, Type: Filename) Input file containing parameters for calculation job. All commandline arguments can be specified in this file for user convenience. Any parameters given at the commandline will override their counterpart in this file. See I. INPUT FILE below for additional help.'
                            ) 
    calc_inputs.add_argument("-c", "--checkpoint_file",
                            type=io.ap_valid_file,
                            default=None, 
                            nargs="+",
                            help='(OPTIONAL, Default: None, Type: Filename) Checkpoint file(s) to resume from last saved checkpoint(s). These checkpoint file(s) includes the original parameters for the run. Any commandline or inputfile arguments that will effect the results of the run (e.g. trajectory_file, topology_file, n_protofilaments, omit_layers, etc.) will be ignored if a checkpoint file is provided).'
                            ) 
    calc_inputs.add_argument("-f", "--trajectory_file", 
                            type=io.ap_valid_file,
                            default=None,
                            nargs='+',
                            help='(OPTIONAL, Type: Filename) Trajectory file(s) containing coordinate information (e.g. XTC, TRR, PDB). If multiple are provided, the systems must match exactly. If none is provided, the coordinates will be collected from the topology file. See https://userguide.mdanalysis.org/stable/formats/index.html for valid file formats.'
                            )
    calc_inputs.add_argument("-t", "--topology_file", 
                            type=io.ap_valid_file,
                            default=None, # Actually required but could be in input or checkpoint file
                            help='(REQUIRED, Type: Filename) Topology file containing atom charges, bonds, and Segment IDs (e.g TPR). If no trajectory file is provided, the topology file must also contain coordinate information. The order of the segments is very important. Please see II. IMPORTANT NOTE ABOUT TOPOLOGY FILES below for more information. See https://userguide.mdanalysis.org/stable/formats/index.html for valid file formats.'
                            )
    calc_output = calc_parser.add_argument_group("OUTPUT")
    calc_output.add_argument("-o", "--output_directory",
                            type=io.ap_valid_path, # Make sure path exists
                            default=None,
                            help=f'(OPTIONAL, Default: {os.getcwd()}, Type: DirectoryPath) Directory to write files to. This directory must already exist. Default is working directory.'
                            )
    calc_raw_group = calc_output.add_mutually_exclusive_group()
    calc_raw_group.add_argument("--saveraw", 
                                action="store_true",
                                help='(OPTIONAL, Default=saveraw) If used, the unprocessed results will be saved.'
                                )
    calc_raw_group.add_argument("--nosaveraw", 
                                action="store_true",
                                help='(OPTIONAL, Default=saveraw) If used, the unprocessed results will not be saved.'
                                )
    calc_output.add_argument("-v", "--verbose", 
                            action="store_true",
                            help="(OPTIONAL, Default=False) If used, intermolecular forces will be passed to standard output."
                            ) # Commandline
    calc_log_group = calc_output.add_mutually_exclusive_group()
    calc_log_group.add_argument("--log", 
                                action="store_true",
                                help="(OPTIONAL, Default=nolog) If used, write standard output to logfile. This option is better than manually passing stdout to a file at the commandline, as it will not write progress bars to the file."
                                )
    calc_log_group.add_argument("--nolog", 
                                action="store_true",
                                help="(OPTIONAL, Default=nolog) If used, don't write standard output to logfile."
                                )
    calc_backup_group = calc_output.add_mutually_exclusive_group()
    calc_backup_group.add_argument("--backup", 
                                   action="store_true",
                                   help="(OPTIONAL, Default=backup) If used, past logfiles will be backed up."
                                   ) 
    calc_backup_group.add_argument("--nobackup", 
                                action="store_true",
                                help="(OPTIONAL, Default=backup) If used, past logfiles will not be backed up."
                                ) 
    calc_options = calc_parser.add_argument_group("OPTIONS")
    calc_options.add_argument("--calctype",  
                            choices=["ALL", "HB", "SB", "PI", "HB+SB", "HB+PI", "SB+PI"],
                            default="ALL",
                            help="(OPTIONAL, Default: %(default)s) What type of interaction to compute. Options are ALL, HB, SB, PI, HB+SB, HB+PI, and SB+PI. ALL computes all, options with HB computes hydrogen bonds, options with SB computes salt bridges, and options with PI computes pi stacking interactions. "
                            ) # Commandline
    calc_options.add_argument("-n", "--n_protofilaments", 
                            type=io.ap_positive_int, # must be positive integer
                            default=None, # Actually required but could be in input or checkpoint file
                            help="(REQUIRED, Type: Int > 0) The number of protofilaments in the fibril (i.e. how many segments are in each layer of the fibril)."
                            ) # Commandline or Input File. If Input File, No Commandline.
    calc_options.add_argument("--omit_layers", 
                            type=io.ap_nonnegative_int, # must be positive integer or 0
                            default=None, 
                            help="(OPTIONAL, Default: 0, Type: Int >= 0) How many layers on each end of the fibril to omit from analysis. This is especially important for analysis of simulation trajectories of a finite fibril model as delamination at the ends of the fibril will bias the results."
                            ) # Commandline or Input File. If Input File, No Commandline.
    calc_options.add_argument("--hbond_distance_cutoff",
                              type=io.ap_positive_float,
                              default=3.5,
                              help="(OPTIONAL, Default: %(default)s, Type: Float > 0) The cutoff distance (in Angstroms) for hydrogen bonds. The distance between a potential donor and potential acceptor must be less than this value to be counted as a hydrogen bond.")
    calc_options.add_argument("--hbond_angle_cutoff",
                              type=io.ap_positive_float,
                              default=150,
                              help="(OPTIONAL, Default: %(default)s, Type: Float > 0) The cutoff angle (in degrees) for hydrogen bonds. The angle from a potential donor to a potential hydrogen to a potential acceptor must be greater than this value to be counted as a hydrogen bond.")
    calc_options.add_argument("--saltbridge_selection_mode",
                              choices=["auto", "manual"],
                              default="auto",
                              help="(OPTIONAL, Default: %(default)s) The salt bridge participant selection mode. If auto, --saltbridge_anion_charge_cutoff and --saltbridge_cation_charge_cutoff will be used to identify potential salt bridge participants. If manual, --saltbridge_anion_sel and --saltbridge_cation_sel will be used to find the participants.")
    calc_options.add_argument("--saltbridge_anion_charge_cutoff",
                              type=float, 
                              default=-0.5, 
                              help="(OPTIONAL, Default: %(default)s) Used if --saltbridge_selection_mode auto. Charge cutoff (in e) for salt bridge participant selection. If an atom group belonging to a anionic residue has a charge less than this value, it will be considered a salt bridge participant.")
    calc_options.add_argument("--saltbridge_cation_charge_cutoff",
                              type=float, 
                              default=-0.5, 
                              help="(OPTIONAL, Default: %(default)s) Used if --saltbridge_selection_mode auto. Charge cutoff (in e) for salt bridge participant selection. If an atom group belonging to a cationic residue has a charge greater than this value, it will be considered a salt bridge participant.")
    calc_options.add_argument("--saltbridge_anion_sel",
                              type=str, 
                              default="((resname ASP and name OD1 OD2) or (resname GLU and name OE1 OE2))", 
                              help="(OPTIONAL, Default: '%(default)s') Used if --saltbridge_selection_mode manual. Selection command for anionic salt bridge participants. For help formatting this string, see the MDAnalysis Documentation: https://docs.mdanalysis.org/stable/documentation_pages/selections.html.")
    calc_options.add_argument("--saltbridge_cation_sel",
                              type=str, 
                              default="((resname LYS and name NZ) or (resname ARG and name NH1 NH2 NE) or (resname HSP and name ND1 NE2))", 
                              help="(OPTIONAL, Default: '%(default)s') Used if --saltbridge_selection_mode manual. Selection command for cationic salt bridge participants. For help formatting this string, see the MDAnalysis Documentation: https://docs.mdanalysis.org/stable/documentation_pages/selections.html.")
    calc_options.add_argument("--saltbridge_distance_cutoff", 
                              type=io.ap_positive_float,
                              default=4.0,
                              help="(OPTIONAL, Default: %(default)s, Type: Float > 0) The cutoff distance (in Angstroms) for salt bridges. The minimum distance between anionic and cationic groups must be less than or equal to this value to be counted as a salt bridge.")
    calc_options.add_argument("--pistacking_phe_sel",
                              type=str,
                              default="(resname PHE and name CG CD2 CE2 CZ CE1 CD1)",
                              help="(OPTIONAL, Default: %(default)s) The MDAnalysis selection command for phenylalanine rings. For help formatting this string, see the MDAnalysis Documentation: https://docs.mdanalysis.org/stable/documentation_pages/selections.html.")
    calc_options.add_argument("--pistacking_tyr_sel",
                              type=str,
                              default="(resname TYR and name CG CD2 CE2 CZ CE1 CD1)",
                              help="(OPTIONAL, Default: %(default)s) The MDAnalysis selection command for tyrosine rings. For help formatting this string, see the MDAnalysis Documentation: https://docs.mdanalysis.org/stable/documentation_pages/selections.html.")
    calc_options.add_argument("--pistacking_his_sel",
                              type=str,
                              default="(resname HSD HSE HSP and name CG CD2 NE2 CE1 ND1)",
                              help="(OPTIONAL, Default: %(default)s) The MDAnalysis selection command for histidine rings. For help formatting this string, see the MDAnalysis Documentation: https://docs.mdanalysis.org/stable/documentation_pages/selections.html.")
    calc_options.add_argument("--pistacking_trp_sel",
                              type=str,
                              default="(resname TRP and name CG CD1 NE1 CE2 CD2)",
                              help="(OPTIONAL, Default: %(default)s) The MDAnalysis selection command for tryptophan rings (should be for the 5-membered ring). For help formatting this string, see the MDAnalysis Documentation: https://docs.mdanalysis.org/stable/documentation_pages/selections.html.")
    calc_options.add_argument("--nprocs", 
                            type=io.ap_cpu_int, # must be -1, -2, or positive integer
                            default=1,
                            help="(OPTIONAL, Default: %(default)s, Type: Int > 0 or Int =-1 or Int=-2) How many processors to use for hydrogen bond and pi stacking calculations. Use -1 to use all available processors, -2 to use half of the available processors, or some positive integer."
                            ) # Commandline

    map_parser = subparsers.add_parser("map", 
                    formatter_class=argparse.RawDescriptionHelpFormatter,
                    help=LOG.output_formatter("Uses results from FibMap.py calc to create a visualization. For additional help use: FibMap.py map -h or FibMap.py map --help"),
                    description=LOG.get_title()+LOG.output_formatter(f"\n\nUse this subcommand second to visualize the interactions computed by the calc subcommand. For more thorough usage and tutorials please check the github repo: {github_link}"),
                    epilog=LOG.output_formatter(textwrap.dedent(f"""
                        ADDITIONAL HELP
                        I. INPUT FILE:
                        The above parameters (and the additional parameters below) can be provided in an input file (specified with -i/--input_file) for the user's convenience. Only one parameter is allowed per line, separate the parameters name and the chosen value(s) with an equal-to sign (=). For flags (i.e. verbose, [no]log, [no]backup), set the value to either 'True' or 'False'. Lines may be commented out with a pound/hash/number sign (#). If a parameter can take multiple values (e.g. trajectory_file), you can either use separate entries for each value (e.g. put checkpoint_file = filename1, checkpoint_file = filename2, etc. on separate lines) or you can separate the filenames with a space in a single entry (e.g. checkpoint_file = filename1 filename2 ...). An input file template is available in README.md. The following additional parameters can also be defined in the input file:
                        
                        HYDROPHOBIC ZIPPERS AND WATER CHANNELS:
                        water_region = WATER_REGION     
                                                        (Type: See Below, Default: None) Define the location of a water channel. See II. SPECIFYING SHADED REGIONS below for details. Multiple can be defined with separate entries.
                        zipper_region = ZIPPER_REGION   
                                                        (Type: See Below, Default: None) Define the location of a hydrophobic zipper. See II. SPECIFYING SHADED REGIONS below for details. Multiple can be defined with separate entries.
                        
                        FIGURE SETTINGS:
                        figure_width = FIGURE_WIDTH
                                                        (Type: Float, >0, Default: 6.5) Figure width in inches.
                        figure_height = FIGURE_HEIGHT
                                                        (Type: Float, >0, Default: 4.5) Figure height in inches.
                        legend = [True/False]
                                                        (Type: Bool, Default: True) Toggle whether or not to include a legend.
                        figure_dpi = FIGURE_DPI
                                                        (Type: Int, Default: 300) The figure resolution in dots per inch (dpi).
                        transparent_background = [True/False]
                                                        (Type: Bool, Default: False) Make the figure transparent.
                        numbered_residues = [True/False]
                                                        (Type: Bool, Default: False) By default, the residues will be labelled with their 1-letter residue name abbreviation. If you use this option, they will instead be labelled with their one-based residue index. This is useful for determining shaded regions.
                        
                        COLORS:
                        * All colors must be valid matplotlib colors. See matplotlib documentation for options: https://matplotlib.org/stable/tutorials/colors/colors.html
                        
                        acidic_color = ACIDIC_COLOR
                                                        (Type: Color, Default: steelblue) Color of acidic residues.
                        acidic_label_color = ACIDIC_LABEL_COLOR
                                                        (Type: Color or "chain", Default: white) Color of acidic residue labels. If "chain" residue labels will be given the same color as their chain backbone color.
                        basic_color = BASIC_COLOR
                                                        (Type: Color, Default: firebrick) Color of acidic residues.
                        basic_label_color = BASIC_LABEL_COLOR
                                                        (Type: Color or "chain", Default: white) Color of basic residue labels. If "chain" residue labels will be given the same color as their chain backbone color.
                        polar_color = POLAR_COLOR
                                                        (Type: Color, Default: seagreen) Color of polar residues.
                        polar_label_color = POLAR_LABEL_COLOR
                                                        (Type: Color or "chain", Default: white) Color of polar residue labels. If "chain" residue labels will be given the same color as their chain backbone color.
                        nonpolar_color = NONPOLAR_COLOR
                                                        (Type: Color, Default: white) Color of nonpolar residues.
                        nonpolar_label_color = NONPOLAR_LABEL_COLOR
                                                        (Type: Color or "chain", Default: white) Color of nonpolar residue labels. If "chain" residue labels will be given the same color as their chain backbone color.
                        backbone_colors = BACKBONE_COLOR ...
                                                        (Type: List of Colors, Default: black dimgray) Color(s) of chain backbone(s). If more than one is given, the specified colors will be cycled through. If a color is given as a tuple or a list it should not have any spaces in it (e.g. (0,0,0) not  (0, 0, 0)). Separate each color with a whitespace (i.e. spaces or tabs).
                        hbond_color_1 = HBOND_COLOR_1
                                                        (Type: Color, Default: black) Color of hydrogen bonds.
                        hbond_color_2 = HBOND_COLOR_2
                                                        (Type: Color, Default: white) Color of dashed line for hydrogen bonds that are both intra- and inter-layer.
                        saltbridge_color_1 = SALTBRIDGE_COLOR_1
                                                        (Type: Color, Default: gold) Fill color for salt bridges.
                        saltbridge_color_2 = SALTBRIDGE_COLOR_2
                                                        (Type: Color, Default: orange) Outline color for salt bridges.
                        saltbridge_color_3 = SALTBRIDGE_COLOR_3
                                                        (Type: Color, Default: white) Color of dashed line for salt bridges that are both intra- and inter-layer.
                        pistacking_color_1 = PISTACKING_COLOR_1
                                                        (Type: Color, Default: gray) Line color for pi stacking interaction lines and edge color for pi stacking interaction markers.
                        pistacking_color_2 = PISTACKING_COLOR_2
                                                        (Type: Color, Default: white) Fill color for pi stacking interaction markers and dashed line color for pi stacking interactions that are both intra-and inter-layer.
                        water_color = WATER_COLOR
                                                        (Type: Color, Default: powderblue) Color of water regions.
                        water_opacity = WATER_OPACITY
                                                        (Type: Float, >0, <=1, Default: 0.5) Opacity of water regions.
                        zipper_color = ZIPPER_COLOR
                                                        (Type: Color, Default: tan) Color of hydrophobic zipper regions.
                        zipper_opacity = ZIPPER_OPACITY
                                                        (Type: Float, >0, <=1, Default: 0.5) Opacity of hydrophobic zipper regions.

                        II. SPECIFYING SHADED REGIONS:
                        On the map, hydrophobic zippers and water channels can be highlighted. These regions can be specified in the input file with the zipper_region and water_region options.  Individual shaded regions are identified by following a path along the residues that  define their borders.

                        To define the path:
                        Enter a series of sides separated by commas (,). Each side must contain a protofilament number (one-based) and a pair of residue numbers (using the resids in your topology). The protofilament number should be separated from the residue numbers by a colon (:) and the residue numbers should be separated from one another by a hyphen (-):
                            [Protofilament Number]:[Residue 1 Number]-[Residue 2 Number] 
                        
                        From the path, a polygon will be defined to go from residue to residue within each side and from the last residue of a side to the first residue of the next side. As such, the first residue in each side should be the one closest to the last residue in the previous side.
                        IMPORTANT: There cannot be any white space (e.g. spaces, tabs, etc.) in the path specification string.
                        TIP 1: Only one side is necessary so you can specify a region encapsulated by a U-bend.
                        TIP 2: If you want a region to contact a part of the side only at one residue you can list that residue twice in a side (e.g. 1:4,4)

                        Examples:
                        A: A hydrophobic zipper between residues 1 to 6 on protofilament 1 and residues
                        1 to 6 on protofilament 2.

                         █████  █████  █████  █████  █████  █████  █████  █████  █████
                         █ 1 ████ 2 ████ 3 ████ 4 ████ 5 ████ 6 ████ 7 ████ 8 ████ 9 █ (Protofilament 1)
                         █████▒▒█████▒▒█████▒▒█████▒▒█████▒▒█████  █████  █████  █████
                           ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
                           ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
                           ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
                         █████▒▒█████▒▒█████▒▒█████▒▒█████▒▒█████  █████  █████  █████
                         █ 1 ████ 2 ████ 3 ████ 4 ████ 5 ████ 6 ████ 7 ████ 8 ████ 9 █ (Protofilament 2)
                         █████  █████  █████  █████  █████  █████  █████  █████  █████
                        
                        To add this region to the visualization you can define it within your map input file: zipper_region = 1:1-6,2:6-1

                        B: A water channel encompassed by residues 2-9 of protofilament 1.

                         █████  █████  █████  █████  █████
                         █ 5 ████ 4 ████ 3 ████ 2 ████ 1 █ (Protofilament 1)
                         █████▒▒█████▒▒█████▒▒█████  █████
                          ██▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
                          ██▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
                          ██▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
                         █████▒▒█████▒▒█████▒▒█████
                         █ 6 ████ 7 ████ 8 ████ 9 █
                         █████  █████  █████  █████
                        
                        To add this region to the visualization you can define it within your map input file: water_region = 1:2-9
                        """), smarttabs=True)
        )
    map_parser._optionals.title = 'HELP'
    map_inputs = map_parser.add_argument_group("INPUT FILES")
    map_inputs.add_argument("-c", "--checkpoint_file",
                            type=io.ap_valid_file,
                            nargs="+",
                            default=None, 
                            help='(REQUIRED, Type: Filename) Checkpoint file(s) to finished calc job or previous map job.'
                            )
    map_inputs.add_argument("-i", "--input_file",
                            type=io.ap_valid_file, 
                            default=None, 
                            help='(OPTIONAL, Default=None, Type: Filename) Input file containing parameters for mapping job. All required commandline arguments and additional formatting parameters can alternatively be specified in this file. Arguments given at the commandline will override any of their counterparts given in this file. See I. INPUT FILE below for a list of parameters that can be set in this file.'
                            )
    map_output = map_parser.add_argument_group("OUTPUT")
    map_output.add_argument("-o", "--figure_file",
                            default=None, 
                            help="(OPTIONAL, Default: fibmap.png, Type: Filename) Name of output image file"
                            )
    map_log_group = map_output.add_mutually_exclusive_group()
    map_log_group.add_argument("--log", 
                               action="store_true",
                               help="(OPTIONAL, Default=nolog) If used, write standard output to logfile for future inspection. This option is better than manually passing stdout to a file at the commandline, as it will not write progress bars to the file."
                               )
    map_log_group.add_argument("--nolog", 
                               action="store_true",
                               help="(OPTIONAL, Default=log) If used, don't write standard output to logfile for future inspection."
                               )
    map_backup_group = map_output.add_mutually_exclusive_group()
    map_backup_group.add_argument("--backup", 
                                  action="store_true",
                                  help="(OPTIONAL, Default=backup) If used, past logfiles and past figure file images will be backed up."
                                  )
    map_backup_group.add_argument("--nobackup", 
                                  action="store_true",
                                  help="(OPTIONAL, Default=backup) If used, neither past logfiles nor past figure file images will be backed up."
                                  )
    map_output.add_argument("--showfig", 
                            action="store_true",
                            help="(OPTIONAL) If used, the figure image will be opened after it is saved."
                            ) # Commandline
    map_options = map_parser.add_argument_group("OPTIONS")
    map_options.add_argument("--p_cutoff",
                            type=io.ap_nonnegative_frac,
                            default=None,
                            help=f"(OPTIONAL, Default: 0.5, Type: Float >= 0 and <= 1) Probability cutoff for hydrogen bonds, salt bridges, and pi stacking interactions. If the probability of a given interaction is less than this value then it will not be displayed on the map. (TIP: To hide all interactions, set this value to 1 and do not set --hbond_n_cutoff, --hbond_p_cutoff, --saltbridge_p_cutoff, or --pistacking_p_cutoff)."
                            )
    map_hb_cutoffs = map_options.add_mutually_exclusive_group()
    map_hb_cutoffs.add_argument("--hbond_n_cutoff",
                            type=io.ap_nonnegative_float,
                            default=None,
                            help="(OPTIONAL, Default: None, Type: Float >= 0) If set, the average number of hydrogen bonds per frame formed between two groups will be used to determine whether or not a hydrogen bond is shown on the figure instead of a probability cutoff. If the average number of hydrogen bonds per frame is less than this cutoff then the hydrogen bond will not be displayed on the map.")
    map_hb_cutoffs.add_argument("--hbond_p_cutoff",
                            type=io.ap_nonnegative_frac,
                            default=None,
                            help=f'(OPTIONAL, Default: None, Type: Float >= 0 and <= 1) Individually set the probability cutoff for hydrogen bonds. If not set, p_cutoff will be used. (TIP: To hide all hydrogen bonds, set this value to 1 and do not set --hbond_n_cutoff).'
                            )
    map_options.add_argument("--saltbridge_p_cutoff",
                            type=io.ap_nonnegative_frac,
                            default=None,
                            help=f'(OPTIONAL, Default: None, Type: Float >= 0 and <= 1) Individually set the probability cutoff for salt bridges. If not set, p_cutoff will be used. (TIP: To hide all salt bridges, set this value to 1).'
                            )
    map_options.add_argument("--pistacking_p_cutoff",
                            type=io.ap_nonnegative_frac,
                            default=None,
                            help=f'(OPTIONAL, Default: None, Type: Float >= 0 and <= 1) Individually set the probability cutoff for pi stacking interactions. If not set, p_cutoff will be used. (TIP: To hide all pi stacking interactions, set this value to 1).'
                            )
    map_options.add_argument("--nprocs", 
                            type=io.ap_cpu_int, # must be -1, -2, or positive integer
                            default=1,
                            help="(OPTIONAL, Default: 1) How many processors to use for residue positions calculation. Use -1 to use all available processors, -2 to use half of the available processors, or some positive integer."
                            ) 

    args = parser.parse_args()

    params = io.Params(args, sys.argv)
    loglines = params.get_loglines()

    # Add title to log file
    LOG.set_logfile(params.output_log)
    LOG.make_title()

    if loglines != "":
        LOG.header("FILE BACKUPS:")
        LOG.output(loglines)
        LOG.clear_line(n=1)

    LOG.header("PARAMETERS")
    LOG.smart_print(f"{params}")

    # ------------------------------------------------------------------------------
    # FIBRIL  SETUP
    # ------------------------------------------------------------------------------
    # CREATE UNIVERSE & FIND SEGIDS THAT WILL BE INCLUDED IN ANALYSIS
    u = mda.Universe(params.topology_file, params.trajectory_file)
    protein_ag = u.select_atoms("protein")

    # Get System Information
    SYSTEMINFO = utils.SystemInfo(protein_ag, params.n_protofilaments, params.omit_layers)

    LOG.output("\nSEGMENTS TO BE ANALYZED:")
    for pf in range(params.n_protofilaments):
        LOG.output(f"Protofilament {pf+1}: {' '.join(SYSTEMINFO.structure[:, pf].tolist())}")

    # ------------------------------------------------------------------------------
    # MAIN : CALCULATE IMFS
    # ------------------------------------------------------------------------------
    if params.command == "calc": # Calculate Method Used
        # ------------------------------- HYDROGEN BONDS -------------------------------
        if params.calctype == "ALL" or "HB" in params.calctype:
            LOG.header("HYDROGEN BONDS")
            LOG.output("Calculating Hydrogen Bonds")

            # First, calculate all hydrogen bonds within the fibril
            hbonds = calc.HydrogenBondCalculator(params.topology_file, params.trajectory_file, params.nprocs, SYSTEMINFO, params.hb_unprocessed_file, params.hbond_angle_cutoff, params.hbond_distance_cutoff)
            LOG.clear_line() # Clear finding all hydrogen bonds line
            if not params.nosaveraw and params.hb_unprocessed_file is None:
                params.set_filename(hb_unprocessed_file=f"{params.output_directory}/unprocessed_hydrogen_bonds.npy")
                hbonds.save(unprocessed_file=params.hb_unprocessed_file)
                LOG.bullet(f"Unprocessed Hydrogen Bonds saved to {utils.relative_path(params.hb_unprocessed_file)}")
            else:
                LOG.bullet(f"Hydrogen bonds were already calculated and stored in {utils.relative_path(params.hb_unprocessed_file)}")

            # Second, sort hydrogen bonds into intramolecular, interprotofilament, interlayer groups
            if params.hb_processed_file is None:
                LOG.output(f"Processing hydrogen bonds")
                hbonds.process()
                LOG.clear_line() # Clear "Processing hydrogen bonds" line
                params.set_filename(hb_processed_file=f"{params.output_directory}/processed_hydrogen_bonds.npy")
                hbonds.save(processed_file=params.hb_processed_file)
                LOG.bullet(f"Processed hydrogen bonds saved to {utils.relative_path(params.hb_processed_file)}")

                if params.verbose:
                    hbonds.show(LOG)
            else:
                LOG.bullet(f"~ Hydrogen bonds were already processed and stored in {utils.relative_path(params.hb_processed_file)}")

        # -------------------------------- SALT BRIDGES --------------------------------
        if params.calctype == "ALL" or "SB" in params.calctype:
            LOG.header("SALT BRIDGES")
            
            if params.saltbridge_selection_mode == "AUTO":
                LOG.output("Automatically Determined Salt Bridge Groups:")
                ag = mda.Universe(params.topology_file, params.trajectory_file).select_atoms(f"segid {' '.join(SYSTEMINFO.structure.flatten())}")
                anion_sel, anion_auto_str = calc.auto_sb_finder(ag, params.saltbridge_anion_charge_cutoff, "anion", SYSTEMINFO)
                cation_sel, cation_auto_str = calc.auto_sb_finder(ag, params.saltbridge_cation_charge_cutoff, "cation", SYSTEMINFO)
                LOG.output(anion_auto_str)
                LOG.output(f"{cation_auto_str}\n")
            elif params.saltbridge_selection_mode == "MANUAL":
                LOG.output("Using Manually Defined Salt Bridge Groups")
                anion_sel = params.saltbridge_anion_sel
                cation_sel = params.saltbridge_cation_sel

            LOG.output("Calculating Salt Bridges")
            saltbridges = calc.SaltBridgeCalculator(params.topology_file, params.trajectory_file, SYSTEMINFO, anion_sel, cation_sel, params.sb_unprocessed_file, params.saltbridge_distance_cutoff)
            LOG.clear_line() # Clear finding all salt bridges

            if not params.nosaveraw and params.sb_unprocessed_file is None:
                params.set_filename(sb_unprocessed_file=f"{params.output_directory}/unprocessed_salt_bridges.npy")
                saltbridges.save(unprocessed_file=params.sb_unprocessed_file)
                LOG.bullet(f"Unprocessed salt bridges saved to {utils.relative_path(params.sb_unprocessed_file)}")
            else:
                LOG.bullet(f"Salt bridges were already calculated and stored in {utils.relative_path(params.sb_unprocessed_file)}")

            if params.sb_processed_file is None:
                LOG.output(f"Parsing salt bridges")
                saltbridges.process()
                LOG.clear_line() # Clear parsing salt bridges
                params.set_filename(sb_processed_file=f"{params.output_directory}/processed_salt_bridges.npy")
                saltbridges.save(processed_file=params.sb_processed_file)
                LOG.bullet(f"Processed salt bridges saved to {utils.relative_path(params.sb_processed_file)}")

                if params.verbose:
                    saltbridges.show(LOG)
            else:
                LOG.bullet(f"Salt bridges were already processed and stored in: {utils.relative_path(params.sb_processed_file)}")

        # ------------------------------- PI-PI STACKING -------------------------------
        if params.calctype == "ALL" or "PI" in params.calctype:
            LOG.header("PI STACKING INTERACTIONS")
            LOG.output("Calculating Pi Stacking Interactions")
            pistacking = calc.PiStackingCalculator(params.topology_file, params.trajectory_file, params.nprocs, SYSTEMINFO, params.pi_unprocessed_file, phe_sel=params.pistacking_phe_sel, tyr_sel=params.pistacking_tyr_sel, his_sel=params.pistacking_his_sel, trp_sel=params.pistacking_trp_sel)
            LOG.clear_line() # Clear finding all pipi stacking interactions line

            if not params.nosaveraw and params.pi_unprocessed_file is None:
                params.set_filename(pi_unprocessed_file=f"{params.output_directory}/unprocessed_pistacking_interactions.npy")
                pistacking.save(unprocessed_file=params.pi_unprocessed_file)
                LOG.bullet(f"Unprocessed pi interactions saved to {utils.relative_path(params.pi_unprocessed_file)}")
            else:
                LOG.bullet(f"Pi stacking interactions were already calculated and stored in {utils.relative_path(params.pi_unprocessed_file)}")

            if params.pi_processed_file is None:
                print(f"Parsing pi-pi stacking interactions")
                pistacking.process()
                LOG.clear_line() # Clear parsing pipi stacking interactions line
                params.set_filename(pi_processed_file=f"{params.output_directory}/processed_pistacking_interactions.npy")
                pistacking.save(processed_file=params.pi_processed_file)
                LOG.bullet(f"Processed pi interactions saved to {utils.relative_path(params.pi_processed_file)}")

                if params.verbose:
                    pistacking.show(LOG)
            else:
                LOG.bullet(f"Pi stacking interactions were already processed and stored in: {utils.relative_path(params.pi_processed_file)}")


    # ------------------------------------------------------------------------------
    # MAIN : CREATE MAP
    # ------------------------------------------------------------------------------
    if params.command == 'map':
        LOG.header("CREATING FIBRIL MAP")

        fmap = mapping.FibrilMap(params, SYSTEMINFO)

        # Create Chains
        CA_positions, SC_positions, Cterm_positions, Nterm_positions, fontsize = fmap.make_chains()
        LOG.bullet(f"Residue label fontsize set to {fontsize}pt Arial")

        if params.map_positions_file is None:
            params.set_filename(map_positions_file=f"{params.output_directory}/map_positions.npz")
            np.savez(params.map_positions_file, CA=CA_positions, SC=SC_positions, CT=Cterm_positions, NT=Nterm_positions)
            LOG.bullet(f"Residue positions saved to {utils.relative_path(params.map_positions_file)}")
        else:
            LOG.bullet(f"Residue positions retrieved from: {utils.relative_path(params.map_positions_file)}")
        
        # Shaded Regions
        if params.zipper_region is not None or params.water_region is not None:
            if params.verbose:
                LOG.header("REGIONS")
            region_string = fmap.add_regions()
            if params.verbose:
                LOG.output(region_string)

        # Add Hydrogen Bonds
        if params.hb_processed_file is not None:
            if params.verbose:
                LOG.header("HYDROGEN BONDS")
            hbonds_string = fmap.add_hydrogen_bonds()
            if params.verbose:
                if params.use_hbond_n_cutoff:
                    LOG.output(f"{'<N H-BONDS>':>43}")
                else:
                    LOG.output(f"{'P(H-BOND)':>42}")
                LOG.output(f"{'DONOR':>10}{'ACCEPTOR':>13}{'INTRALAYER':>13}{'INTERLAYER':>12}")
                LOG.output(hbonds_string)

        # Add Salt Bridges
        if params.sb_processed_file is not None:
            if params.verbose:
                LOG.header("SALT BRIDGES")
            sb_string = fmap.add_salt_bridges()
            if params.verbose:
                LOG.output(f"{'P(SALT BRIDGE)':>47}")
                LOG.output(f"{'DONOR':>10}{'ACCEPTOR':>13}{'INTRALAYER':>13}{'INTERLAYER':>12}")
                LOG.output(sb_string)


        # Add Pipi Interactions
        if params.sb_processed_file is not None:
            if params.verbose:
                LOG.header("PI STACKING INTERACTIONS")
            pi_string = fmap.add_pi_stacking()
            if params.verbose:
                LOG.output(f"{'Intralayer Probabilities':>56}{'Interlayer Probabilities':>39}")
                LOG.output(f"{'RESIDUE-A':>12}{'RESIDUE-B':>12}{'Total':>9}{'T*':>6}{'I*':>7}{'S*':>7}{'D*':>7}{'Total':>11}{'T*':>6}{'I*':>7}{'S*':>7}{'D*':>7}")
                LOG.output(pi_string)
                LOG.smart_print(f"    KEY (*)\n    The following probabilities are the probability of each kind of pi-stacking interaction when a given pi stacking interaction forms:\n{'T: T-Shaped':>15}{'I: Intermediate':>19}{'S: Sandwich':>15}{'D: Parallel Displaced':>25}")

        # # Add legend
        if params.legend:
            fmap.make_legend()

        fmap.save()
        LOG.header("COMPLETE")
        LOG.bullet(f"Fibril Map saved to {params.figure_file}")

if __name__ == '__main__':
    main()