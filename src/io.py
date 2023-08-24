import os
from matplotlib.colors import is_color_like
import MDAnalysis as mda
from datetime import datetime
import textwrap
import argparse
import multiprocessing
import numpy as np

from src import utils

# CONTENTS:
#       file_backup: Backup files in output directory so that old files are not overwritten 
#       Logger: Configurable custom logger to return information to console and log file
#       ap_valid_path: For argparse - check to make sure path exists
#       ap_valid_file: For argparse - check to make sure file exists
#       ap_positive_int: For argparse - check to make sure argument is an integer > 0
#       ap_nonnegative_int: For argparse - check to make sure argument is an integer >= 0
#       ap_cpu_int: For argparse - check to make sure argument is an int > 0 OR int = -1 OR int = -2
#       ap_nonnegative_float: For argparse - check to make sure argumnet is a float > 0
#       ap_nonnegative_frac: For argparse - check to make sure argument is a float > 0 AND float <= 1
#       Params: Class for collecting parameters from input files, checkpoint files, and the commandline

# ------------------------------------------------------------------------------
# FILE BACKUP:
# ------------------------------------------------------------------------------
def file_backup(directory, filename):
        '''
        Check for files in directory with the same filename. If they exist, back them up.

        Parameters
        ----------
        directory : Str
            Directory where file will be written to
        filename : Str
            New file's filename
        
        Returns
        -------
        loglines : Str
            lines for output
        '''
        loglines = ""
        outdir_contents = os.listdir(directory)
        if filename in outdir_contents:
            # Find how many backups have already been made
            nextnum = 1
            nextfile = f"{filename}_{nextnum}.bak"
            file_shifter = [(f"{directory}/{filename}", f"{directory}/{nextfile}")]
            while nextfile in outdir_contents:
                lastfile = nextfile
                nextnum += 1
                nextfile = f"{filename}_{nextnum}.bak"
                file_shifter.append((f"{directory}/{lastfile}", f"{directory}/{nextfile}"))

            # Rename Files
            for files in file_shifter[::-1]:
                os.rename(*files)
                loglines += f"{utils.relative_path(files[0])} --> {utils.relative_path(files[1])}\n"
            
        return loglines

# ------------------------------------------------------------------------------
# LOGGER:
# ------------------------------------------------------------------------------
class Logger:
    '''
    Custom logger to return information to console and log file

    Attributes
    ----------
    logfile : str or None [Default = None]
        Name of log file to write contents to.

    Methods
    -------
    set_logfile
        Set the logfile after initialization
    make_title
        Output title to console and logfile.
    output
        Output specified text to console and logfile.
    error
        Output error statement
    clear_line
        Remove last line printed to console.
    '''
    def __init__(self, logfile=None):
        '''
        Parameters
        ----------
        logfile : str or None
            Name of logfile to write output to. If None, no log file is written.
        '''
        self.logfile = logfile
        self.term_width = os.get_terminal_size()[0]
    
    def set_logfile(self, logfile):
        '''
        Parameters
        ----------
        logfile : str
            Name of logfile to write output to
        '''
        self.logfile = logfile

    def _out(self, text, console=True, mode="a"):
        '''
        An internal use func to log a string, however formmatted. 
        '''
        if console:
            print(text)
        if self.logfile is not None:
            with open(self.logfile, mode) as file:
                file.write(f"{text}\n")
    
    def output_formatter(self, text, smarttabs=False):
        '''
        Format provided text in regular output formatting

        Parameters
        ----------
        text : Str
            Text to format
        smarttabs : Bool
            If true, maintain indentation at start of the line. If False, wrap to beginning of line. Default: False
        
        Returns
        -------
        formatted_text : Str
            Formatted Text
        '''
        if "\n" in text:
            text = text.split("\n")
        else:
            text = [text]
        
        formatted_text = []
        for line in text:
            if not line.isspace() and line != "":
                if smarttabs:
                    tabsize = 0
                    for char in line:
                        if char.isspace():
                            tabsize += 1
                        else:
                            break
                    indent = " "*tabsize
                    line = textwrap.fill(line, width=self.term_width, initial_indent="", subsequent_indent=indent)
                else:
                    line = textwrap.fill(line, width=self.term_width)
            formatted_text.append(line)
        
        return "\n".join(formatted_text)

    def output(self, text="\n"):
        '''
        Output text to console and logfile

        Parameters
        ----------
        text : Str [Default = "\n"]
            Text to return to console and logfile. Default is blank line.
        '''
        self._out(self.output_formatter(text))
    
    def get_title(self):
        '''
        Return title text string

        Returns
        -------
        title_text : Str
            text for title
        '''
        title = textwrap.dedent(
                    f"""
                    ╔════════════════════════════════════════════════════╗
                    ║ ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ ║
                    ║ ░░███████╗██╗██████╗░███╗░░░███╗░█████╗░██████╗░░░ ║
                    ║ ░░██╔════╝██║██╔══██╗████╗░████║██╔══██╗██╔══██╗░░ ║
                    ║ ░░█████╗░░██║██████╦╝██╔████╔██║███████║██████╔╝░░ ║
                    ║ ░░██╔══╝░░██║██╔══██╗██║╚██╔╝██║██╔══██║██╔═══╝░░░ ║
                    ║ ░░██║░░░░░██║██████╦╝██║░╚═╝░██║██║░░██║██║░░░░░░░ ║
                    ║ ░░╚═╝░░░░░╚═╝╚═════╝░╚═╝░░░░░╚═╝╚═╝░░╚═╝╚═╝░░░░░░░ ║
                    ║ ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ ║
                    ╚════════════════════════════════════════════════════╝
                    """
                )
        title += self.output_formatter("Calculates and visualizes key intrafibril interactions within an amyloid fibril.\n\nCreated by Conor B. Abraham in the Straub Group at Boston University")
        return title

    def make_title(self, console=True):
        '''
        Output title to console and logfile

        Parameters
        ----------
        console : bool [Default = True]
            If true, output is written to the console. If false, output is only written to the logfile.
        '''
        self._out(self.get_title(), console=console, mode="w+")
    
    def header(self, text):
        '''
        Create a formatted header with the text

        Parameters
        ----------
        text : Str
            Text to include in header
        '''
        if text[-1] == ":":
            text = text[:-1]
        
        text = f'\n{"─"*self.term_width}\n{text.upper()}:'
        self._out(text)
    
    def bullet(self, text):
        '''
        Prints text to a big bullet (~) with proper formatting

        Parameters
        ----------
        text : Str
            Text to place at bullet point
        '''
        text = text.strip()
        if text[0] == "~":
            text = text[1:].strip()
        
        text = textwrap.fill(textwrap.dedent(text), width=self.term_width, initial_indent="~ ", subsequent_indent="  ")
        self._out(text)
    
    def subbullet(self, text):
        '''
        Prints text to a small bullet (-) with proper formatting

        Parameters
        ----------
        text : Str
            Text to place at bullet point
        '''
        text = text.strip()
        if text[0] == "-":
            text = text[1:].strip()
        
        text = textwrap.fill(textwrap.dedent(text), width=self.term_width, initial_indent="    - ", subsequent_indent="      ")
        self._out(text)
    
    def smart_print(self, text):
        '''
        Reformat a multiline string to contain regular output text, bullet text, and subbullet text. Any line starting with ~ will be given bullet formatting. Any line starting with - will be given subbullet formmatting, and all other lines will be given regular formatting.

        Parameters
        ----------
        text : Str
            Text to smart print
        '''
        if "\n" in text:
            lines = text.split("\n")
        else:
            lines = [text]
        
        for line in lines:
            line = line.strip()
            if len(line) == 0:
                self.output()
            elif line[0] == "~":
                self.bullet(line)
            elif line[0] == "-":
                self.subbullet(line)
            else:
                self.output(line)

    def clear_line(self, n=1):
        '''
        Clear the last line that was printed to the console

        Parameters
        ----------
        n : Int
            How many lines to clear. Default is 1
        '''
        for _ in range(n):
            print('\033[1A', end='\x1b[2K') # Clear last line

# ------------------------------------------------------------------------------
# ArgParse Types
# ------------------------------------------------------------------------------
def ap_valid_path(s):
    '''
    For ArgParse: Check if directory exists. This function should not be called directly. Instead, set the `type` parameter in argparser.add_argument() to this function.

    Parameters
    ----------
    s : Str
        Argument from ArgParse 
    
    Returns
    -------
    Str
        Directory name with absolute path from argument

    Raises
    ------
    argparse.ArgumentTypeError
        If the path is not found
    '''
    if not os.path.isdir(s):
        raise argparse.ArgumentTypeError(f"The path, {s}, was not found.")
    return os.path.abspath(s)

def ap_valid_file(s):
    '''
    For ArgParse: Check if file exists. This function should not be called directly. Instead, set the `type` parameter in argparser.add_argument() to this function.

    Parameters
    ----------
    s : Str
        Argument from ArgParse 

    Returns
    -------
    Str
        File name with absolute path from argument

    Raises
    ------
    argparse.ArgumentTypeError
        If the file is not found
    '''
    if not os.path.isfile(s):
        raise argparse.ArgumentTypeError(f"The file, {s}, was not found.")
    return os.path.abspath(s)

def ap_positive_int(s):
    '''
    For ArgParse: Check if value is positive integer. This function should not be called directly. Instead, set the `type` parameter in argparser.add_argument() to this function.

    Parameters
    ----------
    s : Str
        Argument from ArgParse 

    Returns
    -------
    Int
        Integer value from argument
    
    Raises
    ------
    argparse.ArgumentTypeError
        If the argument can not be converted into an integer or if it is in integer but is not greater than 0
    '''
    try:
        i = int(s)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Expected positive integer, got {s!r}.")
    
    if i <= 0:
        raise argparse.ArgumentTypeError(f"Expected positive integer, got negative: {i}")
    return i

def ap_nonnegative_int(s):
    '''
    For ArgParse: Check if value is positive integer. This function should not be called directly. Instead, set the `type` parameter in argparser.add_argument() to this function.

    Parameters
    ----------
    s : Str
        Argument from ArgParse 
    
    Returns
    -------
    Int
        Integer value from argument

    Raises
    ------
    argparse.ArgumentTypeError
        If the argument can not be converted into an integer or if it is in integer but is not greater than or equal to 0
    '''
    try:
        i = int(s)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Expected positive integer, got {s!r}.")
    
    if i < 0:
        raise argparse.ArgumentTypeError(f"Expected positive integer, got negative: {i}")
    return i

def ap_cpu_int(s):
    '''
    For ArgParse: Check if value is a positive integer or -1 or -2. This function should not be called directly. Instead, set the `type` parameter in argparser.add_argument() to this function.

    Parameters
    ----------
    s : Str
        Argument from Argparse
    
    Returns
    -------
    Int
        Integer value from argument
    
    Raises
    ------
    argparse.ArgumentTypeError
        If the argument cannot be converted into an integer or if it is an integer but is not greater than zero, equal to -1, or equal to -2
    '''
    try:
        i = int(s)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Expected integer, got {s!r}.")
    
    if i <= 0 and not -1 and not -2:
        raise argparse.ArgumentTypeError(f"Expected positive integer, -1, or -2; got other: {i}")
    
    ncpus = multiprocessing.cpu_count()
    if i == -1 or i > ncpus:
        i = ncpus
    elif i == -2:
        i = int(np.floor(ncpus/2))
    return i

def ap_positive_float(s):
    '''
    For ArgParse: Check if value is positive float. This function should not be called directly. Instead, set the `type` parameter in argparser.add_argument() to this function.

    Parameters
    ----------
    s : Str
        Argument from argparse
    
    Returns
    -------
    Int
        Float value from argument
    
    Raises
    ------
    argparse.ArgumentTypeError
        If the argument cannot be converted into an float or if it is an float but is not greater than 0
    '''
    try:
        fl = float(s)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Expected positive float, got {s!r}.")
    
    if fl <= 0:
        raise argparse.ArgumentTypeError(f"Expected positive float, got negative or zero: {fl}")
    return fl

def ap_nonnegative_float(s):
    '''
    For ArgParse: Check if value is a nonnegative float. This function should not be called directly. Instead, set the `type` parameter in argparser.add_argument() to this function.

    Parameters
    ----------
    s : Str
        Argument from argparse
    
    Returns
    -------
    Int
        Float value from argument
    
    Raises
    ------
    argparse.ArgumentTypeError
        If the argument cannot be converted into an float or if it is an float but is not greater than 0
    '''
    try:
        fl = float(s)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Expected nonnegative float, got {s!r}.")
    
    if fl < 0:
        raise argparse.ArgumentTypeError(f"Expected nonnegative float, got negative: {fl}")
    return fl

def ap_nonnegative_frac(s):
    '''
    For ArgParse: Check if value is a float between 0 and 1. This function should not be called directly. Instead, set the `type` parameter in argparser.add_argument() to this function.

    Parameters
    ----------
    s : Str
        Argument from argparse
    
    Returns
    -------
    Int
        Float value from argument
    
    Raises
    ------
    argparse.ArgumentTypeError
        If the argument cannot be converted into an float or if it is an float but is not greater than or equal to 0
    '''
    try:
        fl = float(s)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Expected float, got {s!r}.")
    
    if fl < 0 or fl > 1:
        raise argparse.ArgumentTypeError(f"Expected float between 0 and 1, got {fl}")
    return fl

# ------------------------------------------------------------------------------
# Params
# ------------------------------------------------------------------------------
def _valid_path(p, s):
    '''
    Check if path to directory is valid

    Parameters
    ----------
    p : Name of parameter
    s : Value of parameter
    '''
    if not os.path.isdir(s):
        raise SystemExit(f"Error: parameter {p}: The path, {s}, could not be found.")
    out = os.path.abspath(s)
    if out[-1] == '/':
        out = out[:-1]
    return out

def _valid_file(p, s):
    '''
    Check if file name is valid

    Parameters
    ----------
    p : Name of parameter
    s : Value of parameter
    '''
    if not os.path.isfile(s):
        raise SystemExit(f"Error: parameter {p}: The file, {s}, was not found.")
    return os.path.abspath(s)

def _notype(p, s):
    '''
    A placeholder. Won't check type

    Parameters
    ----------
    p : str
        Name of parameter
    s : str
        Value of parameter. 
    '''
    return s

def _valid_bool(p, s):
    '''
    Check if value is boolean

    Parameters
    ----------
    p : Name of parameter
    s : Value of parameter
    '''
    if type(s) == bool:
        b = s
    elif s.lower() == "true":
        b = True
    elif s.lower() == "false":
        b = False
    else:
        raise SystemExit(f"Error: parameter {p}: Expected boolean, got {s!r}")
    return b

def _positive_int(p, s):
    '''
    Check if value is positive integer
    
    Parameters
    ----------
    p : Name of parameter
    s : Value of parameter
    '''
    try:
        i = int(s)
    except ValueError:
        raise SystemExit(f"Error: parameter {p}: Expected positive integer, got {s!r}.")
    
    if i <= 0:
        raise SystemExit(f"Error: parameter {p}: Expected positive integer, got negative: {i}")
    return i

def _nonnegative_int(p, s):
    '''
    Check if value is nonnegative integer

    Parameters
    ----------
    p : Name of parameter
    s : Value of parameter
    '''
    try:
        i = int(s)
    except ValueError:
        raise SystemExit(f"Error: parameter {p}: Expected positive integer, got {s!r}.")
    
    if i < 0:
        raise SystemExit(f"Error: parameter {p}: Expected positive integer, got negative: {i}")
    return i

def _any_float(p, s):
    '''
    Check if value is a float
    
    Parameters
    ----------
    p : Name of parameter
    s : Value of parameter
    '''
    try:
        fl = float(s)
    except ValueError:
        raise SystemExit(f"Error: parameter {p}: Expected float, got {s!r}.")
    
    return fl

def _positive_float(p, s):
    '''
    Check if value is positive float
    
    Parameters
    ----------
    p : Name of parameter
    s : Value of parameter
    '''
    try:
        fl = float(s)
    except ValueError:
        raise SystemExit(f"Error: parameter {p}: Expected positive float, got {s!r}.")
    
    if fl < 0:
        raise SystemExit(f"Error: parameter {p}: Expected positive float, got negative: {fl}")
    return fl

def _nonnegative_float(p, s):
    '''
    Check if value is nonnegative float
    
    Parameters
    ----------
    p : Name of parameter
    s : Value of parameter
    '''
    try:
        fl = float(s)
    except ValueError:
        raise SystemExit(f"Error: parameter {p}: Expected nonnegative float, got {s!r}.")
    
    if fl <= 0:
        raise SystemExit(f"Error: parameter {p}: Expected nonnegative float, got negative or zero: {fl}")
    return fl

def _positive_frac(p, s):
    '''
    Check if value is a float between 0 and 1

    Parameters
    ----------
    p : Name of parameter
    s : Value of parameter
    '''
    try:
        fl = float(s)
    except ValueError:
        raise SystemExit(f"Error: parameter {p}: Expected float, got {s!r}.")
    
    if fl <= 0 or fl > 1:
        raise SystemExit(f"Error: parameter {p}: Expected float >0 and <=1, got {fl}")
    return fl

def _nonnegative_frac(p, s):
    '''
    Check if value is a float between 0 and 1

    Parameters
    ----------
    p : Name of parameter
    s : Value of parameter
    '''
    try:
        fl = float(s)
    except ValueError:
        raise SystemExit(f"Error: parameter {p}: Expected float, got {s!r}.")
    
    if fl < 0 or fl > 1:
        raise SystemExit(f"Error: parameter {p}: Expected float >=0 and <=1, got {fl}")
    return fl

def _valid_color_nochain(p, s):
    '''
    Check if value is a valid matplotlib color.

    Parameters
    ----------
    p : Str
        Name of parameter
    s : Color
        Value of parameter
    '''
    if s.lower() == "chain":
        raise SystemExit(f"Error: parameter {p}: Expected valid matplotlib color, got {s}. 'chain' not allowed for this color.")
    else:
        if (s[0] == "(" and s[-1] == ")") or s[0] == "[" and s[-1] == "]":
            s=s.replace("(","")
            s=s.replace(")","")
            s=s.replace("[","")
            s=s.replace("]","")
            s=s.replace(" ","")
            s=s.split(",")
        if not is_color_like(s):
            raise SystemExit(f"Error: parameter {p}: Expected valid matplotlib color, got {s}")
    
    return s

def _valid_color_orchain(p, s):
    '''
    Check if value is a valid matplotlib color or "chain"

    Parameters
    ----------
    p : Str
        Name of parameter
    s : Color
        Value of parameter
    '''
    if s.lower() == "chain":
        s = "chain"
    else:
        if (s[0] == "(" and s[-1] == ")") or s[0] == "[" and s[-1] == "]":
            s=s.replace("(","")
            s=s.replace(")","")
            s=s.replace("[","")
            s=s.replace("]","")
            s=s.replace(" ","")
            s=s.split(",")
        if not is_color_like(s):
            raise SystemExit(f"Error: parameter {p}: Expected valid matplotlib color, got {s}")
    
    return s

def _valid_regions(p, s):
    '''
    Check if provided region follows required format. Also reformat into [(pf, (res, res)), ...],
    where each inner tuple describes a continuous path of residues on a protofilament, pf is the
    protofilament index, res1 is the first residue in the path, and res2 is the last residue in 
    the path.
    
    Parameters
    ----------
    p : Name of parameter
    s : Value of parameter
    '''
    if ',' in s: # Separate sides if more than one was provided
        sides = s.split(',')
    else:
        sides = [s]
    
    path = []
    for side in sides:
        if ":" not in side: # Make sure side is formatted [Protofilament]:[Residues]
            raise SystemExit(f"Error: parameter {p}: Provided region definition, {s}, is not properly formatted. See helptext for proper formatting")
        else:
            side = side.split(":")
            if len(side) != 2: # Make sure side is formatted [Protofilament]:[Residues]
                raise SystemExit(f"Error: parameter {p}: Provided region definition, {s}, is not properly formatted. See helptext for proper formatting")
            
            try: # Make sure Protofilament is an integer
                pf = int(side[0])
            except ValueError:
                raise SystemExit(f"Error: parameter {p}: Provided region definition, {s}, is not properly formatted. See helptext for proper formatting")
            
            residues = side[1]
            if "-" not in residues: # Make Sure Residues is formatted [Residue]-[Residue]
                raise SystemExit(f"Error: parameter {p}: Provided region definition, {s}, is not properly formatted. See helptext for proper formatting")
            else:
                residues = residues.split("-")
                if len(residues) != 2: # Make sure 2 residues are given
                    raise SystemExit(f"Error: parameter {p}: Provided region definition, {s}, is not properly formatted. See helptext for proper formatting")
                
                try: # Make sure residues are integers
                    res1 = int(residues[0])
                    res2 = int(residues[1])
                except ValueError:
                    raise SystemExit(f"Error: parameter {p}: Provided region definition, {s}, is not properly formatted. See helptext for proper formatting")
                
                # Add side to path in format tuple(Protofilament, tuple(FirstResidue, LastResidue))
                path.append((pf, (res1, res2)))
    return path

def _valid_calctype(p, s):
    '''
    Check if value is a valid calctype

    Parameters
    ----------
    p : Str
        Name of parameter
    s : Str
        Value of parameter (ALL, HB, SB, PI, HB+SB, HB+PI, or SB+PI)
    '''
    if s.upper() in ["ALL", "HB", "SB", "PI", "HB+SB", "HB+PI", "SB+PI"]:
        s = s.upper()
    else:
        raise SystemExit(f"Error: parameter {p}: Invalid calctype, {s}. Valid options are: ALL, HB, SB, PI, HB+SB, HB+PI, or SB+PI.")
    
    return s

def _valid_sbselmode(p, s):
    '''
    Check if value is a valid saltbridge participant selection mode

    Parameters
    ----------
    p : Str
        Name of parameter
    s : Str
        Value of parameter (ALL, HB, SB, PI, HB+SB, HB+PI, or SB+PI)
    '''
    if s.upper() in ["AUTO", "MANUAL"]:
        s = s.upper()
    else:
        raise SystemExit(f"Error: parameter {p}: Invalid salt bridge participant selection mode, {s}. Valid options are: auto or manual.")
    
    return s

def _valid_cpu_int(p, s):
    '''
    Check if value is acceptable for --nprocs (>0, -1, or -2)

    Parameters
    ----------
    p : Str
        Name of parameter
    s : Str
        Valid of parameter (-1, -2, or > 0)
    '''
    if type(s) == int or s.isnumeric():
        i = int(s)
    else:
        raise SystemExit(f"Error: parameter {p}: Expected integer >=-2, got {s!r}.")
    
    if i <= 0 and not -1 and not -2:
        raise SystemExit(f"Error: parameter {p}: Expected integer >=-2, got {s}.")
    
    ncpus = multiprocessing.cpu_count()
    if i == -1 or i > ncpus:
        i = ncpus
    elif i == -2:
        i = int(np.floor(ncpus/2))
    return i

class Params:
    '''
    Reads and stores run parameters from commandline args, input_file, and checkpoint_file

    Attributes
    ----------
    command : Str
        Name of command to run (i.e. 'calc' or 'map')
    parameters : Various
        All arguments from the commandline or input file have a counterpart in this class. If they are not set, they are NoneType
    output_namestem : Str
        Namestem of output file (other than figure)
    output_cpt : Str
        Name of output_cpt file
    output_log : Str
        Name of output_log file

    Methods
    -------
    get_loglines()
        Retrieve and reset loglines following initialization.
    set_filename(hb_unprocessed_file=None, hb_processed_file=None, sb_unprocessed_file=None, sb_processed_file=None, pi_unprocessed_file=None, pi_processed_file=None, map_positions_file=None)
        Set the name of a file and add to checkpoint file
    '''
    def _get_cmdargs(self, cmdin):
        '''
        Get a list of arguments that were given at the commandline

        Parameters
        ----------
        cmdin : list
            The command entered by user at the commandline (from sys.argv)
        
        Returns
        -------
        cmdargs : list
            List of the arguments given at the commandline
        '''
        abbrevs = {
            "-i":"--input_file",
            "-c":"--checkpoint_file",
            "-f":"--trajectory_file",
            "-t":"--topology_file",
            "-o":{"calc":"--output_directory", "map":"--figure_file"}[self.command],
            "-n":"--n_protofilaments",
            "-v":"--verbose"
        }
        flags = [a for a in cmdin if a[0] == "-"] # just the flags
        cmdargs = []
        for flag in flags:
            if flag in abbrevs.keys(): # used an abbreviation
                flag = abbrevs[flag]
            cmdargs.append(flag[2:]) # Crop off the "--"
        return cmdargs
            
    def _read_input_line(self, line, li, input_file):
        '''
        Read line of input file

        Parameters
        ----------
        line : Str
            A line in an input file
        li : Int
            The line number
        input_file : Str
            The name of the input file

        Returns
        -------
        key : Str
            parameter name
        value : Str
            parameter value
        '''
        if "#" in line: # Get rid of comments
            line = line[:line.find("#")]
        line = line.strip() # Strip off whitespace and new lines
        if len(line) == 0:
            key = None
            value = None
        elif "=" in line:
            key_value = [i.strip() for i in line.split("=")] # Split key and value and remove spaces around the =
            
            if len(key_value) != 2: # Check to make sure only 1 = was provided
                raise SystemExit(f"Parameter Error: {input_file} (line {li}): could not understand line: {line}")

            key = key_value[0] # Get key
            if any(c.isspace() for c in key_value[1]): # Split list at whitespace and get value
                value = key_value[1].split() 
                for i, v in enumerate(value):
                    if v.upper() == "FALSE": # Handle Boolean
                        value[i] = False
                    elif v.upper() == "TRUE":
                        value[i] = True
            else:
                value = key_value[1]
                if value.upper() == "FALSE": # Handle Boolean
                    value = False
                elif value.upper() == "TRUE":
                    value = True

        return key, value

    def _inputfile_reader(self, input_file):
        '''
        Reads parameters from input file or checkpoint file

        Parameters
        ----------
        input_file : Str
            Name of input file or checkpoint file
        '''
        used_exclusives = {"backup":False, "nobackup":False, "log": False, "nolog":False, "saveraw":False, "nosaveraw":False, "hbond_n_cutoff": False, "hbond_p_cutoff": False}
        found_keys = []
        with open(input_file) as file:
            for li, line in enumerate(file.readlines()):
                key, value = self._read_input_line(line, li, input_file)
                if key is None and value is None:
                    continue

                if key in self.__param_info.keys():
                    pinfo = self.__param_info[key]
                    if type(pinfo[0]) == list: # Takes multiple arguments
                        if (len(pinfo[0]) == 1 and (pinfo[0][0] is None or pinfo[0][0] == "REQ")) or key not in found_keys: # No arguments supplied yet
                            if type(value) == str: # Only one new argument provided
                                self.__param_info[key][0] = [value]
                            elif type(value) == list: # More than one new argument provided
                                self.__param_info[key][0] = value
                        else: # Arguments have already been supplied
                            if type(value) == str: # Only one new argument provided
                                self.__param_info[key][0].append(value)
                            elif type(value) == list: # more than one new argument provided
                                self.__param_info[key][0] += value
                    else: # Takes only one argument
                        self.__param_info[key][0] = value
                else:
                    raise SystemExit(f"Parameter Error: {input_file} (line {li}): invalid parameter, {key}.")
                
                if key in used_exclusives.keys():
                    used_exclusives[key] = True
                    if key[:2] == "no":
                        self.__param_info[key[2:]][0] = not self.__param_info[key][0]
                    else:
                        self.__param_info[f"no{key}"][0] = not self.__param_info[key][0]

                found_keys.append(key)

        if used_exclusives["backup"] and used_exclusives["nobackup"]:
            raise SystemExit(f"Parameter Error: {input_file}: Cannot set backup and nobackup at the same time.")
        if used_exclusives["log"] and used_exclusives["nolog"]:
            raise SystemExit(f"Parameter Error: {input_file}: Cannot set log and nolog at the same time.")
        if used_exclusives["saveraw"] and used_exclusives["nosaveraw"]:
            raise SystemExit(f"Parameter Error: {input_file}: Cannot set saveraw and nosaveraw at the same time.")
        if used_exclusives["hbond_n_cutoff"] and used_exclusives["hbond_p_cutoff"]:
            raise SystemExit(f"Parameter Error: {input_file}: Cannot set hbond_n_cutoff and hbond_p_cutoff at the same time.")

    def _compare_checkpoints(self):
        '''
        Call if more than one checkpoint file is provided. If so, compare the contents of each checkpoint file to make sure there are no parameter conflicts.
        '''
        found_parameters = {
            'trajectory_file':[],
            'topology_file':[],
            'output_directory':[],
            'n_protofilaments':[],
            'omit_layers':[],
            'hb_processed_file':[],
            'sb_processed_file':[],
            'pi_processed_file':[],
            'hb_unprocessed_file':[],
            'sb_unprocessed_file':[],
            'pi_unprocessed_file':[],
            'map_positions_file':[]
        }
        for filename in self.__param_info["checkpoint_file"][0]:
            filename = _valid_file("checkpoint_file", filename)
            with open(filename) as file:
                for li, line in enumerate(file.readlines()):
                    key, value = self._read_input_line(line, li, filename)
                    if key is None and value is None:
                        continue

                    if key in found_parameters.keys():
                        found_parameters[key].append(value)
        
        for key, value in found_parameters.items():
            if len(value) < 2:
                continue
            if value.count(value[0]) != len(value):
                raise SystemError(f"Error: parameter conflict in checkpoint files: the checkpoint files specified define {key} differently.")

    def _cli_reader(self, args):
        '''
        Updates parameters with commandline arguments

        Parameters
        ----------
        args : ArgParse.parsed_args
            Parsed args from argparse
        '''
        found_keys = []
        for key in self.__cmdargs:
            value = getattr(args, key)

            if key in self.__param_info.keys():
                pinfo = self.__param_info[key]
                if type(pinfo[0]) == list: # Takes multiple arguments
                    if (len(pinfo[0]) == 1 and (pinfo[0][0] is None or pinfo[0][0] == "REQ")) or key not in found_keys: # No arguments supplied yet
                        if type(value) == str: # Only one new argument provided
                            self.__param_info[key][0] = [value]
                        elif type(value) == list: # More than one new argument provided
                            self.__param_info[key][0] = value
                    else: # Arguments have already been supplied
                        if type(value) == str: # Only one new argument provided
                            self.__param_info[key][0].append(value)
                        elif type(value) == list: # more than one new argument provided
                            self.__param_info[key][0] += value
                else: # Takes only one argument
                    self.__param_info[key][0] = value
                found_keys.append(key)
    # --------------------------------------------------------------------------------

    def __init__(self, args, cmdin):
        '''
        Parameters
        ----------
        args : argparse.Namespace
            Parsed Arguments from argparse
        cmdin : str
            Command entered upon running the program (output of sys.argv)

        Raises
        ------
        SystemExit
            If input file or checkpoint file does not exist, contains an invalid parameter, or is improperly formatted
        '''
        self.command = args.command
        self.__cmdargs = self._get_cmdargs(cmdin)

        # param_info contains each of the valid parameters that could be passed through the commandline, the inputfile, or a checkpoint file for the given command. For each parameter, a list contains its present value (starting at default), and a function that can be used to make sure its final value is of the right type. If the type of the present value is a list, then it can accept more than one value.
        self.__param_info = {
            "trajectory_file":[[None], _valid_file],
            "topology_file":["REQ", _valid_file],
            "checkpoint_file":[{"calc":[None], "map":["REQ"]}[self.command], _valid_file],
            "n_protofilaments":["REQ", _positive_int],
            "omit_layers":[0, _nonnegative_int],
            "output_directory":[os.getcwd(), _valid_path],
            "verbose":[True, _valid_bool],
            "nprocs":[1, _valid_cpu_int], 
            "log":[False, _valid_bool],
            "nolog":[True, _valid_bool],
            "nobackup":[False, _valid_bool],
            "backup":[True, _valid_bool],
            "hb_processed_file":[None, _valid_file],
            "hb_unprocessed_file":[None, _valid_file],
            "sb_processed_file":[None, _valid_file],
            "sb_unprocessed_file":[None, _valid_file],
            "pi_processed_file":[None, _valid_file],
            "pi_unprocessed_file":[None, _valid_file],
            "map_positions_file":[None, _valid_file]
        }
        if self.command == "calc":
            self.__param_info = {**self.__param_info, **{
                "nosaveraw":[False, _valid_bool],
                "saveraw":[True, _valid_bool],
                "calctype":["ALL", _valid_calctype],
                "hbond_distance_cutoff":[3.5, _positive_float],
                "hbond_angle_cutoff":[150, _positive_float],
                "saltbridge_selection_mode":["auto", _valid_sbselmode],
                "saltbridge_anion_charge_cutoff":[-0.5, _any_float],
                "saltbridge_cation_charge_cutoff":[0.5, _any_float],
                "saltbridge_anion_sel":["((resname ASP and name OD1 OD2) or (resname GLU and name OE1 OE2))", _notype],
                "saltbridge_cation_sel":["((resname LYS and name NZ) or (resname ARG and name NH1 NH2 NE) or (resname HSP and name ND1 NE2))", _notype],
                "saltbridge_distance_cutoff":[4.0, _positive_float],
                "pistacking_phe_sel":["(resname PHE and name CG CD2 CE2 CZ CE1 CD1)", _notype],
                "pistacking_tyr_sel":["(resname TYR and name CG CD2 CE2 CZ CE1 CD1)", _notype],
                "pistacking_his_sel":["(resname HSD HSE HSP and name CG CD2 NE2 CE1 ND1)", _notype],
                "pistacking_trp_sel":["(resname TRP and name CG CD1 NE1 CE2 CD2)", _notype],
                }}
        elif self.command == "map":
            self.__param_info = {**self.__param_info, **{
                "figure_file":["figure.png", _notype],
                "p_cutoff":[0.5, _nonnegative_frac],
                "hbond_n_cutoff":[None, _nonnegative_float],
                "hbond_p_cutoff":[None, _nonnegative_frac],
                "saltbridge_p_cutoff":[None, _nonnegative_frac],
                "pistacking_p_cutoff":[None, _nonnegative_frac],
                "transparent_background":[False, _valid_bool],
                "numbered_residues":[False, _valid_bool],
                "water_region":[[None], _valid_regions],
                "zipper_region":[[None], _valid_regions],
                "figure_width":[6.5, _positive_float],
                "figure_height":[4.5, _positive_float],
                "legend":[True, _valid_bool],
                "figure_dpi":[300, _positive_int],
                "acidic_color":["firebrick", _valid_color_nochain],
                "acidic_label_color":["white", _valid_color_orchain],
                "basic_color":["steelblue", _valid_color_nochain],
                "basic_label_color":["white", _valid_color_orchain],
                "polar_color":["seagreen", _valid_color_nochain],
                "polar_label_color":["white", _valid_color_orchain],
                "nonpolar_color":["white", _valid_color_nochain],
                "nonpolar_label_color":["chain", _valid_color_orchain],
                "backbone_color":[["black", "dimgray"], _valid_color_nochain],
                "hbond_color_1":["black", _valid_color_nochain],
                "hbond_color_2":["white", _valid_color_nochain],
                "saltbridge_color_1":["gold", _valid_color_nochain],
                "saltbridge_color_2":["orange", _valid_color_nochain],
                "saltbridge_color_3":["white", _valid_color_nochain],
                "pistacking_color_1":["gray", _valid_color_nochain],
                "pistacking_color_2":["white", _valid_color_nochain],
                "water_color":["powderblue", _valid_color_nochain],
                "water_opacity":[0.5, _positive_float],
                "zipper_color":["tan", _valid_color_nochain],
                "zipper_opacity":[0.5, _positive_float],
                "showfig":[False, _valid_bool]
            }}
        
        # Collect critical arguments from commandline
        self.__input_file = args.input_file

        # Collect parameters from various sources
        # First, collect from input file (These arguments are of the lowest priority)
        self._inputfile_reader(self.__input_file)
        # Second, collect from commandline (These arguments trump input file arguments but not checkpoint file arguments)
        # for arg in self.__cmdargs:
        self._cli_reader(args)
        # Third, If using checkpoint file(s), collect arguments from it. (These arguments are of the highest priority)
        if self.__param_info["checkpoint_file"][0][0] != "REQ" and self.__param_info["checkpoint_file"][0][0] is not None:
            self._compare_checkpoints() # Check to make sure there are not conflicts
            for cptfile in self.__param_info["checkpoint_file"][0]:
                self._inputfile_reader(cptfile)
        elif self.command == "map": # Checkpoint file is required for map subcommand
            raise SystemExit("Parameter Error: -c / --checkpoint_file: Checkpoint file is required for the map command but was not provided.")
        
        # Check to make sure parameters are valid
        for pname, pvalue in self.__param_info.items():
            if type(pvalue[0]) == list:
                pvals = []
                for pval in pvalue[0]:
                    if pval == "REQ":
                        raise SystemExit(f"Parameter Error: The {pname} parameter is required, but none was provided.")
                    elif pval is not None:
                        pvals.append(pvalue[1](pname, pval))
                if len(pvals) == 0:
                    setattr(self, pname, None)
                else:
                    setattr(self, pname, pvals)
            else:
                if pvalue[0] == "REQ":
                    raise SystemExit(f"Parameter Error: The {pname} parameter is required, but none was provided.")
                elif pvalue[0] is not None:
                    setattr(self, pname, pvalue[1](pname, pvalue[0]))
                else:
                    setattr(self, pname, None)
        
        # If using a legend, make sure figure size is large enough
        if self.command == "map" and self.legend:
            if self.figure_height < 2.5 or self.figure_width < 2:
                raise SystemExit(f"Parameter Error: When legend=True, figure_height must be greater than 2.5 inches and figure_width must be greater than 2 inches. User set figure_height={self.figure_height} and figure_width={self.figure_width}")

        # Check to make sure topology and trajectory files are compatible
        try:
            u = mda.Universe(self.topology_file, self.trajectory_file)
        except:
            if self.trajectory_file == self.topology_file:
                raise SystemExit(f"Error: parameter topology_file : Could not construct universe from the topology file {self.topology_file} without a trajectory file. Check to make sure topology file contains coordinate information.")
            else:
                raise SystemExit(f"Error: parameters topology_file & trajectory_file : Could not construct universe from the topology file {self.topology_file}, and the trajectory file(s), {self.trajectory_file}.")
        if not hasattr(u.atoms, 'charges'): # Check to make sure topology file contained charge information
            raise SystemExit(f"Error: parameter topology_file: Topology file, {self.topology_file}, is missing required charge information.")
        if not hasattr(u, 'bonds'):
            raise SystemExit(f"Error: parameter topology_file: Topology file, {self.topology_file}, is missing required bond information.")
        if not hasattr(u.residues, 'segids') or u.select_atoms("protein").segments.n_segments == 1:
            raise SystemExit(f"Error: parameter topology_file: Topology file, {self.topology_file}, either only contains one segment or does not contain segids.")
        
        # Reconcile cutoff parameters
        if self.command == "map":
            self.hbond_cutoff = self.p_cutoff
            self.use_hbond_n_cutoff = False
            self.saltbridge_cutoff = self.p_cutoff
            self.pistacking_cutoff = self.p_cutoff
            if self.hbond_n_cutoff is not None:
                self.use_hbond_n_cutoff = True
                self.hbond_cutoff = self.hbond_n_cutoff
            if self.hbond_p_cutoff is not None:
                self.hbond_cutoff = self.hbond_p_cutoff
            if self.saltbridge_p_cutoff is not None:
                self.saltbridge_cutoff = self.saltbridge_p_cutoff
            if self.pistacking_p_cutoff is not None:
                self.pistacking_cutoff = self.pistacking_p_cutoff
            del self.p_cutoff
            del self.hbond_n_cutoff
            del self.hbond_p_cutoff
            del self.saltbridge_p_cutoff
            del self.pistacking_p_cutoff
        
        # Reconcile nobackup/backup and log/nolog and saveraw/nosaveraw
        if self.backup == True:
            self.nobackup = False
        if self.nolog == True:
            self.log = False
        if self.command == "calc" and self.saveraw == True:
            self.nosaveraw = False
            del self.saveraw
        del self.backup
        del self.nolog

        # Reconcile saltbridge selection options
        if self.command == "calc":
            if self.saltbridge_selection_mode.lower() == "auto":
                del self.saltbridge_anion_sel
                del self.saltbridge_cation_sel
            elif self.saltbridge_selection_mode.lower() == "manual":
                del self.saltbridge_anion_charge_cutoff
                del self.saltbridge_cation_charge_cutoff
        
        # Set Output Names
        self.__loglines = []
        if self.command == "calc":
            self.output_namestem = f"calc_{self.calctype.replace('+', '-')}"
        else:
            self.output_namestem = f"map"
            if not self.nobackup:
                self.__loglines.append(file_backup(self.output_directory, self.figure_file)) # Backup Previously Made Figures
            self.figure_file = f"{self.output_directory}/{self.figure_file}" # Update output figure file

        if self.log and not self.nobackup:
            self.__loglines.append(file_backup(self.output_directory, f"{self.output_namestem}.log")) # Backup Previously Made LogFiles
        self.output_namestem = f"{self.output_directory}/{self.output_namestem}"
        if self.log:
            self.output_log = f"{self.output_namestem}.log"
        else:
            self.output_log = None
        del self.log
        self.output_cpt = f"{self.output_namestem}.cpt"

        # Write Starting State Checkpoint File
        if self.command == "calc" and self.checkpoint_file is not None and type(self.checkpoint_file) != list:
            self.output_cpt = self.checkpoint_file
        else:
            with open(self.output_cpt, "w+") as cpt:
                cpt.write(f"# Created checkpoint file on {datetime.now().strftime('%d/%m/%Y at %H:%M:%S')}\n")
                if type(self.trajectory_file) == str:
                    cpt.write(f"trajectory_file = {self.trajectory_file}\n")
                else:
                    cpt.write(f"trajectory_file = {' '.join(self.trajectory_file)}\n")
                cpt.write(f"topology_file = {self.topology_file}\n")
                cpt.write(f"output_directory = {self.output_directory}\n")
                cpt.write(f"n_protofilaments = {self.n_protofilaments}\n")
                cpt.write(f"omit_layers = {self.omit_layers}\n")
                if self.hb_processed_file is not None:
                    cpt.write(f"hb_processed_file = {self.hb_processed_file}\n")
                if self.command == 'calc' and self.hb_unprocessed_file is not None:
                    cpt.write(f"hb_unprocessed_file = {self.hb_unprocessed_file}\n")
                if self.sb_processed_file is not None:
                    cpt.write(f"sb_processed_file = {self.sb_processed_file}\n")
                if self.command == 'calc' and self.sb_unprocessed_file is not None:
                    cpt.write(f"sb_unprocessed_file = {self.sb_unprocessed_file}\n")
                if self.pi_processed_file is not None:
                    cpt.write(f"pi_processed_file = {self.pi_processed_file}\n")
                if self.command == 'calc' and self.pi_unprocessed_file is not None:
                    cpt.write(f"pi_unprocessed_file = {self.pi_unprocessed_file}\n")
                if self.command == 'map' and self.map_positions_file is not None:
                    cpt.write(f"map_positions_file = {self.map_positions_file}\n")
    
    def __str__(self):
        types = {
            "Input": ["trajectory_file","topology_file","checkpoint_file","hb_processed_file","hb_unprocessed_file","sb_processed_file","sb_unprocessed_file","pi_processed_file","pi_unprocessed_file","map_positions_file"],
            "Output": ["output_directory","nosaveraw","saveraw","verbose","nprocs", "log","nolog","nobackup","backup","figure_file","showfig", "output_log", "output_cpt"],
            "Options": ["n_protofilaments","omit_layers","calctype","hbond_distance_cutoff","hbond_angle_cutoff","saltbridge_selection_mode","saltbridge_anion_charge_cutoff","saltbridge_cation_charge_cutoff","saltbridge_anion_sel","saltbridge_cation_sel","saltbridge_distance_cutoff","pistacking_phe_sel","pistacking_tyr_sel","pistacking_his_sel","pistacking_trp_sel"],
            "Figure Options":["figure_width","figure_height","legend","figure_dpi","transparent_background","numbered_residues"],
            "Regions": ["water_region","zipper_region"],
            "Cutoffs": ["p_cutoff","hbond_n_cutoff","hbond_p_cutoff","saltbridge_p_cutoff","pistacking_p_cutoff"],
            "Colors": ["acidic_color","acidic_label_color","basic_color","basic_label_color","polar_color","polar_label_color","nonpolar_color","nonpolar_label_color","backbone_color","hbond_color_1","hbond_color_2","saltbridge_color_1","saltbridge_color_2","saltbridge_color_3","pistacking_color_1","pistacking_color_2","water_color","water_opacity","zipper_color","zipper_opacity"]}
        
        return_string = f"  - command: {self.command}\n"
        for section, params in types.items():
            found = False
            for param in params:
                if hasattr(self, param):
                    if not found:
                        return_string += f"{section}\n"
                        found = True
                    value = getattr(self, param)
                    if value is not None:
                        if "file" in param or param in ["output_directory", "output_log", "output_cpt"]:
                            value = utils.relative_path(value)
                        if type(value) == list:
                            if type(value[0]) != str:
                                value = ', '.join(map(str, value))
                            else:
                                value = ', '.join(value)
                    return_string += f"  - {param}: {value}\n"

        return return_string

    def get_loglines(self):
        '''
        Retrieve and reset loglines following initialization.
        '''
        loglines = "".join(self.__loglines)
        del self.__loglines
        return loglines
    
    def set_filename(self, hb_unprocessed_file=None, hb_processed_file=None, 
                     sb_unprocessed_file=None, sb_processed_file=None, 
                     pi_unprocessed_file=None, pi_processed_file=None, 
                     map_positions_file=None):
        '''
        Set the name of a file and add to checkpoint file

        Parameters
        ----------
        hb_unprocessed_file : Str or None [Default is None]
            the new name of the unprocessed hydrogen bond file
        hb_processed_file : Str or None [Default is None]
            the new name of the processed hydrogen bond file
        sb_unprocessed_file : Str or None [Default is None]
            the new name of the unprocessed salt bridge file
        sb_processed_file : Str or None [Default is None]
            the new name of the processed salt bridge file
        pi_unprocessed_file : Str or None [Default is None]
            the new name of the unprocessed pi stacking interaction file
        pi_processed_file : Str or None [Default is None]
            the new name of the processed pi stacking interaction file
        map_positions_file : Str or None [Default is None]
            the new name of the map positions file
        '''
        def _add_to_checkpoint(parameter_name, parameter_value, cpt_file):
            with open(cpt_file, "a") as file:
                file.write(f"# The following addition was made to checkpoint file on {datetime.now().strftime('%d/%m/%Y at %H:%M:%S')}\n")
                file.write(f"{parameter_name} = {parameter_value}\n")
        #----------------------------------------------------------------------

        if hb_unprocessed_file is not None:
            self.hb_unprocessed_file = hb_unprocessed_file
            _add_to_checkpoint("hb_unprocessed_file", hb_unprocessed_file, self.output_cpt)
        if hb_processed_file is not None:
            self.hb_processed_file = hb_processed_file
            _add_to_checkpoint("hb_processed_file", hb_processed_file, self.output_cpt)
        if sb_unprocessed_file is not None:
            self.sb_unprocessed_file = sb_unprocessed_file
            _add_to_checkpoint("sb_unprocessed_file", sb_unprocessed_file, self.output_cpt)
        if sb_processed_file is not None:
            self.sb_processed_file = sb_processed_file
            _add_to_checkpoint("sb_processed_file", sb_processed_file, self.output_cpt)
        if pi_unprocessed_file is not None:
            self.pi_unprocessed_file = pi_unprocessed_file
            _add_to_checkpoint("pi_unprocessed_file", pi_unprocessed_file, self.output_cpt)
        if pi_processed_file is not None:
            self.pi_processed_file = pi_processed_file
            _add_to_checkpoint("pi_processed_file", pi_processed_file, self.output_cpt)
        if map_positions_file is not None:
            self.map_positions_file = map_positions_file
            _add_to_checkpoint("map_positions_file", map_positions_file, self.output_cpt)





