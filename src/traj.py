import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, AutoMinorLocator, FixedLocator
import matplotlib as mpl
import MDAnalysis as mda
from scipy.stats import binned_statistic

# CONTAINS:
# Contains functions used for trajectory analysis
#        TrajectoryAnalysis: Calculates average number of interactions per frame

# ------------------------------------------------------------------------------
# TRAJ ANALYSIS
# ------------------------------------------------------------------------------
class TrajectoryAnalysis:
    '''
    Calculate the average number of a given type of interaction per frame

    Attributes
    ----------
    n_total
    n_intralayer
    n_interlayer
    n_times

    Methods
    -------
    add_interaction_type()

    make_figure()
    '''
    def __init__(self, sysinfo, params):
        '''
        Parameters
        ----------
        sysinfo : utils.SystemInfo
            System information
        params : io.Params
            Parameters
        '''
        self.__sysinfo = sysinfo
        self.__params = params

        # need the total number of frames
        u = mda.Universe(self.__params.topology_file, self.__params.trajectory_file)
        self.__n_frames = u.trajectory.n_frames
        self.__timestep = u.trajectory.dt

        self.n_interlayer = {}
        self.n_intralayer = {}
        self.n_total = {}
        self.__n_incl = 0
        self.__rawtimes = np.arange(0, self.__n_frames)*self.__timestep
        self.__ymax = 0

        # Convert times to reasonable units
        maxtime = self.__rawtimes.max()
        timeOoM = np.floor(np.log10(maxtime))
        cexp = int(timeOoM-1)
        if cexp > 0:
            if cexp % 3 != 0: # If not whole units
                cexp = int(np.round(cexp/3)*3) # round to nearest whole units
            self.__time_units = ["ps", "Err", "Err", "ns", "Err", "Err", r"m$\mathrm{\micro}$", "Err", "Err", "ms"][cexp]
        else: # Getting smaller
            negcexp = -cexp
            if negcexp % 3 != 0: # If not whole units
                negcexp = int(np.round(negcexp/3)*3) # round to nearest whole units
                cexp = -negcexp
            self.__time_units = ["ps", "Err", "Err", "fs", "Err", "Err", "as"][negcexp]
        conversion_factor = (10**(-cexp))
        self.times = self.__rawtimes * conversion_factor

    def add_interaction_type(self, frames, layer_a, layer_b, typename):
        '''
        Add a new interaction type to the figure

        Parameters
        ----------
        frames : np.ndarray
            1D array containing the frames of all calculated interactions
        layer_a : np.ndarray
            1D array containing the first layer interaction of all calculated interactions. Must be same size as layer_b.
        layer_b : np.ndarray
            1D array containing the second layer interaction of all calculated interactions. Must be same size as layer_a.
        typename : "HB", "SB", "PI"
            Name of interaction. HB=Hydrogen Bonds, SB=Salt Bridges, PI=Pi Stacking
        '''
        # Process the interactions
        layerspan = np.abs(layer_a-layer_b)
        all_interactions = np.column_stack((frames, self.__sysinfo.structure.shape[0]-layerspan, layerspan))
        all_interactions[all_interactions[:,-1]!=0,-1] = 1 # Intralayer=0, Interlayer=1
        # all_interaction columns:: 0:frames, 1:n_possible (max possible for given interaction per frame), 2:intralayerVinterlayer

        unique_interactions, unique_counts = np.unique(all_interactions, axis=0, return_counts=True)
        weighted_counts = unique_counts/unique_interactions[:,1]

        bins = np.arange(0, self.__n_frames+1)-0.5
        sel_intra = unique_interactions[:,-1] == 0
        self.n_intralayer[typename], _, _ = binned_statistic(unique_interactions[sel_intra,0], weighted_counts[sel_intra], bins=bins)
        sel_inter = unique_interactions[:,-1] == 1
        self.n_interlayer[typename], _, _ = binned_statistic(unique_interactions[sel_inter,0], weighted_counts[sel_inter], bins=bins)

        self.n_total[typename] = self.n_intralayer[typename]+self.n_interlayer[typename]

        checkymax = np.max((self.n_intralayer[typename].max(), self.n_interlayer[typename].max(), self.n_total[typename].max()))
        if checkymax > self.__ymax:
            self.__ymax = checkymax

        self.__n_incl += 1
    
    def show(self, LOG):
        '''
        Print a summary of the results

        Parameters
        ----------
        LOG : io.Logger
            The logger object used for returning standard output to the console and logfile
        '''
        LOG.output("Results Summary:")
        typelabels = {"HB":"Hydrogen Bonds:", "SB":"Salt Bridges:", "PI":"Pi Stacking Interactions:"}
        for typename in self.n_total.keys():
            LOG.output(typelabels[typename])
            total = f"{np.round(np.mean(self.n_total[typename]), 3):.3f}+-{np.round(np.std(self.n_total[typename]), 3):.3f}"
            intra = f"{np.round(np.mean(self.n_intralayer[typename]), 3):.3f}+-{np.round(np.std(self.n_intralayer[typename]), 3):.3f}"
            inter = f"{np.round(np.mean(self.n_interlayer[typename]), 3):.3f}+-{np.round(np.std(self.n_interlayer[typename]), 3):.3f}"
            LOG.output(f"{'N(Intralayer)/layer':>24}: {intra}")
            LOG.output(f"{'N(Interlayer)/layer':>24}: {inter}")
            LOG.output(f"{'N(Total)/layer':>24}: {total}")
        LOG.output("")

    def save(self):
        '''
        Save results
        '''
        results_dict = {}
        results_dict["times"] = self.__rawtimes
        for typename in ["HB", "SB", "PI"]:
            if typename in self.n_interlayer.keys() and typename in self.n_intralayer.keys() and typename in self.n_total.keys():
                results_dict[typename] = np.column_stack((self.n_intralayer[typename], self.n_interlayer[typename], self.n_total[typename]))
        np.savez(f"{self.__params.output_directory}/traj_results.npz", **results_dict)

    def make_figure(self):
        '''
        Create the figure
        '''
        def _roundupOoM(n):
            # Round n up to nearest multiple of order of magnitude
            m = 10**np.floor(np.log10(n))
            return np.ceil(n/m)*m

        mpl.rcParams['font.sans-serif'] = "Arial"
        mpl.rcParams['font.family'] = "sans-serif"
        mpl.rcParams['font.size'] = 6

        fig, ax = plt.subplots(nrows=self.__n_incl, ncols=1, figsize=(self.__params.figure_width, self.__params.figure_height), gridspec_kw={"hspace":0}, dpi=self.__params.figure_dpi)
        colors = {"HB":self.__params.hbond_color, "SB":self.__params.saltbridge_color, "PI":self.__params.pistacking_color}
        ylabels  = {"HB": "H-Bonds", "SB":"Salt Bridges", "PI":"Pi Stacking"}
        for ai, (axis, typename) in enumerate(zip(ax, self.n_total.keys())):
            axis.plot(self.times, self.n_total[typename], color=colors[typename], linestyle="solid", linewidth=1, label="Total")
            axis.plot(self.times, self.n_intralayer[typename], color=colors[typename], linestyle="dashed", linewidth=1, label="Intralayer")
            axis.plot(self.times, self.n_interlayer[typename], color=colors[typename], linestyle="dotted", linewidth=1, label="Interlayer")

            axis.set_xlim(self.times.min(), _roundupOoM(self.times.max()))
            axis.set_ylim(0, _roundupOoM(self.__ymax))

            if len(ax) == 3:
                if ai == 0:
                    prune = None
                    xticks = "top"
                    xlabel = False
                    legend = True
                elif ai == 1:
                    prune = 'upper'
                    xticks = "off"
                    xlabel = False
                    legend = False
                elif ai == 2:
                    prune = 'upper'
                    xticks = "bottom"
                    xlabel = True
                    legend = False
            elif len(ax) == 2:
                if ai == 0:
                    prune = None
                    xticks = "top"
                    xlabel = False
                    legend = True
                elif ai == 1:
                    prune = 'upper'
                    xticks = "bottom"
                    xlabel = True
                    legend = False
            else:
                prune = None
                xticks = "bottom"
                xlabel = True
                legend = True
            
            axis.yaxis.set_major_locator(MaxNLocator(nbins=5, prune=prune, min_n_ticks=4))
            axis.yaxis.set_minor_locator(AutoMinorLocator())
            if len(self.times) <= 10:
                axis.xaxis.set_major_locator(FixedLocator(self.times))
                usingwhichxticks = "major"
            else:
                axis.xaxis.set_major_locator(MaxNLocator(nbins=10, min_n_ticks=4))
                axis.xaxis.set_minor_locator(AutoMinorLocator())
                usingwhichxticks = "both"

            axis.tick_params(which=usingwhichxticks, axis="x", bottom=xticks=="bottom", top=xticks=="top")
            axis.tick_params(which="major", axis="x", labelbottom=xticks=="bottom")

            if xlabel:
                axis.set_xlabel(f"time ({self.__time_units})")
            axis.set_ylabel(r"$\mathrm{N_{"+ylabels[typename]+r"}/layer}$")
            if legend:
                axis.legend(ncol=3, edgecolor="inherit", framealpha=1)
        plt.tight_layout()
        plt.savefig(self.__params.figure_file)
        plt.show()






