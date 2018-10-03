from __future__ import print_function
import warnings
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import glob

import lsst.sims.maf.metricBundles as mb
import lsst.sims.maf.plots as plots
from lsst.sims.maf.runComparison import RunComparison

_TABULATE_HERE = True

try:
    from tabulate import tabulate
except ImportError:
     _TABULATE_HERE = False
     warnings.warn('\n'+'This class requires tabulate to be installed'+'\n'+
                 'in order output markdown tables.'+'\n'+
                 'Run: pip install tabulate then restart your jupyter notebook kernel.')


__all__ = ['standardComp']


class standardComp(object):
    """
    Class to create a standard comparison of OpSim runs and produce a Markdown
    page that is easily shared on GitHub.

    Parameters
    ----------
    compOutDir : str
        The name of a directory where you would like to save the comparison output
        tables and plots.
     mafSubDir : str
        The directory with the MAF output that will be used for the comparison.
    runList : list
        List of runs that will be compared.
    compObject : object (opt)
        A comparison object created by lsst.sims.maf.runComparison. If a
        runComparison object is not provided, it will be created based on the
        runList and the mafSubDir.
    """
    def __init__(self, compOutDir, mafSubDir, runList, compObject = None,
                 mapDict = None, histDict = None, md_tables = None, metricList = None):
        self.compOutDir = compOutDir
        self.mafSubDir = mafSubDir
        self.runList = runList
        self.compObject = compObject

        if not os.path.exists(self.compOutDir+'/figures'):
            os.makedirs(self.compOutDir+'/figures')

        # histDict is {plot name:,plotDict]}
        if mapDict is None:
            self.mapDict = {'NVisits all bands HealpixSlicer':
                            {'figsize':(12,4),'colorMin':700,'colorMax':1200},
                            'Nvisits as function of Alt/Az all bands HealpixSlicer':
                            {'figsize':(12,4),'rot': (90, 90, 90), 'flip': 'geo'},
                            'Median airmass all bands HealpixSlicer':
                            {'figsize':(12,4),'colorMin':1.02,'colorMax':1.31},
                            'Max airmass all bands HealpixSlicer':
                            {'figsize':(12,4),'colorMin':1.1,'colorMax':2.5},
                            'CoaddM5 r band HealpixSlicer':
                            {'figsize':(12,4),'colorMin':-3.0,'colorMax':0.0},
                            'Normalized Parallax @ 22.4 All visits HealpixSlicer':
                            {'figsize':(12,4),'colorMin':0.5,'colorMax':1.0},
                            'Normalized Proper Motion @ 20.5 All visits HealpixSlicer':
                            {'figsize':(12,4),'colorMin':0.42,'colorMax':0.62}}
        else:
            self.mapDict = mapDict

        # histDict is {plot name: [plot function, plotDict]}
        if histDict is None:
            self.histDict = {'CoaddM5 r band HealpixSlicer':[plots.HealpixHistogram(),
                                                             {'figsize':(6,6),
                                                              'xMin':-3.5,
                                                              'xMax':0.5,
                                                              'yMax':11500,
                                                              'zp':27.6,
                                                              'linewidth':3.,
                                                              'bins':np.arange(-4.5,0.5,0.1)}],
                         'Slew Time Histogram All visits OneDSlicer':[plots.OneDBinnedData(),
                                                                      {'figsize':(6,6),
                                                                       'xMin':-10,'xMax':160,
                                                                       'linewidth':2,
                                                                       'bins':np.arange(0,155,5)}],
                         'Zoom Slew Time Histogram All visits OneDSlicer':[plots.OneDBinnedData(),
                                                                           {'figsize':(6,6),
                                                                           'xMin':3.5,
                                                                           'xMax':8.0,
                                                                           'yMax':985000,
                                                                           'linewidth':2,
                                                                           'bins':np.arange(0,155,5),
                                                                           'logScale':False}],
                         'Slew Distance Histogram All visits OneDSlicer':[plots.OneDBinnedData(),
                                                                          {'figsize':(6,6),
                                                                           'xMin':-10,
                                                                          'xMax':140,
                                                                          'linewidth':2,
                                                                          'bins':np.arange(0,155,1)}],
                         'Zoom Slew Distance Histogram All visits OneDSlicer':[plots.OneDBinnedData(),
                                                                               {'figsize':(6,6),
                                                                                'xMin':-1,
                                                                                'xMax':10,'yMax':985000,
                                                                                'linewidth':2,
                                                                                'bins':np.arange(0,155,1),
                                                                                'logScale':False}]}
        else:
            self.histDict = histDict

        # File names of markdown tables that will be created and saved.
        self.metricList = metricList
        if md_tables is None:
            self.md_tables = ['fo.md',
                              'total_teff.md',
                              'norm_teff.md',
                              'open_shutter.md',
                              'parallax.md',
                              'prop_mo.md',
                              'rapid_revist.md',
                              'fraction_pairs.md',
                              'slew.md',
                              'filters.md',
                              'nvisits.md',
                              'proposal_fractions.md',
                              'med_nvists_wfd.md',
                              'med_coadd_wfd.md',
                              'med_fivesig_wfd.md',
                              'med_internight_wfd.md',
                              'med_airmass_wfd.md',
                              'med_seeing_wfd.md']
        else:
            self.md_tables = md_tables

        # Metrics that will be compared
        if metricList is None:
            self.metricList = ['fO',
                               'Total Teff',
                               'Normalized Teff WFD all bands',
                               'Open',
                               'Median Parallax',
                               'Median Proper Motion',
                               'RapidRevisits',
                               'Median Fraction of visits in pairs',
                               'slew',
                               'Filter',
                               'Nvisits All props',
                               'Fraction of total',
                               'Median NVisits WFD',
                               'Median CoaddM5 WFD',
                               'Median Median Inter-Night Gap WFD',
                               'Median Median fiveSigmaDepth WFD',
                               'Median Median airmass WFD',
                               'Median Median seeingEff WFD']
        else:
            self.metricList = metricList

        # Dictionary of {Metric:markdown table}
        self.metricSaveDict = dict(zip(self.metricList, self.md_tables))

        if compObject is None:
            self.compObject = RunComparison(baseDir='.',runlist=runList)
            self.metric_dict = self.compObject.buildMetricDict(subdir=mafSubDir)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                self.compObject.addSummaryStats(self.metric_dict)

        if compObject is not None:
            self.metric_dict = self.compObject.buildMetricDict(subdir=mafSubDir)

    def compTable(self, metricLike, output_file=None, drop_cols = True):
        """
        Produce a markdown comparison table based on the metricLike name provided.

        Parameters
        ----------
        metricLike : str
            String containing the metric you would like to compare.
        output_file : str, opt
            File name for markdown table output. If this is left None, no file
            is saved.
        drop_cols : bool
            If True, will drop metric that include Max, Min, Sigma, and RMS in
            their names.

        Returns
        -------
        pandas dataframe
            The returned dataframe only includes the comparison for the metricLike
            name provided.

        """
        df = self.compObject.summaryStats.T.filter(like=metricLike,axis=0)
        cols = df.dropna().T.columns.values
        df['idx'] = cols
        if ('Sigma' in metricLike) or ('fO' == metricLike):
            drop_cols=False
        if drop_cols:
            new_df = df[(df['idx'].str.contains('Max|Min|Sigma|Rms'))==False].round(3)
        else:
            new_df = df.round(3)
        if output_file is not None:
            out_put = os.path.join(self.compOutDir+'/figures',output_file)
            with open(out_put, 'w') as outputfile:
                print(tabulate(new_df[self.runList], tablefmt="pipe", headers="keys",showindex=True,),
                      file=outputfile)
        return new_df[self.runList]

    def compPlot(self, plotName, plotFunc, colorList=None, savefig=False,
                 userPlotDict=None, setlineColor = False):
        """
        Read metric data and produce a comparison plot.

        Parameters
        ----------
        plotName : str
            String containing the metric you would like to compare.
        plotFunc : lsst.sims.maf.plots
            MAF Plot function
        colorList : list, opt
            List of colors to use for lines in histogram comparisons.
        savefig : bool
            If false the comparison figure is not saved
        userPlotDict : dict, opt
            Dictionary to to set plot parameters
        setlineColor : bool
            If True the line colors used in the histograms will be set by
            the colorList.


        Returns
        -------
        pandas dataframe
            The returned dataframe only includes the comparison for the metricLike
            name provided.

        """

        outDir = self.compOutDir+'/figures'
        metricDict = self.metric_dict

        if colorList is None:
            colorList = ['r','b','k','g','m','c']

        colorDict = {}
        for i,r in enumerate(self.runList):
            colorDict[r] = colorList[i]

        metricData, metricName = self.compObject.readMetricData(metricDict[plotName]['metricName'],
                                                                metricDict[plotName]['metricMetadata'],
                                                                metricDict[plotName]['slicerName'])
        if setlineColor:
            for r in self.runList:
                metricData[r].setPlotDict({'color':colorDict[r]})

        if userPlotDict is None:
            userPlotDict = {}

        with warnings.catch_warnings():
            self.compObject.plotMetricData(metricData,
                                           plotFunc=plotFunc,
                                           userPlotDict=userPlotDict,
                                           outDir=outDir,savefig=savefig)


    def generateAll(self, showTables = True, plotMaps = True, plotHists = True):
        """
        Generate all of the comparison markdown tables and plots.

        Parameters
        ----------
        showTables : bool, opt
            If True, the comparison tables are printed as the are produced
        plotMaps : bool, opt
            If True the comparison maps are plotted and saved. This is useful to
            set to false if you do not need to recreate the map plots.
        plotHists : bool, opt
            If True the comparison histograms are plotted and saved. This is useful to
            set to false if you do not need to recreate the histogram plots.

        Results
        --------
        A new attribute is added to the standardComp called comboComp. This is
        a dataframe that combines all of the metrics compared into a single
        dataframe.

        """
        df_list = []
        for metric in self.metricList:
            metricDF = self.compTable(metric, output_file=self.metricSaveDict[metric], drop_cols = True)
            if showTables:
                print (metric)
                print (metricDF)
            df_list.append(metricDF)

        self.comboComp = pd.concat(df_list)

        if plotMaps:
            for maps in self.mapDict.keys():
                self.compPlot(maps,userPlotDict=self.mapDict[maps], plotFunc=plots.HealpixSkyMap(), savefig=True)

        if plotHists:
            for hists in self.histDict.keys():
                self.compPlot(hists,userPlotDict=self.histDict[hists][1],
                              plotFunc=self.histDict[hists][0], setlineColor=True, savefig=True)

    def mdMaker(self, compRoot = None):
        """
        Generate a single markdown file that with links to the comparison plots
        and tables.

        Parameters
        ----------
        compRoot : str, opt
            The root string of the comparison plots in self.compOutDir/figures.
            This is needed to created the correct relative links in the markdown
            page.
        """
        outDir = self.compOutDir

        if compRoot is None:
            # The order of the run names can change, but the length
            # of the starting string will always be the same.
            runString = '_'.join(self.runList)+'_'
            pdfNames = [os.path.basename(x) for x in glob.glob(self.compOutDir+'/figures/*.pdf*')]
            compRoot = pdfNames[0][:len(runString)]


        # Markdown headers for summary stats comparison tables
        headers = ['# fO',
                   '# Total Effective Time',
                   '# Normalized Effective Time',
                   '# Open Shutter Fraction',
                   '# Parallax',
                   '# Proper Motion',
                   '# Rapid Revisit',
                   '# Fraction in Pairs',
                   '# Slews',
                   '# Filter Changes',
                   '# Nvisits',
                   '# Proposal Fractions',
                   '# Median Nvisits WFD',
                   '# Median CoaddM5 WFD',
                   '# Median FiveSigmaDepth',
                   '# Median Internight Gap',
                   '# Median Airmass WFD',
                   '# Median Seeing WFD']

        # Make markdown links for table of contents
        # Makes items in headers lowercase and replaces ' ' with '-'
        toc = [x.replace('# ',"#").replace(' ','-').lower() for x in headers]

        sky_maps = ['NVisits_all_bands_HEAL_ComboSkyMap',
                    'Nvisits_as_function_of_Alt_Az_all_bands_HEAL_ComboSkyMap',
                    'Median_airmass_all_bands_HEAL_ComboSkyMap',
                    'Max_airmass_all_bands_HEAL_ComboSkyMap',
                    'CoaddM5_r_band_HEAL_ComboSkyMap',
                    'Normalized_Proper_Motion_@_20_5_All_visits_HEAL_ComboSkyMap',
                    'Normalized_Parallax_@_22_4_All_visits_HEAL_ComboSkyMap']

        sky_maps_headers = ['Nvisits all bands',
                            'Nvisits alt/az all bands',
                            'Median airmass all bands',
                            'Max airmass all bands',
                            'CoaddM5 r band',
                            'Normalized Proper Motion at 20.5',
                            'Normalized Parallax at 22.4']

        histograms = ['CoaddM5_r_band_HEAL_ComboHistogram',
                      'Slew_Distance_Histogram_All_visits_ONED_ComboBinnedData',
                      'Zoom_Slew_Distance_Histogram_All_visits_ONED_ComboBinnedData',
                      'Slew_Time_Histogram_All_visits_ONED_ComboBinnedData',
                      'Zoom_Slew_Time_Histogram_All_visits_ONED_ComboBinnedData']

        hist_headers = ['CoaddM5 r band HealPix Histogram',
                        'Slew Distance Histogram',
                        'Zoom Slew Distance Histogram',
                        'Slew Time Histogram',
                        'Zoom Slew Time Histogram ']

        # The following creates the the final markdown file with a table
        # of contents and all of the relative links.
        with open(os.path.join(outDir,"README.md"), "w") as output:
            output.write("# Table of Contents" + "\n")
            for i,(h,t) in enumerate(zip(headers,toc)):
                output.write(str(i+1)+'. '+'['+h.lstrip('# ')+']'+'('+t+')')
                output.write("\n")
            output.write(str(i+2)+'. '+'[Skymap comparisons](#skymap-comparisons)')
            output.write("\n")
            output.write(str(i+3)+'. '+'[Histogram comparisons](#histogram-comparisons)')
            output.write("\n")
            output.write("\n")
            for fname,header in zip(self.md_tables,headers):
                table = os.path.join(outDir,'figures/'+fname)
                output.write(header+"\n")
                with open(table) as f:
                    output.write(f.read())
                output.write("\n")
            output.write("# Skymap comparisons" + "\n")
            for fname,header in zip(sky_maps, sky_maps_headers):
                pdf_file = 'figures/'+compRoot+fname+'.pdf'
                png_file = 'figures/thumb.'+compRoot+fname+'.png'
                output.write('- ['+header+']'+"("+pdf_file+")" )
                output.write("\n")
                output.write('![png]'+"("+png_file+")")
                output.write("\n")
            output.write("# Histogram comparisons" + "\n")
            for fname,header in zip(histograms, hist_headers):
                pdf_file = 'figures/'+compRoot+fname+'.pdf'
                png_file = 'figures/thumb.'+compRoot+fname+'.png'
                output.write('### '+header)
                output.write("\n")
                output.write('![png]'+"("+png_file+")")
                output.write("\n")
