# The base class for all spatial slicers.
# Slicers are 'data slicers' at heart; spatial slicers slice data by RA/Dec and
#  return the relevant indices in the simData to the metric.
# The primary things added here are the methods to slice the data (for any spatial slicer)
#  as this uses a KD-tree built on spatial (RA/Dec type) indexes.

import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.ticker import FuncFormatter
from functools import wraps
import warnings
from lsst.sims.maf.utils import optimalBins, percentileClipping
from scipy.spatial import cKDTree as kdtree
from lsst.obs.lsstSim import LsstSimMapper
from lsst.sims.coordUtils import CameraCoords, AstrometryBase
from lsst.sims.catalogs.generation.db.ObservationMetaData import ObservationMetaData
from .baseSlicer import BaseSlicer

class BaseSpatialSlicer(BaseSlicer):
    """Base slicer object, with added slicing functions for spatial slicer."""
    def __init__(self, verbose=True, spatialkey1='fieldRA', spatialkey2='fieldDec',
                 badval=-666, leafsize=100, radius=1.75, plotFuncs='all', useCamera=False,
                 rotSkyPosColName='rotSkyPos', mjdColName='expMJD'):
        """Instantiate the base spatial slicer object.
        spatialkey1 = ra, spatialkey2 = dec, typically.
        'leafsize' is the number of RA/Dec pointings in each leaf node of KDtree
        'radius' (in degrees) is distance at which matches between
        the simData KDtree
        and slicePoint RA/Dec values will be produced
        plotFuncs = plotting methods to run. default 'all' runs all methods that start
        with 'plot'.
        useCamera = boolean. False means all observations that fall in the radius are assumed to be observed
        True means the observations are checked to make sure they fall on a chip."""

        super(BaseSpatialSlicer, self).__init__(verbose=verbose, badval=badval,
                                                plotFuncs=plotFuncs)
        self.spatialkey1 = spatialkey1
        self.spatialkey2 = spatialkey2
        self.rotSkyPosColName = rotSkyPosColName
        self.mjdColName = mjdColName
        self.columnsNeeded = [spatialkey1, spatialkey2]
        self.useCamera = useCamera
        if useCamera:
            self.columnsNeeded.append(rotSkyPosColName)
            self.columnsNeeded.append(mjdColName)
        self.slicer_init={'spatialkey1':spatialkey1, 'spatialkey2':spatialkey2,
                          'radius':radius, 'badval':badval, 'plotFuncs':plotFuncs,
                          'useCamera':useCamera}
        self.radius = radius
        self.leafsize = leafsize
        self.useCamera = useCamera
        # RA and Dec are required slicePoint info for any spatial slicer.
        self.slicePoints['sid'] = None
        self.slicePoints['ra'] = None
        self.slicePoints['dec'] = None
        self.nslice = None


    def setupSlicer(self, simData, maps=None):
        """Use simData[self.spatialkey1] and simData[self.spatialkey2]
        (in radians) to set up KDTree.

        maps = list of map objects that will run to build up slicePoint"""
        if maps is None:
            maps = []
        else:
            if self.cacheSize != 0:
                warnings.warn('Warning:  Loading maps but cache on. Should probably set useCache=False in slicer.')
        for skyMap in maps:
            self.slicePoints = skyMap.run(self.slicePoints)

        self._setRad(self.radius)
        if self.useCamera:
            self._setupLSSTCamera()
            self._presliceFootprint(simData)
        else:
            self._buildTree(simData[self.spatialkey1], simData[self.spatialkey2], self.leafsize)


        @wraps(self._sliceSimData)

        def _sliceSimData(islice):
            """Return indexes for relevant opsim data at slicepoint
            (slicepoint=spatialkey1/spatialkey2 value .. usually ra/dec)."""

            if self.useCamera:
                indices = self.sliceLookup[islice]
            else:
                sx, sy, sz = self._treexyz(self.slicePoints['ra'][islice], self.slicePoints['dec'][islice])
                # Query against tree.
                indices = self.opsimtree.query_ball_point((sx, sy, sz), self.rad)

            # Build dict for slicePoint info
            slicePoint={}
            for key in self.slicePoints.keys():
                # If we have used the _presliceFootprint to
                if np.size(self.slicePoints[key]) > 1:
                    slicePoint[key] = self.slicePoints[key][islice]
                else:
                    slicePoint[key] = self.slicePoints[key]
            return {'idxs':indices, 'slicePoint':slicePoint}
        setattr(self, '_sliceSimData', _sliceSimData)

    def _setupLSSTCamera(self):
        """If we want to include the camera chip gaps, etc"""

        mapper = LsstSimMapper()
        self.camera = mapper.camera
        self.myCamCoords = CameraCoords()
        self.epoch = 2000.0
        self.obs_metadata = ObservationMetaData(m5=0.)

    def _presliceFootprint(self, simData):
        """Loop over each pointing and find which sky points are observed """
        # Now to make a list of lists for looking up the relevant observations at each slicepoint
        self.sliceLookup = [[] for dummy in xrange(self.nslice)]
        # Make a kdtree for the _slicepoints_
        self._buildTree(self.slicePoints['ra'], self.slicePoints['dec'], leafsize=self.leafsize)

        astrometryObject = AstrometryBase()
        # Loop over each unique pointing position
        for ind,ra,dec,rotSkyPos,mjd in zip(np.arange(simData.size), simData[self.spatialkey1],
                                            simData[self.spatialkey2],
                                            simData[self.rotSkyPosColName], simData[self.mjdColName]):
            dx,dy,dz = self._treexyz(ra,dec)
            # Find healpixels inside the FoV
            hpIndices = np.array(self.opsimtree.query_ball_point((dx, dy, dz), self.rad))
            if hpIndices.size > 0:
                self.obs_metadata.unrefractedRA = ra
                self.obs_metadata.unrefractedDec = dec
                self.obs_metadata.rotSkyPos = rotSkyPos
                self.obs_metadata.mjd = mjd
                # Correct ra,dec for
                raCorr, decCorr = astrometryObject.correctCoordinates(self.slicePoints['ra'][hpIndices],
                                                                      self.slicePoints['dec'][hpIndices],
                                                                      obs_metadata=self.obs_metadata,
                                                                      epoch=self.epoch)
                chipNames = self.myCamCoords.findChipName(ra=raCorr,dec=decCorr,
                                                         epoch=self.epoch,
                                                         camera=self.camera, obs_metadata=self.obs_metadata)
                # Find the healpixels that fell on a chip for this pointing
                hpOnChip = hpIndices[np.where(chipNames != [None])[0]]
                for i in hpOnChip:
                    self.sliceLookup[i].append(ind)

        if self.verbose:
            "Created lookup table after checking for chip gaps."

    def _treexyz(self, ra, dec):
        """Calculate x/y/z values for ra/dec points, ra/dec in radians."""
        # Note ra/dec can be arrays.
        x = np.cos(dec) * np.cos(ra)
        y = np.cos(dec) * np.sin(ra)
        z = np.sin(dec)
        return x, y, z

    def _buildTree(self, simDataRa, simDataDec,
                  leafsize=100):
        """Build KD tree on simDataRA/Dec and set radius (via setRad) for matching.

        simDataRA, simDataDec = RA and Dec values (in radians).
        leafsize = the number of Ra/Dec pointings in each leaf node."""
        if np.any(np.abs(simDataRa) > np.pi*2.0) or np.any(np.abs(simDataDec) > np.pi*2.0):
            raise ValueError('Expecting RA and Dec values to be in radians.')
        x, y, z = self._treexyz(simDataRa, simDataDec)
        data = zip(x,y,z)
        if np.size(data) > 0:
            self.opsimtree = kdtree(data, leafsize=leafsize)
        else:
            raise ValueError('SimDataRA and Dec should have length greater than 0.')

    def _setRad(self, radius=1.75):
        """Set radius (in degrees) for kdtree search.

        kdtree queries will return pointings within rad."""
        x0, y0, z0 = (1, 0, 0)
        x1, y1, z1 = self._treexyz(np.radians(radius), 0)
        self.rad = np.sqrt((x1-x0)**2+(y1-y0)**2+(z1-z0)**2)

    def sliceSimDataMultiSlicepoint(self, islices):
        """Return indexes for opsim data at multiple slicepoints (rarely used). """
        binx, biny, binz=self._treexyz(self.slicePoints['ra'][islices], self.slicePoints['dec'][islices])
        indices = self.opsimtree.query_ball_point(zip(binx, biny, binz), self.rad)
        return indices


    ## Plot histogram (base spatial slicer method).
    def plotHistogram(self, metricValueIn, title=None, xlabel=None, units=None, ylabel=None,
                      fignum=None, label=None, addLegend=False, legendloc='upper left',
                      bins=None, binsize=None, cumulative=False, anticumulative=False,
                      xMin=None, xMax=None, yMin=None, yMax=None,
                      logScale='auto', flipXaxis=False,
                      scale=1.0, yaxisformat='%.3f', color='b',
                      zp=None, normVal=None, percentileClip=None, **kwargs):
        """Plot a histogram of metricValue, labelled by metricLabel.

        title = the title for the plot (default None)
        fignum = the figure number to use (default None - will generate new figure)
        label = the label to use in the figure legend (default None)
        addLegend = flag for whether or not to add a legend (default False)
        legendloc = location for legend (default 'upper left')
        bins = bins for histogram (numpy array or # of bins)
        binsize = size of bins to use.  Will override "bins" if both are set.
        (default None, uses Freedman-Diaconis rule to set binsize)
        cumulative = make histogram cumulative (default False) (<0 value makes cumulative the 'less than' way).
        xMin/Max = histogram range (default None, set by matplotlib hist)
        yMin/Max = histogram y range
        flipXaxis = flip the x axis (i.e. for magnitudes) (default False)
        scale = scale y axis by 'scale' (i.e. to translate to area)
        zp = zeropoing to subtract off metricVals
        normVal = normalization value to divide metric values by (overrides zp)"""
        if bins is None and binsize is None:
            bins = optimalBins(metricValueIn)
        # Histogram metricValues.
        fig = plt.figure(fignum)
        if not xlabel:
            xlabel = units
        if zp:
            metricValue = metricValueIn.compressed() - zp
        elif normVal:
            metricValue = metricValueIn.compressed()/normVal
        else:
            metricValue = metricValueIn.compressed()
        # Need to only use 'good' values in histogram,
        # but metricValue is masked array (so bad values masked when calculating max/min).
        if xMin is None and xMax is None:
            if percentileClip:
                xMin, xMax = percentileClipping(metricValue, percentile=percentileClip)
                histRange = [xMin, xMax]
            else:
                histRange = None
        else:
            histRange=[xMin, xMax]
        if yMin is not None or yMax is not None:
            plt.ylim([yMin,yMax])
        # See if should use log scale.
        if logScale == 'auto':
            if np.min(histRange) > 0:
                if (np.log10(np.max(histRange)-np.log10(np.min(histRange))) > 3 ):
                    logScale = True
                else:
                    logScale = False
            else:
                logScale = False
        # If we want all the plots to have the same binsize
        if binsize is not None:
            if histRange is None:
                bins = np.arange(metricValue.min(), metricValue.max()+binsize, binsize)
            else:
                bins = np.arange(histRange[0], histRange[1]+binsize, binsize)
        # Plot histograms.
        # Add a test to see if data falls within histogram range.. because otherwise histogram will fail.
        if histRange is not None:
            if (histRange[0] is None) and (histRange[1] is not None):
                condition = (metricValue <= histRange[1])
            elif (histRange[1] is None) and (histRange[0] is not None):
                condition = (metricValue >= histRange[0])
            else:
                condition = ((metricValue >= histRange[0]) & (metricValue <= histRange[1]))
            plotValue = metricValue[condition]
        else:
            plotValue = metricValue

        # If there is only one value to histogram, need to set histRange
        rangePad = 20.
        if (np.unique(plotValue).size == 1) & (histRange is None):
            warnings.warn('Only one metric value, making a guess at a good histogram range.')
            histRange = [plotValue.max()-rangePad, plotValue.max()+rangePad]
            if (plotValue.min() >= 0) & (histRange[0] < 0):
                histRange[0] = 0.
            bins=np.arange(histRange[0], histRange[1], binsize)

        if plotValue.size == 0:
            if histRange is None:
                warnings.warn('Warning! Could not plot metric data: histRange is None and all data masked' )
            else:
                warnings.warn('Warning! Could not plot metric data: none fall within histRange %.2f %.2f' %
                              (histRange[0], histRange[1]))
            return None
        else:
            n, b, p = plt.hist(plotValue, bins=bins, histtype='step', log=logScale,
                               cumulative=cumulative, range=histRange, label=label, color=color)
        # Option to use 'scale' to turn y axis into area or other value.
        def mjrFormatter(y,  pos):
            return yaxisformat % (y * scale)
        ax = plt.gca()
        ax.yaxis.set_major_formatter(FuncFormatter(mjrFormatter))
        # There is a bug in histype='step' that can screw up the ylim.  Comes up when running allSlicer.Cfg.py
        if plt.axis()[2] == max(n):
            plt.ylim([n.min(),n.max()])
        if xlabel is not None:
            plt.xlabel(xlabel)
        if ylabel is not None:
            plt.ylabel(ylabel)
        if flipXaxis:
            # Might be useful for magnitude scales.
            x0, x1 = plt.xlim()
            plt.xlim(x1, x0)
        if addLegend:
            plt.legend(fancybox=True, prop={'size':'smaller'}, loc=legendloc)
        if title!=None:
            plt.title(title)
        # Return figure number (so we can reuse this if desired).
        return fig.number

    ### Generate sky map (base spatial slicer methods, using ellipses for each RA/Dec value)
    ### a healpix slicer will not have self.ra / self.dec functions, but plotSkyMap is overriden.

    def _plot_tissot_ellipse(self, lon, lat, radius, ax=None, **kwargs):
        """Plot Tissot Ellipse/Tissot Indicatrix

        Parameters
        ----------
        lon : float or array_like
        longitude-like of ellipse centers (radians)
        lat : float or array_like
        latitude-like of ellipse centers (radians)
        radius : float or array_like
        radius of ellipses (radians)
        ax : Axes object (optional)
        matplotlib axes instance on which to draw ellipses.

        Other Parameters
        ----------------
        other keyword arguments will be passed to matplotlib.patches.Ellipse.

        # The code in this method adapted from astroML, which is BSD-licensed.
        # See http://github.com/astroML/astroML for details.
        """
        # Code adapted from astroML, which is BSD-licensed.
        # See http://github.com/astroML/astroML for details.
        ellipses = []
        if ax is None:
            ax = plt.gca()
        for l, b, diam in np.broadcast(lon, lat, radius*2.0):
            el = Ellipse((l, b), diam / np.cos(b), diam)
            ellipses.append(el)
        return ellipses

    def _plot_ecliptic(self, raCen=0, ax=None):
        """Plot a red line at location of ecliptic"""
        if ax is None:
            ax = plt.gca()
        ecinc = 23.439291*(np.pi/180.0)
        ra_ec = np.arange(0, np.pi*2., (np.pi*2./360))
        dec_ec = np.sin(ra_ec) * ecinc
        lon = -(ra_ec - raCen - np.pi) % (np.pi*2) - np.pi
        ax.plot(lon, dec_ec, 'r.', markersize=1.8)

    def plotSkyMap(self, metricValueIn, title=None, xlabel=None, units=None,
                   projection='aitoff', radius=1.75/180.*np.pi,
                   logScale='auto', cbar=True, cbarFormat=None,
                   cmap=cm.jet, alpha=1, fignum=None,
                   zp=None, normVal=None,
                   colorMin=None, colorMax=None, percentileClip=None, cbar_edge=True,
                   label=None, plotMask=False, metricIsColor=False, raCen=0.0, **kwargs):
        """
        Plot the sky map of metricValue.
        """
        from matplotlib.collections import PatchCollection
        from matplotlib import colors
        if fignum is None:
            fig = plt.figure()
        else:
            fig = plt.figure(fignum)
        metricValue = metricValueIn
        if zp or normVal:
            if zp:
                metricValue = metricValue - zp
            if normVal:
                metricValue = metricValue/normVal
        # other projections available include
        # ['aitoff', 'hammer', 'lambert', 'mollweide', 'polar', 'rectilinear']
        ax = fig.add_subplot(111, projection=projection)
        # Set up valid datapoints and colormin/max values. 
        if plotMask:
            # Plot all data points.
            mask = np.ones(len(metricValue), dtype='bool')
        else:
            # Only plot points which are not masked. Flip numpy ma mask where 'False' == 'good'.
            mask = ~metricValue.mask
        # Determine color min/max values. metricValue.compressed = non-masked points.
        if percentileClip:
            pcMin, pcMax = percentileClipping(metricValue.compressed(), percentile=percentileClip)
        if colorMin is None:
            if percentileClip:
                colorMin = pcMin
            else:
                colorMin = metricValue.compressed().min()
        if colorMax is None:
            if percentileClip:
                colorMax = pcMax
            else:
                colorMax = metricValue.compressed().max()
                # Avoid colorbars with no range.
                if colorMax == colorMin:
                    colorMax = colorMax+1
                    colorMin = colorMin-1
        # Combine to make clims:
        clims = [colorMin, colorMax]
        # Determine whether or not to use auto-log scale.
        if logScale == 'auto':
            if colorMin > 0:
                if np.log10(colorMax)-np.log10(colorMin) > 3:
                    logScale = True
                else:
                    logScale = False
            else:
                logScale = False
        if logScale:
            # Move min/max values to things that can be marked on the colorbar.
            colorMin = 10**(int(np.log10(colorMin)))
            colorMax = 10**(int(np.log10(colorMax)))
        # Add ellipses at RA/Dec locations
        lon = -(self.slicePoints['ra'][mask] - raCen - np.pi) % (np.pi*2) - np.pi
        ellipses = self._plot_tissot_ellipse(lon, self.slicePoints['dec'][mask], radius, ax=ax)
        if metricIsColor:
            for ellipse, mVal in zip(ellipses, metricValue.data[mask]):
                ellipse.set_alpha(mVal[3])
                ellipse.set_color((mVal[0], mVal[1], mVal[2]))
                ax.add_patch(ellipse)
        else:
            if logScale:
                norml = colors.LogNorm()
                p = PatchCollection(ellipses, cmap=cmap, alpha=alpha, linewidth=0, edgecolor=None,
                                    norm=norml, rasterized=True)
            else:
                p = PatchCollection(ellipses, cmap=cmap, alpha=alpha, linewidth=0, edgecolor=None,
                                    rasterized=True)
            p.set_array(metricValue.data[mask])
            p.set_clim(clims)
            ax.add_collection(p)
            # Add color bar (with optional setting of limits)
            if cbar:
                cb = plt.colorbar(p, aspect=25, extend='both', extendrect=True, orientation='horizontal',
                                format=cbarFormat)
                # If outputing to PDF, this fixes the colorbar white stripes
                if cbar_edge:
                    cb.solids.set_edgecolor("face")
                if xlabel is not None:
                    cb.set_label(xlabel)
                elif units is not None:
                    cb.set_label(units)
        # Add ecliptic
        self._plot_ecliptic(raCen, ax=ax)
        ax.grid(True, zorder=1)
        ax.xaxis.set_ticklabels([])
        # Add label.
        if label is not None:
            plt.figtext(0.75, 0.9, '%s' %label)
        if title is not None:
            plt.text(0.5, 1.09, title, horizontalalignment='center', transform=ax.transAxes)
        return fig.number