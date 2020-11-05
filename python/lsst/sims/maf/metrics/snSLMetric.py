import numpy as np
import lsst.sims.maf.metrics as metrics
import healpy as hp
from lsst.sims.maf.stackers import snStacker
import numpy.lib.recfunctions as rf

__all__ = ['SNSLMetric']


class SNSLMetric(metrics.BaseMetric):
    def __init__(self, metricName='SNSLMetric',
                 mjdCol='observationStartMJD', RaCol='fieldRA', DecCol='fieldDec',
                 filterCol='filter', exptimeCol='visitExposureTime',
                 nightCol='night', obsidCol='observationId', nexpCol='numExposures',
                 vistimeCol='visitTime', m5Col='fiveSigmaDepth', season=[-1], night_collapse=False,
                 uniqueBlocks=False, season_gap=80., **kwargs):
        """
        Strongly Lensed SN metric

        The number of is given by:

        N (lensed SNe Ia with well measured time delay) = 45.7 * survey_area /
        (20000 deg^2) * cumulative_season_length / (2.5 years) / (2.15 *
        exp(0.37 * gap_median_all_filter))

        where:
        survey_area: survey area (in deg2)
        cumulative_season_length: cumulative season length (in years)
        gap_median_all_filter: median gap (all filters)

        Parameters
        --------------
        metricName : str, opt
         metric name
         Default : SNCadenceMetric
        mjdCol : str, opt
         mjd column name
         Default : observationStartMJD,
        RaCol : str,opt
         Right Ascension column name
         Default : fieldRa
        DecCol : str,opt
         Declinaison column name
         Default : fieldDec
        filterCol : str,opt
         filter column name
         Default: filter
        exptimeCol : str,opt
         exposure time column name
         Default : visitExposureTime
        nightCol : str,opt
         night column name
         Default : night
        obsidCol : str,opt
         observation id column name
         Default : observationId
        nexpCol : str,opt
         number of exposure column name
         Default : numExposures
        vistimeCol : str,opt
         visit time column name
         Default : visitTime
        season: int (list) or -1, opt
         season to process (default: -1: all seasons)


        """
        self.mjdCol = mjdCol
        self.filterCol = filterCol
        self.RaCol = RaCol
        self.DecCol = DecCol
        self.exptimeCol = exptimeCol
        self.nightCol = nightCol
        self.obsidCol = obsidCol
        self.nexpCol = nexpCol
        self.vistimeCol = vistimeCol
        self.seasonCol = 'season'
        self.m5Col = m5Col
        self.season_gap = season_gap

        cols = [self.nightCol, self.filterCol, self.mjdCol, self.obsidCol,
                self.nexpCol, self.vistimeCol, self.exptimeCol]

        super(SNSLMetric, self).__init__(
            col=cols, metricName=metricName, units='N SL', **kwargs)
        self.badVal = 0
        self.season = season
        self.bands = 'ugrizy'

        self.night_collapse = night_collapse

    def run(self, dataSlice, slicePoint=None):
        """
        Runs the metric for each dataSlice

        Parameters
        ---------------
        dataSlice: simulation data
        slicePoint:  slicePoint(default None)

        Returns
        -----------
        number of SL time delay supernovae

        """
        dataSlice.sort(order=self.mjdCol)
        # get the pixel area
        area = hp.nside2pixarea(slicePoint['nside'], degrees=True)

        if len(dataSlice) == 0:
            return self.badVal

        # Collapse down by night if requested
        if self.night_collapse:
            # Or maybe this should be specific per filter?
            key = np.char.array(dataSlice[self.nightCol].astype(str)) + np.char.array(dataSlice[self.filterCol].astype(str))
            u_key, indx = np.unique(key, return_index=True)
            # Normally we would want to co-add depths, increase the number of exposures, average mjdCol. But none of that gets used later.
            dataSlice = dataSlice[indx]
            # Need to resort I think
            dataSlice.sort(order=self.mjdCol)

        dataSlice, season_id = self.seasonCalc(dataSlice)

        seasons = self.season

        if self.season == [-1]:
            seasons = np.unique(season_id)

        season_lengths = []
        median_gaps = []
        for season in seasons:
            idx = np.where(season_id == season)[0]
            slice_sel = dataSlice[idx]
            if len(slice_sel) < 5:
                continue
            mjds_season = dataSlice[self.mjdCol][idx]
            cadence = mjds_season[1:]-mjds_season[:-1]
            season_lengths.append(mjds_season[-1]-mjds_season[0])
            median_gaps.append(np.median(cadence))

        # get the cumulative season length

        cumul_season_length = np.sum(season_lengths)

        if cumul_season_length == 0:
            return self.badVal
        # get gaps
        gap_median = np.mean(median_gaps)

        # estimate the number of lensed supernovae
        cumul_season = cumul_season_length/(12.*30.)

        N_lensed_SNe_Ia = 45.7 * area / 20000. * cumul_season /\
            2.5 / (2.15 * np.exp(0.37 * gap_median))

        return N_lensed_SNe_Ia
