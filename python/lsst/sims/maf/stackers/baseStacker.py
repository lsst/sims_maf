import inspect
import warnings
import numpy as np
import numpy.lib.recfunctions as rfn

class StackerRegistry(type):
    """
    Meta class for Stackers, to build a registry of stacker classes.
    """
    def __init__(cls, name, bases, dict):
        super(StackerRegistry, cls).__init__(name, bases, dict)
        if not hasattr(cls, 'registry'):
            cls.registry = {}
        modname = inspect.getmodule(cls).__name__ + '.'
        if modname.startswith('lsst.sims.maf.stackers'):
            modname = '' 
        stackername = modname + name
        if stackername in cls.registry:
            raise Exception('Redefining stacker %s! (there are >1 stackers with the same name)' %(stackername))
        if stackername != 'BaseStacker':
            cls.registry[stackername] = cls
    def getClass(cls, stackername):
        return cls.registry[stackername]
    def list(cls, doc=False):
        for stackername in sorted(cls.registry):
            if not doc:
                print stackername
            if doc:
                print '---- ', stackername, ' ----'
                print cls.registry[stackername].__doc__
                stacker = cls.registry[stackername]()                
                print ' Columns added to SimData: ', ','.join(stacker.colsAdded)
                print ' Default columns required: ', ','.join(stacker.colsReq)
                
                        
class BaseStacker(object):
    """Base MAF Stacker: add columns generated at run-time to the simdata array."""
    __metaclass__ = StackerRegistry
    
    def __init__(self):
        """
        Instantiate the stacker.
        This method should be overriden by the user. This serves as an example of
        the variables required by the framework.
        """
        # List of the names of the columns generated by the Stacker.
        self.colsAdded = [None]
        # List of the names of the columns required from the database (to generate the Stacker columns).
        self.colsReq = [None]
        # Optional: provide a list of units for the columns defined in colsAdded.
        self.units = [None]

    def _addStackers(self, simData):
        """
        Add the new Stacker columns to the simData array.
        If columns already present in simData, just allows 'run' method to overwrite.
        Returns simData array with these columns added (so 'run' method can set their values).
        """
        newcolList = [simData]
        for col in self.colsAdded:
            if col not in simData:
                newcol = np.empty(len(simData), dtype=[(col, float)])
                newcolList.append(newcol)
        return rfn.merge_arrays(newcolList, flatten=True, usemask=False)

    def run(self, simData):
        """
        Generate the new stacker columns, given the simdata columns from the database.
        Returns the new simdata structured aray that includes the new stacker columns.
        """
        # Add new columns
        simData=self._addStackers(simData)
        # Populate the data in those columns.
        ## simData['newcol'] = XXXX
        return simData
