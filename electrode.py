import numpy as np
from traits.api import (HasTraits, List, Float, Tuple, Instance, Bool, Str, 
    Int, Either, Property)

class Electrode(HasTraits):
#    ct_coords = List(Float)
#    surf_coords = List(Float)
#    snap_coords = List(Float)
    ct_coords = Tuple
    surf_coords = Tuple
    #snap_coords = Tuple
    snap_coords = Instance(np.ndarray)

    is_interpolation = Bool(False)
    grid_name = Str('unsorted')
    grid_transition_to = Str('')
    
    hemi = Str
    vertno = Int(-1)
    pial_coords = Instance(np.ndarray)

    geom_coords = Either(None, Tuple)

    name = Str

    strrepr = Property
    def _get_strrepr(self):
        return str(self)

    #def __eq__(self, other):
    #    return np.all(self.snap_coords == other)

    def __str__(self):
        return 'Elec: %s %s'%(self.grid_name, self.ct_coords)
    def __repr__(self):
        return self.__str__()

    def astuple(self):
        return nparrayastuple(self.snap_coords)

    def asras(self):
        return tuple(self.surf_coords)

    def asct(self):
        return tuple(self.ct_coords)

def nparrayastuple(nparray):
    nparray = np.array(nparray)
    return (nparray[0], nparray[1], nparray[2])
