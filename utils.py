import numpy as np

from traits.api import HasTraits, Str, Color
from traitsui.api import View, Item, HGroup

from mayavi import mlab

def virtual_points3d(coords, figure=None, scale_factor=None, color=None, 
    name=None):

    c = np.array(coords)
    source = mlab.pipeline.scalar_scatter( c[:,0], c[:,1], c[:,2],
        figure=figure)

    return mlab.pipeline.glyph( source, scale_mode='none', scale_factor=3.0,
        mode='sphere', figure=figure, color=color, name=name)

    #return mlab.points3d( c[:,0], c[:,1], c[:,2],
    #    figure=figure, scale_factor=10*scale_factor, color=color, name=name)

def clear_scene(scene):
    for child in scene.children:
        child.remove() 

class NameHolder(HasTraits):
    name = Str
    traits_view = View()

class GeometryNameHolder(NameHolder):
    geometry = Str
    color = Color
    traits_view = View( 
        HGroup(
            Item('geometry', style='readonly'),
            Item('color', style='readonly'),
        ),
    )
