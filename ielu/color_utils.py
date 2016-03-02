from __future__ import division
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

def map_to_table(cmap,nvals=256):
    '''
    Takes a LinearSegmentedColormap, and returns a table of RGBA values
    spanning that colormap.

    arguments:
    cmap - the LinearSegmentedColormap instance
    nvals- the number of values to span over.  The default is 256.
    '''
    return cmap(xrange(nvals),bytes=True)

def mayavi2vtk_color(mayavi_color):
    '''
    converts a 3-tuple mayavi color with values [0,1] to a 4-tuple RGBA
    VTK color with float values [0.0, 255.0]
    '''
    rgb_col = map(lambda color:round(255*color), mayavi_color)
    rgba_col = (rgb_col[0], rgb_col[1], rgb_col[2], 255.0)
    return rgba_col

def mayavi2traits_color(mayavi_color):
    '''
    converts a 3-tuple mayavi color with values [0,1] to a 3-tuple RGB
    color with integer values [0, 255]

    does not use QColor objects at all currently
    '''
    return tuple(map(lambda color:int(255*color), mayavi_color))

def traits2mayavi_color(traits_color):
    '''
    Queries the backend:
        wx:
    converts a 4-tuple traits color with integer values [0, 255] to a
    3-tuple mayavi color with values [0,1]
        Qt:
    converts a 4-tuple traits color with integer values [0, 255] embedded
    inside a QColor object to a 3-tuple mayavi color with values [0,1]
    '''
    from traits.trait_base import ETSConfig
    _tk = ETSConfig.toolkit
    if _tk == 'wx':
        rgba_col = map(lambda color:color/255, traits_color)
    elif _tk == 'qt4':
        rgba_col = map(lambda color:color/255, traits_color.getRgb())
    rgb_col = (rgba_col[0], rgba_col[1], rgba_col[2])
    return rgb_col 

def set_discrete_lut(mayavi_obj, colors, use_vector_lut=False):
    '''
    sets the lookup table of the mayavi object provided to an N color
    color map with the provided list of colors. Also sets the data range
    trait of the LUT.

    Parameters
    ----------
    mayavi_obj : Instance(mlab.pipeline)
        A mayavi object with a module_manager instance
    colors : List(3-tuple)
        List of RGB colors as 3-tuples with values [0,1].
        The color at index 0 corresponds to scalar value 0, and so on
            up to scalar value N-1
    use_vector_lut : Bool
        If true, uses the vector_lut_manager. Otherwise, uses the
        scalar_lut_manager. Defaults to False.
    '''
    if use_vector_lut:
        lut_mgr=mayavi_obj.module_manager.vector_lut_manager
    else:
        lut_mgr=mayavi_obj.module_manager.scalar_lut_manager

    mayavi_obj.actor.mapper.scalar_visibility = True

    n = len(colors)
    #cmap = LinearSegmentedColormap.from_list('ign', colors)
    cmap = map(mayavi2vtk_color, colors)

    #lut_mgr.lut.table = map_to_table(cmap, nvals=n)
    lut_mgr.lut.table = cmap
    lut_mgr.number_of_colors = n
    lut_mgr.data_range = [0, n-1]

def make_transparent(mayavi_obj, index, use_vector_lut=False):
    '''
    Make transparent the color at specified index in the provided LUT.
    Otherwise expects a well formed LUT.
    '''
    if use_vector_lut:
        lut_mgr=mayavi_obj.module_manager.vector_lut_manager
    else:
        lut_mgr=mayavi_obj.module_manager.scalar_lut_manager

    mayavi_obj.actor.mapper.scalar_visibility = True
    
    table = lut_mgr.lut.table
    table[index] = (0., 0., 0., 0.)
    lut_mgr.lut.table = np.array(table).tolist()

    #why in the hell is this necessary
    #without these lines, the hiding occurs on the first time, but on no
    #subsequent times. it's not a race condition either because the random
    #arithmetic below doesn't result in success.
    n = len(table)
    lut_mgr.number_of_colors = n
    lut_mgr.data_range = [0, n-1]

    #f = 37
    #g = f**16 + 21

def change_single_glyph_color(mayavi_glyph, index, color):
    '''
    changes the scalars trait of a mayavi glyph safely to display the altered
    color of a single node

    Parameters
    ----------

    mayavi_glyph : mayavi.modules.Glyph
        The glyph object
    index : int
        The offset of the node within the glyph
    color : number
        The new value of the scalar to set at this index
    '''
    colors = np.array(mayavi_glyph.mlab_source.dataset.point_data.scalars)
    colors[index] = color
    mayavi_glyph.mlab_source.dataset.point_data.scalars = colors

def set_binary_lut(mayavi_obj, color1, color2, use_vector_lut=False):
    '''
    sets the lookup table to of the mayavi object provided to a 2 color
    color map ranging between color1 and color2. Also sets the appropriate
    attributes of the LUT so that the scalars are immediately displayed

    Parameters
    ----------
    mayavi_obj : Instance(mlab.pipeline)
        A mayavi object with a module_manager instance
    color1 : 3-tuple
        An RGB color represented as a 3-tuple with values [0,1], and
        corresponding to the scalar value 0
    color2 : 3-tuple
        An RGB color represented as a 3-tuple with values [0,1], and
        corresponding to the scalar value 1
    use_vector_lut : Bool
        If true, uses the vector_lut_manager. Otherwise, uses the
        scalar_lut_manager. Defaults to False.
    '''
    mayavi_obj.actor.mapper.scalar_visibility = True

    if use_vector_lut:
        lut_mgr=mayavi_obj.module_manager.vector_lut_manager
    else:
        lut_mgr=mayavi_obj.module_manager.scalar_lut_manager

    lut_mgr.lut.table = map_to_table(
        LinearSegmentedColormap.from_list('ign', [color1, color2]))
        #map(mayavi2vtk_color, [color1, color2])))

    #lut_mgr.use_default_range = False
    lut_mgr.data_range = [0,1]

def set_monochrome_lut(mayavi_obj):
    '''
    sets the attributes of the mayavi object so that scalar mapped
    colors are not displayed
    '''
    mayavi_obj.actor.mapper.scalar_visibility = False
