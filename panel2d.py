from __future__ import division

import numpy as np

from traits.api import (HasTraits, List, Instance, Any, Enum, Tuple)
from traitsui.api import (View, Item, HGroup, VGroup, Group, NullEditor)

from enable.component_editor import ComponentEditor
from chaco.api import Plot, ArrayPlotData
from chaco.api import OverlayPlotContainer
from chaco.api import bone as bone_cmap
from chaco.api import RdBu as rdbu_cmap
from chaco.api import reverse as reverse_cmap
from chaco.tools.api import SelectTool

import nibabel as nib

class Click2DPanelTool(SelectTool):
    
    panel2d = Any #Instance(TwoDimensionalPanel)
    panel_id = Enum('xy','xz','yz')

    def __init__(self, panel2d, panel_id):
        self.panel2d = panel2d
        self.panel_id = panel_id

    def normal_left_down(self, event):
        x,y,z = self.panel2d.cursor            

        #if the panel is not in the image (e.g. a click on the axis), ignore and pass

        #remember event.x and event.y are in space of pixels
        print 'CLICKETY %s %i %i'%(self.panel_id,event.x, event.y)

        if self.panel_id == 'xy':
            mx, my = self.panel2d.xy_plane.map_data((event.x, event.y))
            if x == mx and y == my:
                return
            self.panel2d.move_cursor(mx, my, z)

        elif self.panel_id == 'xz':
            mx, my = self.panel2d.xz_plane.map_data((event.x, event.y))
            if x == mx and z == my:
                return
            self.panel2d.move_cursor(mx, y, my)

        elif self.panel_id == 'yz':
            mx, my = self.panel2d.xy_plane.map_data((event.y, event.x))
            if z == mx and y == my:
                return
            self.panel2d.move_cursor(x, my, mx)

        else:
            raise NotImplementedError('FailFish')

class TwoDimensionalPanel(HasTraits):
    images = List

    current_image = Any # np.ndarray XxYxZ

    xy_plane = Instance(Plot)
    xz_plane = Instance(Plot)
    yz_plane = Instance(Plot)

    #later we will rename cursor to "coord"

    native_cursor = Tuple # 3-tuple
    cursor = Tuple # 3-tuple
    ras_cursor = Tuple # 3-tuple

    null = Any # None

    traits_view = View(
        Group(
        HGroup(
            Item(name='xy_plane', editor=ComponentEditor(),
            #Item(name='xy_container', editor=ComponentEditor(),
                height=400, width=400, show_label=False, resizable=True),
            Item(name='yz_plane', editor=ComponentEditor(),
                height=400, width=400, show_label=False, resizable=True),
        ),
        HGroup(
            Item(name='xz_plane', editor=ComponentEditor(),
                height=400, width=400, show_label=False, resizable=True),

            Item(name='null', editor=NullEditor(),
                height=400, width=400, show_label=False, resizable=True),
        ),
        ),
        title='Contact 867-5309 for blobfish sales',
    )

    def map_cursor(self, affine, cursor):
        import geometry as geo
        x,y,z = cursor
        map, = geo.apply_affine([cursor], np.linalg.inv(affine))
        return map

    def cut_data(self, data, mcursor):
        xm,ym,zm = [int(np.round(c)) for c in mcursor]
        #xm, ym, zm = mcursor
        yz_cut = np.rot90(data[xm,:,:])
        xz_cut = np.rot90(data[:,ym,:])
        xy_cut = np.rot90(data[:,:,zm])
        return xy_cut, xz_cut, yz_cut

    def load_img(self, imgf):
        print 'volumzgou'

        #img_like = new_img_like(imedc, imedc.get_data(), imedc.get_affine())
        img = nib.load(imgf)

        imgd = img.get_data()
        print imgd.shape
        #imgd = img_like.get_data()

        self.images.append(imgd)
        self.current_image = imgd

        self.native_cursor = x,y,z = tuple(np.array(imgd.shape) // 2)

        self.cursor = xm,ym,zm = self.map_cursor(img.get_affine(), 
            self.native_cursor)

        xy_cut, xz_cut, yz_cut = self.cut_data(imgd, self.cursor)

        xy_plotdata = ArrayPlotData()
        xy_plotdata.set_data('imagedata', xy_cut)
        #xy_plotdata.set_data('imagedata', imgd[:,:,z])
        xy_plotdata.set_data('cursor_x', np.array((x,)))
        xy_plotdata.set_data('cursor_y', np.array((y,)))

        xz_plotdata = ArrayPlotData()
        xz_plotdata.set_data('imagedata', xz_cut)
        #xz_plotdata.set_data('imagedata', imgd[:,y,:])
        xz_plotdata.set_data('cursor_x', np.array((x,)))
        xz_plotdata.set_data('cursor_z', np.array((z,)))

        yz_plotdata = ArrayPlotData()
        yz_plotdata.set_data('imagedata', yz_cut.T)
        #yz_plotdata.set_data('imagedata', imgd[x,:,:].T)
        yz_plotdata.set_data('cursor_y', np.array((y,)))
        yz_plotdata.set_data('cursor_z', np.array((z,)))

        #self.xy_plane = Plot(ArrayPlotData(imagedata=imgd[:,:,z]))
        #self.xz_plane = Plot(ArrayPlotData(imagedata=imgd[:,y,:]))
        #self.yz_plane = Plot(ArrayPlotData(imagedata=imgd[x,:,:].T))
        self.xy_plane = Plot(xy_plotdata)
        self.xz_plane = Plot(xz_plotdata)
        self.yz_plane = Plot(yz_plotdata)

        self.xy_plane.img_plot('imagedata',name='',colormap=bone_cmap)
        self.xz_plane.img_plot('imagedata',name='',colormap=bone_cmap)
        self.yz_plane.img_plot('imagedata',name='',colormap=bone_cmap)

        self.xy_plane.plot(('cursor_x','cursor_y'), type='scatter', 
            color='red', marker='plus', size=3)
        self.xz_plane.plot(('cursor_x','cursor_z'), type='scatter',
            color='red', marker='plus', size=3)
        self.yz_plane.plot(('cursor_z','cursor_y'), type='scatter',
            color='red', marker='plus', size=3)

        self.xy_plane.tools.append(Click2DPanelTool(self, 'xy'))
        self.xz_plane.tools.append(Click2DPanelTool(self, 'xz'))
        self.yz_plane.tools.append(Click2DPanelTool(self, 'yz'))

        from PyQt4.QtCore import pyqtRemoveInputHook
        import pdb
        pyqtRemoveInputHook()
        pdb.set_trace()

    def move_cursor(self, x, y, z):

        self.cursor = x,y,z 

        print 'CURSING %s'%str(self.cursor)

        self.xy_plane.data.set_data('imagedata', self.current_image[:,:,z])
        self.xz_plane.data.set_data('imagedata', self.current_image[:,y,:])
        self.yz_plane.data.set_data('imagedata', self.current_image[x,:,:].T)

        self.xy_plane.data.set_data('cursor_x', np.array((x,)))
        self.xy_plane.data.set_data('cursor_y', np.array((y,)))
