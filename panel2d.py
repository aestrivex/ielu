from __future__ import division

import numpy as np

from traits.api import (HasTraits, List, Instance, Any, Enum, Tuple)
from traitsui.api import (View, Item, HGroup, VGroup, Group, NullEditor)

from enable.component_editor import ComponentEditor
from chaco.api import Plot, ArrayPlotData
from chaco.api import bone as bone_cmap
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

    cursor = Tuple # 3-tuple

    null = Any # None

    traits_view = View(
        Group(
        HGroup(
            Item(name='xy_plane', editor=ComponentEditor(),
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

    def load_img(self, img):
        imgd = nib.load(img).get_data()

        self.images.append(imgd)
        self.current_image = imgd

        self.cursor = x,y,z = tuple(np.array(imgd.shape) // 2)

        self.xy_plane = Plot(ArrayPlotData(imagedata=imgd[:,:,z]))
        self.xz_plane = Plot(ArrayPlotData(imagedata=imgd[:,y,:]))
        self.yz_plane = Plot(ArrayPlotData(imagedata=imgd[x,:,:].T))

        self.xy_plane.img_plot('imagedata',name='',colormap=bone_cmap)
        self.xz_plane.img_plot('imagedata',name='',colormap=bone_cmap)
        self.yz_plane.img_plot('imagedata',name='',colormap=bone_cmap)
        
        self.xy_plane.tools.append(Click2DPanelTool(self, 'xy'))
        self.xz_plane.tools.append(Click2DPanelTool(self, 'xz'))
        self.yz_plane.tools.append(Click2DPanelTool(self, 'yz'))

    def move_cursor(self, x, y, z):

        self.cursor = x,y,z 

        print 'CURSING %s'%str(self.cursor)

        self.xy_plane.data.set_data('imagedata', self.current_image[:,:,z])
        self.xz_plane.data.set_data('imagedata', self.current_image[:,y,:])
        self.yz_plane.data.set_data('imagedata', self.current_image[x,:,:].T)
