import numpy as np

from traits.api import (HasTraits, Str, Color, List, Instance, Int, Method,
    on_trait_change, Color, Any)
from traitsui.api import (View, Item, HGroup, Handler, CSVListEditor,
    InstanceEditor, Group, OKCancelButtons, TableEditor, ObjectColumn, 
    TextEditor, OKButton)

from mayavi import mlab
from electrode import Electrode

def virtual_points3d(coords, figure=None, scale_factor=None, color=None, 
    name=None):

    c = np.array(coords)
    source = mlab.pipeline.scalar_scatter( c[:,0], c[:,1], c[:,2],
        figure=figure)

    return mlab.pipeline.glyph( source, scale_mode='none', 
        scale_factor=scale_factor,
        mode='sphere', figure=figure, color=color, name=name)

    #return mlab.points3d( c[:,0], c[:,1], c[:,2],
    #    figure=figure, scale_factor=10*scale_factor, color=color, name=name)

def clear_scene(scene):
    #this bugs out for a reason I haven't figured out yet
    #for child in scene.children:
    #    child.remove() 

    #this causes a picker bug with or without gc
    #scene.children = []
    #import gc
    #gc.collect()

    #while scene.children: 
    #    for child in scene.children:
    #        child.remove()

    #Now we handle this by building a new viewport-like object
    #which CANNOT be done inside the viewport itself
    pass

    #mlab.clf(figure=scene)

def _count():
    i=0
    while True:
        yield i
        i+=1
_counter=_count()
def gensym():
    global _counter
    return _counter.next()

def crash_if_freesurfer_is_not_sourced():
    import os, subprocess
    with open(os.devnull) as nil:
        p = subprocess.call(['which', 'mri_info'], stdout=nil, stderr=nil)
    if p!=0:
        print 'Freesurfer is not sourced'
        import sys
        sys.exit(1)

class NameHolder(HasTraits):
    name = Str
    traits_view = View()

    def __str__(self):
        return 'Grid: %s'%self.name

class GeometryNameHolder(NameHolder):
    geometry = Str
    color = Color
    traits_view = View( 
        HGroup(
            Item('geometry', style='readonly'),
            Item('color', style='readonly'),
        ),
    )

    def __str__(self):
        return 'Grid: %s, col:%s, geom:%s'%(self.name,self.color,
            self.geometry)
    def __repr__(self):
        return '%s\n'%str(self)

class GeomGetterWindow(Handler):
    #has to do the handler thing
    #for now proof of concept and notimplementederror
    geometry = List(Int)
    holder = Instance(NameHolder)
    
    traits_view = View(
        #Item('grid_representation', editor=InstanceEditor(), style='custom'),
        Item('holder', editor=InstanceEditor(), style='custom', 
            show_label=False),
        Item('geometry', editor=CSVListEditor(), label='list geometry'),
        title='Specify geometry',
        kind='livemodal',
        buttons=OKCancelButtons,
    )

class ManualLabelAssignmentWindow(Handler):
    #model = Instance(ElectrodePositionsModel)
    model = Any
    #we clumsily hold a reference to the model only to fire its events

    cur_grid = Str

    electrodes = List(Electrode)  
    cur_sel = Instance(Electrode)
    selection_callback = Method

    #selection_color = Color('yellow')
    previous_sel = Instance(Electrode)
    previous_color = Int

    traits_view = View(
        Item('electrodes',
            editor=TableEditor( columns = 
                [ObjectColumn(label='electrode',
                              editor=TextEditor(),
                              style='readonly',
                              editable=False,
                              name='strrepr'),

#                 ObjectColumn(label='geometry',
#                              editor=TextEditor(),
#                              style='readonly',
#                              editable=False,
#                              name='geom_coords'),
                              
                 ObjectColumn(label='channel name',
                              editor=TextEditor(),
                              name='name'),
                 ],
                selected='cur_sel',
                #on_select='selection_callback'),
                ),
            show_label=False, height=350, width=333),
        resizable=True, kind='panel', title='assign labels',
        buttons=[OKButton])

    def closed(self, is_ok, info):
        if self.previous_sel is not None:
            self.model._new_glyph_color = self.previous_color
            self.model._single_glyph_to_recolor = self.previous_sel.asct()
            self.model._update_single_glyph_event = True

    @on_trait_change('cur_sel')
    def selection_callback(self):
        #from color_utils import traits2mayavi_color

        if self.cur_sel is None:
            return

        if self.previous_sel is not None:
            self.model._new_glyph_color = self.previous_color
            self.model._single_glyph_to_recolor = self.previous_sel.asct()
            self.model._update_single_glyph_event = True

        self.previous_sel = self.cur_sel
        self.previous_color = self.model._colors.keys().index(self.cur_grid)

        selection_color = (
            self.model._colors.keys().index('selection'))

        self.model._new_glyph_color = selection_color
        self.model._single_glyph_to_recolor = self.cur_sel.asct()
        self.model._update_single_glyph_event = True
