import numpy as np

from traits.api import (HasTraits, Str, Color, List, Instance, Int, Method,
    on_trait_change, Color, Any, Enum)
from traitsui.api import (View, Item, HGroup, Handler, CSVListEditor,
    InstanceEditor, Group, OKCancelButtons, TableEditor, ObjectColumn, 
    TextEditor, OKButton, CheckListEditor, OKCancelButtons, Label)
from traitsui.message import error as error_dialog

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
        return str(self)

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

                 ObjectColumn(label='geometry',
                              editor=CSVListEditor(),
                              #editor=TextEditor(),
                              #style='readonly',
                              #editable=False,
                              name='geom_coords'),
                              
                 ObjectColumn(label='channel name',
                              editor=TextEditor(),
                              name='name'),
                 ],
                selected='cur_sel',
                #on_select='selection_callback'),
                ),
            show_label=False, height=350, width=400),
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

        selection_color = (self.model._colors.keys().index('selection'))

        self.model._new_glyph_color = selection_color
        self.model._single_glyph_to_recolor = self.cur_sel.asct()
        self.model._update_single_glyph_event = True

class AutomatedAssignmentWindow(Handler):
    model = Any
    #we clumsily hold a reference to the model only to fire its events

    cur_grid = Str
    cur_sel = Instance(Electrode)
    selection_callback = Method
    
    naming_convention = Enum('default', 'line')
    #first_axis = Enum('corner 1/corner 2','corner 1/corner 3')
    first_axis = Enum('standard','reverse (short side first)')
    name_stem = Str
    
    electrodes = List(Instance(Electrode))
    c1, c2, c3 = 3*(Instance(Electrode),)

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

                 ObjectColumn(label='corner',
                              editor=CheckListEditor(
                                values=['corner 1', 'corner 2', 'corner 3']),
                              #style='custom',
                              style='simple',
                              name='corner',),
                 ],
                selected='cur_sel'),
            show_label=False, height=350, width=400,),
        Label('Note in NxN grid, corner1/corner2 axis is standard'),
        HGroup(
            Item('naming_convention',), 
            Item('first_axis',),
        ),
        Item('name_stem', label='stem'),

        resizable=True, kind='panel', title='indicate corner electrodes',
        buttons=OKCancelButtons)

    def closed(self, info, is_ok):
        #uncolor last selection
        if self.previous_sel is not None:
            self.model._new_glyph_color = self.previous_color
            self.model._single_glyph_to_recolor = self.previous_sel.asct()
            self.model._update_single_glyph_event = True

        #if the user clicked cancel, do no processing
        if not is_ok:
            return

        #figure out c1, c2, c3
        c1,c2,c3 = 3*(None,)
        for e in self.electrodes:
            if len(e.corner) == 0:
                continue
            elif len(e.corner) > 1:
                error_dialog('Too many corners specified for single electrode')
                return
    
            elif 'corner 1' in e.corner:
                c1 = e
            elif 'corner 2' in e.corner:
                c2 = e
            elif 'corner 3' in e.corner:
                c3 = e

        if c1 is None or c2 is None or c3 is None:
            error_dialog('Not all corners were specified')
            return
    
        cur_geom = self.model._grid_geom[self.cur_grid]
        if cur_geom=='user-defined':
            from color_utils import mayavi2traits_color
            nameholder = GeometryNameHolder(
                geometry=cur_geom,
                color=mayavi2traits_color(self.model._colors[self.cur_grid]))
            geomgetterwindow = GeomGetterWindow(holder=nameholder)

        if geomgetterwindow.edit_traits().result:
            cur_geom = geomgetterwindow.geometry
        else:
            error_dialog("User did not specify any geometry")
            return

        import pipeline as pipe
        if self.naming_convention == 'line':
            pipe.fit_grid_to_line(self.electrodes, c1.asct(), c2.asct(),
                c3.asct(), cur_geom)
            #do actual labeling
            for elec in self.model._grids[self.cur_grid]:
                _,y = elec.geom_coords
                index = y+1
                elec.name = '%s%i'%(self.name_stem, index)

        else:
            pipe.fit_grid_to_plane(self.electrodes, c1.asct(), c2.asct(), 
                c3.asct(), cur_geom)

            #do actual labeling
            for elec in self.model._grids[self.cur_grid]:
                x,y = elec.geom_coords
                if self.first_axis=='standard':
                    index = y*np.max(cur_geom) + x + 1
                else:
                    index = x*np.min(cur_geom) + y + 1
                
                elec.name = '%s%i'%(self.name_stem, index)

    @on_trait_change('cur_sel')
    def selection_callback(self):
        if self.cur_sel is None:
            return

        if self.previous_sel is not None:
            self.model._new_glyph_color = self.previous_color
            self.model._single_glyph_to_recolor = self.previous_sel.asct()
            self.model._update_single_glyph_event = True

        self.previous_sel = self.cur_sel
        self.previous_color = self.model._colors.keys().index(self.cur_grid)

        selection_color = (self.model._colors.keys().index('selection'))

        self.model._new_glyph_color = selection_color
        self.model._single_glyph_to_recolor = self.cur_sel.asct()
        self.model._update_single_glyph_event = True

        
