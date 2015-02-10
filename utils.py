import numpy as np

from traits.api import (HasTraits, Str, Color, List, Instance, Int, Method,
    on_trait_change, Color, Any, Enum, Button, Float)
from traitsui.api import (View, Item, HGroup, Handler, CSVListEditor,
    InstanceEditor, Group, OKCancelButtons, TableEditor, ObjectColumn, 
    TextEditor, OKButton, CheckListEditor, OKCancelButtons, Label, Action)
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
        print 'Freesurfer is not sourced or not in the subshell path'
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
    #selected_ixes = List(Int)
    #selected_ixes = List
    selected_ixes = Any
    swap_action = Action(name='Swap', action='do_swap')

    #selection_color = Color('yellow')
    previous_sel = Instance(Electrode)
    previous_color = Int

    distinct_prev_sel = Instance(Electrode)

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
                #selected_indices='selected_ixes',
                #selection_mode='rows',
                #selected_rows='selected_ixes',
                #selected_items='selected_ixes',

                #on_select='selection_callback'),
                ),
            show_label=False, height=350, width=450),
        resizable=True, kind='panel', title='assign labels',
        buttons=[swap_action, OKButton])

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

        if self.distinct_prev_sel != self.previous_sel:
            self.distinct_prev_sel = self.previous_sel

        self.previous_sel = self.cur_sel
        self.previous_color = self.model._colors.keys().index(self.cur_grid)

        selection_color = (self.model._colors.keys().index('selection'))

        self.model._new_glyph_color = selection_color
        self.model._single_glyph_to_recolor = self.cur_sel.asct()
        self.model._update_single_glyph_event = True

    def do_swap(self, info):
        #if not len(self.selected_ixes) == 2:
        #    return
        if self.distinct_prev_sel == self.cur_sel:
            return
        elif None in (self.distinct_prev_sel, self.cur_sel):
            return

        #i,j = self.selected_ixes
        #e1 = self.electrodes[i]
        #e2 = self.electrodes[j]
        e1 = self.cur_sel
        e2 = self.distinct_prev_sel

        geom_swap = e1.geom_coords
        name_swap = e1.name

        e1.geom_coords = e2.geom_coords
        e1.name = e2.name

        e2.geom_coords = geom_swap
        e2.name = name_swap

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
                                values=['','corner 1', 'corner 2', 
                                    'corner 3']),
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

        try:

            #figure out c1, c2, c3
            c1,c2,c3 = 3*(None,)
            for e in self.electrodes:
                if len(e.corner) == 0:
                    continue
                elif len(e.corner) > 1:
                    error_dialog('Too many corners specified for single'
                        'electrode')
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
            if cur_geom=='user-defined' and self.naming_convention != 'line':
                from color_utils import mayavi2traits_color
                nameholder = GeometryNameHolder(
                    geometry=cur_geom,
                    color=mayavi2traits_color(
                        self.model._colors[self.cur_grid]))
                geomgetterwindow = GeomGetterWindow(holder=nameholder)

                if geomgetterwindow.edit_traits().result:
                    cur_geom = geomgetterwindow.geometry
                else:
                    error_dialog("User did not specify any geometry")
                    return

            import pipeline as pipe
            if self.naming_convention == 'line':
                pipe.fit_grid_to_line(self.electrodes, c1.asct(), c2.asct(),
                    c3.asct(), cur_geom, delta=self.model.delta,
                    rho_loose=self.model.rho_loose)
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
                        #index = y*np.max(cur_geom) + x + 1
                        index = x*np.min(cur_geom) + y + 1
                    else:
                        #index = x*np.min(cur_geom) + y + 1
                        index = y*np.max(cur_geom) + x + 1
                    
                    elec.name = '%s%i'%(self.name_stem, index)
        except Exception as e:
            print str(e)
            return

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

class RegistrationAdjustmentWindow(Handler):
    model = Any
    #we clumsily hold a reference to the model only to fire its events

    cur_grid = Str

    x_d = Button('-x')
    x_i = Button('+x')
    xval = Float(0.)
    y_d = Button('-y')
    y_i = Button('+y')
    yval = Float(0.)
    z_d = Button('-z')
    z_i = Button('+z')
    zval = Float(0.)

    pitch_d = Button('pitch-')
    pitch_i = Button('pitch+')
    pitchval = Float(0.)
    roll_d = Button('roll-')
    roll_i = Button('roll+')
    rollval = Float(0.)
    yaw_d = Button('yaw-')
    yaw_i = Button('yaw+')
    yawval = Float(0.)

    reset_button = Button('Reset')

    cos5 = Float(np.cos(np.pi/36))
    sin5 = Float(np.sin(np.pi/36))

    traits_view = View(
        Group(
        HGroup(
            Item('x_d'),
            Item('xval'),
            Item('x_i'),
        show_labels=False),
        HGroup(
            Item('y_d'),
            Item('yval'),
            Item('y_i'),
        show_labels=False),
        HGroup(
            Item('z_d'),
            Item('zval'),
            Item('z_i'),
        show_labels=False),
        HGroup(
            Item('pitch_d'),
            Item('pitchval'),
            Item('pitch_i'),
        show_labels=False),
        HGroup(
            Item('roll_d'),
            Item('rollval'),
            Item('roll_i'),
        show_labels=False),
        HGroup(
            Item('yaw_d'),
            Item('yawval'),
            Item('yaw_i'),
        show_labels=False),
        Item('reset_button', show_label=False)
        ), 
    )

    def _rot_matrix(self, axis, neg=False, deg=2):
        cval = np.cos( deg*np.pi/180 )
        sval = -np.sin( deg*np.pi/180 ) if neg else np.sin( deg*np.pi/180 )

        if axis == 'x':
            mat = np.array(( (1, 0, 0, 0),
                             (0, cval, -sval, 0),
                             (0, sval, cval, 0),
                             (0, 0, 0, 1), ))
        elif axis == 'y':
            mat = np.array(( (cval, 0, sval, 0),
                             (0, 1, 0, 0),
                             (-sval, 0, cval, 0),
                             (0, 0, 0, 1), ))
        elif axis == 'z':
            mat = np.array(( (cval, -sval, 0, 0),
                             (sval, cval, 0, 0),
                             (0, 0, 1, 0),
                             (0, 0, 0, 1), ))
        else:
            return np.eye(4)

        return mat

    def _trans_matrix(self, axis, neg=False, dist=.5):
        mat = np.eye(4)

        if neg:
            dist *= -1

        if axis=='x': mat[0,3]=dist
        elif axis=='y': mat[1,3]=dist
        elif axis=='z': mat[2,3]=dist

        return mat

    def _x_d_fired(self):
        self.model.reorient_glyph(self.cur_grid, 
            self._trans_matrix('x', neg=True))
    def _x_i_fired(self):
        self.model.reorient_glyph(self.cur_grid, 
            self._trans_matrix('x', neg=False))
    def _y_d_fired(self):
        self.model.reorient_glyph(self.cur_grid, 
            self._trans_matrix('y', neg=True))
    def _y_i_fired(self):
        self.model.reorient_glyph(self.cur_grid, 
            self._trans_matrix('y', neg=False))
    def _z_d_fired(self):
        self.model.reorient_glyph(self.cur_grid, 
            self._trans_matrix('z', neg=True))
    def _z_i_fired(self):
        self.model.reorient_glyph(self.cur_grid, 
            self._trans_matrix('z', neg=False))

    def _roll_d_fired(self):
        self.model.reorient_glyph(self.cur_grid, 
            self._rot_matrix('x', neg=True) )
    def _roll_i_fired(self):
        self.model.reorient_glyph(self.cur_grid, 
            self._rot_matrix('x', neg=False) )
    def _pitch_d_fired(self):
        self.model.reorient_glyph(self.cur_grid, 
            self._rot_matrix('y', neg=True) )
    def _pitch_i_fired(self):
        self.model.reorient_glyph(self.cur_grid, 
            self._rot_matrix('y', neg=False) )
    def _yaw_d_fired(self):
        self.model.reorient_glyph(self.cur_grid, 
            self._rot_matrix('z', neg=True) )
    def _yaw_i_fired(self):
        self.model.reorient_glyph(self.cur_grid, 
            self._rot_matrix('z', neg=False) )
