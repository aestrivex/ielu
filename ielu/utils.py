import numpy as np
import os
from traits.api import (HasTraits, Str, Color, List, Instance, Int, Method,
    on_trait_change, Color, Any, Enum, Button, Float, File, Bool, Range,
    Event)
from traitsui.api import (View, Item, HGroup, Handler, CSVListEditor,
    InstanceEditor, Group, OKCancelButtons, TableEditor, ObjectColumn, 
    TextEditor, OKButton, CheckListEditor, OKCancelButtons, Label, Action,
    VSplit, HSplit, VGroup)
from traitsui.message import error as error_dialog

from functools import partial
from mayavi import mlab

class SortingLabelingError(Exception):
    pass

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

def intize( tuple_of_floats ):
    return tuple( 
        map ( int,
        map ( lambda x:x*1e4, 
        map( partial( round, ndigits=4 ), tuple_of_floats ))))

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

def get_subjects_dir(subjects_dir=None, subject=None):
    if subjects_dir is None or subjects_dir=='':
        subjects_dir = os.environ['SUBJECTS_DIR']
    if subject is None or subject=='':
        subject = os.environ['SUBJECT']

    return os.path.join(subjects_dir, subject)

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
    with open(os.devnull, 'w') as nil:
        p = subprocess.call(['which', 'mri_info'], stdout=nil)
    if p!=0:
        print 'Freesurfer is not sourced or not in the subshell path'
        import sys

        print os.environ['PATH']

        sys.exit(1)

def ask_user_for_savefile(title=None):
    #from traitsui.file_dialog import save_file
    from pyface.api import FileDialog, OK
    
    dialog = FileDialog(action='save as')
    if title is not None:
        dialog.title = title
    dialog.open()
    if dialog.return_code != OK:
        return

    return os.path.join( dialog.directory, dialog.filename )

def ask_user_for_loadfile(title=None):
    from pyface.api import FileDialog, OK
    dialog = FileDialog(action='open')
    if title is not None:
        dialog.title = title
    dialog.open()
    if dialog.return_code != OK:
        return

    return os.path.join( dialog.directory, dialog.filename )

def get_default_color_scheme():
    predefined_colors = [(.2,.5,.8), #focal blue
                         (.6,.3,.9), #dark purple
                         (.8,.5,.9), #light purple
                         (1,.2,.5), #hot pink
                         (.7,.7,.9), #lavender
                         (.36,.58,.04), #dark green
                         (.22,.94,.64), #turquoise
                         (1,.6,.2), #orange
                         (.5,.9,.4), #semi-focal green
                         (0,.6,.8), #royal blue
                        ]

    for color in predefined_colors:
        yield color
    while True:
        yield tuple(np.random.random(3))

class AddLabelsWindow(Handler):
    model = Any
    #clumsily old a reference to the model object

    annotation = Str
    label = File

    add_annot_button = Button('Add annotation')
    add_label_button = Button('Add label file')

    annot_borders = Bool
    annot_opacity = Range(0., 1., 1.)
    annot_hemi = Enum('both','lh','rh')
    label_borders = Bool
    label_opacity = Range(0., 1., 1.)
    label_color = Color('blue')

    remove_labels_action = Action(name='Remove all labels', 
        action='do_remove')

    def _add_annot_button_fired(self):
        self.model.add_annotation(self.annotation, border=self.annot_borders,
            hemi=self.annot_hemi, opacity=self.annot_opacity)
         
    def _add_label_button_fired(self):
        self.model.add_label(self.label, border=self.label_borders,
            opacity=self.label_opacity, color=self.label_color)

    def do_remove(self, info):
        self.model.remove_labels()

    traits_view = View(
        HSplit(
        VGroup(
            Item('annotation'),
            Item('annot_borders', label='show border only'),
            Item('annot_opacity', label='opacity'),
            Item('annot_hemi', label='hemi'),
            Item('add_annot_button', show_label=False),
        ),
        VGroup(
            Item('label'),
            Item('label_borders', label='show_border_only'),
            Item('label_opacity', label='opacity'),
            Item('label_color', label='color'),
            Item('add_label_button', show_label=False),
        ),
        ),
        buttons=[remove_labels_action, OKButton],
        kind='livemodal',
        title='Dial 1-800-COLLECT and save a buck or two',
    )

#class RegistrationAdjustmentWindow(Handler):
#    model = Any
#    #we clumsily hold a reference to the model only to fire its events
#
#    cur_grid = Str
#
#    x_d = Button('-x')
#    x_i = Button('+x')
#    xval = Float(0.)
#    y_d = Button('-y')
#    y_i = Button('+y')
#    yval = Float(0.)
#    z_d = Button('-z')
#    z_i = Button('+z')
#    zval = Float(0.)
#
#    pitch_d = Button('pitch-')
#    pitch_i = Button('pitch+')
#    pitchval = Float(0.)
#    roll_d = Button('roll-')
#    roll_i = Button('roll+')
#    rollval = Float(0.)
#    yaw_d = Button('yaw-')
#    yaw_i = Button('yaw+')
#    yawval = Float(0.)
#
#    reset_button = Button('Reset')
#
#    cos5 = Float(np.cos(np.pi/36))
#    sin5 = Float(np.sin(np.pi/36))
#
#    traits_view = View(
#        Group(
#        HGroup(
#            Item('x_d'),
#            Item('xval'),
#            Item('x_i'),
#        show_labels=False),
#        HGroup(
#            Item('y_d'),
#            Item('yval'),
#            Item('y_i'),
#        show_labels=False),
#        HGroup(
#            Item('z_d'),
#            Item('zval'),
#            Item('z_i'),
#        show_labels=False),
#        HGroup(
#            Item('pitch_d'),
#            Item('pitchval'),
#            Item('pitch_i'),
#        show_labels=False),
#        HGroup(
#            Item('roll_d'),
#            Item('rollval'),
#            Item('roll_i'),
#        show_labels=False),
#        HGroup(
#            Item('yaw_d'),
#            Item('yawval'),
#            Item('yaw_i'),
#        show_labels=False),
#        Item('reset_button', show_label=False)
#        ), 
#    )
#
#    def _rot_matrix(self, axis, neg=False, deg=2):
#        cval = np.cos( deg*np.pi/180 )
#        sval = -np.sin( deg*np.pi/180 ) if neg else np.sin( deg*np.pi/180 )
#
#        if axis == 'x':
#            mat = np.array(( (1, 0, 0, 0),
#                             (0, cval, -sval, 0),
#                             (0, sval, cval, 0),
#                             (0, 0, 0, 1), ))
#        elif axis == 'y':
#            mat = np.array(( (cval, 0, sval, 0),
#                             (0, 1, 0, 0),
#                             (-sval, 0, cval, 0),
#                             (0, 0, 0, 1), ))
#        elif axis == 'z':
#            mat = np.array(( (cval, -sval, 0, 0),
#                             (sval, cval, 0, 0),
#                             (0, 0, 1, 0),
#                             (0, 0, 0, 1), ))
#        else:
#            return np.eye(4)
#
#        return mat
#
#    def _trans_matrix(self, axis, neg=False, dist=.5):
#        mat = np.eye(4)
#
#        if neg:
#            dist *= -1
#
#        if axis=='x': mat[0,3]=dist
#        elif axis=='y': mat[1,3]=dist
#        elif axis=='z': mat[2,3]=dist
#
#        return mat
#
#    def _x_d_fired(self):
#        self.model.reorient_glyph(self.cur_grid, 
#            self._trans_matrix('x', neg=True))
#    def _x_i_fired(self):
#        self.model.reorient_glyph(self.cur_grid, 
#            self._trans_matrix('x', neg=False))
#    def _y_d_fired(self):
#        self.model.reorient_glyph(self.cur_grid, 
#            self._trans_matrix('y', neg=True))
#    def _y_i_fired(self):
#        self.model.reorient_glyph(self.cur_grid, 
#            self._trans_matrix('y', neg=False))
#    def _z_d_fired(self):
#        self.model.reorient_glyph(self.cur_grid, 
#            self._trans_matrix('z', neg=True))
#    def _z_i_fired(self):
#        self.model.reorient_glyph(self.cur_grid, 
#            self._trans_matrix('z', neg=False))
#
#    def _roll_d_fired(self):
#        self.model.reorient_glyph(self.cur_grid, 
#            self._rot_matrix('x', neg=True) )
#    def _roll_i_fired(self):
#        self.model.reorient_glyph(self.cur_grid, 
#            self._rot_matrix('x', neg=False) )
#    def _pitch_d_fired(self):
#        self.model.reorient_glyph(self.cur_grid, 
#            self._rot_matrix('y', neg=True) )
#    def _pitch_i_fired(self):
#        self.model.reorient_glyph(self.cur_grid, 
#            self._rot_matrix('y', neg=False) )
#    def _yaw_d_fired(self):
#        self.model.reorient_glyph(self.cur_grid, 
#            self._rot_matrix('z', neg=True) )
#    def _yaw_i_fired(self):
#        self.model.reorient_glyph(self.cur_grid, 
#            self._rot_matrix('z', neg=False) )
