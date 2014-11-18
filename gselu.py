from __future__ import division
import os
import numpy as np
from mayavi.core.ui.api import MayaviScene, SceneEditor, MlabSceneModel
from traits.api import (Bool, Button, cached_property, File, HasTraits,
    Instance, on_trait_change, Str, Property, Directory, Dict, DelegatesTo,
    HasPrivateTraits, Any, List, Enum, Int, Event)
from traitsui.api import (View, Item, Group, OKCancelButtons, ShellEditor,
    HGroup,VGroup, InstanceEditor, TextEditor, ListEditor, CSVListEditor)

from electrode import Electrode
from utils import virtual_points3d, NameHolder, GeometryNameHolder

class ElectrodePositionsModel(HasPrivateTraits):
    ct_scan = File
    t1_scan = File
    subjects_dir = Directory
    subject = Str
    fsdir_writable = Bool
    hemisphere = Enum('rh','lh')

    electrode_geometry = List(List(Int), [[8,8]]) # Gx2 list

    _electrodes = List(Electrode)
    interactive_mode = Instance(NameHolder)
    _grids = Dict # Grid -> List(Electrode)
    _grid_named_objects = Property(depends_on='_grids')

    _sorted_electrodes = Dict # Tuple -> Electrode
    _interpolated_electrodes = Dict # Tuple -> Electrode
    _unsorted_electrodes = Dict # Tuple -> Electrode
    _all_electrodes = Dict # Tuple -> Electrode
        # dictionary from surface coordinates (as hashable) to reused
        # electrode objects

    _points_to_cur_grid = Dict
    _points_to_unsorted = Dict

    _visualize_event = Event

    _colors = Any # OrderedDict(Grid -> Color)
    _grid_geom = Dict # Grid -> Gx2 list

    @cached_property
    def _get__grid_named_objects(self):
        from color_utils import mayavi2traits_color
        grid_names = [NameHolder(name=''), 
            GeometryNameHolder(name='unsorted',
                geometry='n/a',
                #TODO dont totally hardcode this color
                color=mayavi2traits_color((1,0,0)))]

        for key in self._grids.keys():
            grid_names.append(GeometryNameHolder(name=key, 
                geometry=str(self._grid_geom[key]), 
                color=mayavi2traits_color(self._colors[key])))

        return grid_names

    def _interactive_mode_changed(self):
        self._commit_grid_changes()

        self._points_to_cur_grid = {}
        self._points_to_unsorted = {}

    def _commit_grid_changes(self):
        for p in (self._points_to_cur_grid, self._points_to_unsorted):
            for loc in p:
                elec = p[loc]
                
                old = elec.grid_name
                new = elec.grid_transition_to

                elec.grid_name = new
                elec.grid_transition_to = ''
        
                if old != 'unsorted':
                    self._grids[old].remove(elec)
                if new != 'unsorted':
                    self._grids[new].append(elec)
    
    def _run_pipeline(self):
        if self.subjects_dir is None or self.subjects_dir=='':
            self.subjects_dir = os.environ['SUBJECTS_DIR']
        if self.subject is None or self.subject=='':
            self.subject = os.environ['SUBJECT']

        self._electrodes=[]

        import pipeline as pipe
        
        ct_mask = pipe.create_brainmask_in_ctspace(self.ct_scan,
            subjects_dir=self.subjects_dir, subject=self.subject)

        self._electrodes = pipe.identify_electrodes_in_ctspace(
            self.ct_scan, mask=ct_mask) 

        aff = pipe.register_ct_to_mr_using_mutual_information(self.ct_scan,
            subjects_dir=self.subjects_dir, subject=self.subject)

        pipe.create_dural_surface(subjects_dir=self.subjects_dir, 
            subject=self.subject)

        #initial sorting
        #self._grids, self._colors = pipe.classify_electrodes(
        self._colors, self._grid_geom, self._grids = pipe.classify_electrodes(
            self._electrodes, self.electrode_geometry)

        # add grid labels to electrodes
        for key in self._grids:
            for elec in self._grids[key]:
                elec.grid_name = key

        # add interpolated points to overall list
        for key in self._grids:
            for elec in self._grids[key]:
                if elec.is_interpolation:
                    self._electrodes.append(elec)

        pipe.translate_electrodes_to_surface_space(
            self._electrodes, aff, subjects_dir=self.subjects_dir,
            subject=self.subject)

        #a very rapid cooling schedule shows pretty good performance
        #additional cooling offers very marginal returns and we prioritize
        #quick results so the user can adjust them
        pipe.snap_electrodes_to_surface(
            self._electrodes, self.hemisphere, subjects_dir=self.subjects_dir,
            #subject=self.subject, max_steps=2500)
            subject=self.subject, max_steps=200)

        # Store the sorted/interpolated points in separate maps for access
        for key in self._grids:
            for elec in self._grids[key]:
                if elec.is_interpolation:
                    self._interpolated_electrodes[elec.astuple()] = elec
                else:
                    self._sorted_electrodes[elec.astuple()] = elec

        # store the unsorted points in a separate map for access
        for elec in self._electrodes:
            sorted = False
            for key in self._grids:
                if sorted:
                    break
                for elec_other in self._grids[key]:
                    if elec is elec_other:
                        sorted=True
                        break
            if not sorted:
                self._unsorted_electrodes[elec.astuple()] = elec

        self._all_electrodes.update(self._interpolated_electrodes)
        self._all_electrodes.update(self._unsorted_electrodes)
        self._all_electrodes.update(self._sorted_electrodes)
    
        self._visualize_event = True

    def _fit_changes(self):
        #maybe this should be a different call which evaluates a single
        #grid
        _, _, self._grids = pipe.classify_electrodes(
            self._electrodes, self.electrode_geometry,
            fixed_points=self._grids.values())

class SurfaceVisualizerPanel(HasTraits):
    scene = Instance(MlabSceneModel,())
    model = Instance(ElectrodePositionsModel)

    _visualize_event = DelegatesTo('model')
    subject = DelegatesTo('model')
    subjects_dir = DelegatesTo('model')
    hemisphere = DelegatesTo('model')
    _colors = DelegatesTo('model')

    _grids = DelegatesTo('model')
    interactive_mode = DelegatesTo('model')

    _points_to_unsorted = DelegatesTo('model')
    _points_to_cur_grid = DelegatesTo('model')

    _all_electrodes = DelegatesTo('model')
    _unsorted_electrodes = DelegatesTo('model')

    brain = Any
    gs_glyphs = Dict

    traits_view = View(
        Item('scene', editor=SceneEditor(scene_class=MayaviScene),
            show_label=False),
        height=500, width=500)

    def __init__(self, model, **kwargs):
        super(SurfaceVisualizerPanel, self).__init__(**kwargs)
        self.model = model

    @on_trait_change('model:_visualize_event')
    def show_grids_on_surface(self):
        from mayavi import mlab
        #mlab.clf(figure = self.scene.mayavi_scene)

        #there is a bug in mlab.clf which causes the callbacks to become
        #disconnected in such a way that they cannot be reattached to the
        #scene. I tracked this bug to the VTK picker before giving up.
        #To avoid it, we write our own clf which does nothing
        #to the picker. Then, when mayavi reattaches the picker it
        #overwrites itself, which correctly discards the old one.
        
        #As a warning, this code is not safe from memory leaks. However,
        #it is no less unsafe than mlab.clf. The only truly safe way to
        #avoid memory leaks is as in CVU, where the Viewport (ie, this
        #SurfaceVisualizerPanel object) would be garbage collected and
        #rebuilt whenever the scene is changed.

        #A guy on stackoverflow pointed out the memory leaks of this approach
        #(using mlab.clf) after creating a loop which creates a complex
        #scene and takes screenshots thousands of times.
        #So for typical use, the memory leaks are probably not a big issue.
    
        from utils import clear_scene

        clear_scene(self.scene.mayavi_scene)

        from color_utils import set_discrete_lut

        import surfer
        brain = self.brain = surfer.Brain( 
            self.subject, subjects_dir=self.subjects_dir,
            surf='pial', curv=False, hemi=self.hemisphere,
            figure=self.scene.mayavi_scene)

        brain.brains[0]._geo_surf.actor.property.opacity = 0.35
        brain.toggle_toolbars(True)

        unsorted_elecs = map((lambda x:getattr(x, 'snap_coords')),
            self._unsorted_electrodes.values())
        self.gs_glyphs['unsorted'] = glyph = virtual_points3d( 
            unsorted_elecs, scale_factor=0.3, name='unsorted',
            figure=self.scene.mayavi_scene, color=self._colors['unsorted'])  

        set_discrete_lut(glyph, self._colors.values())
        glyph.mlab_source.dataset.point_data.scalars=(
            np.zeros(len(unsorted_elecs)))

        for i,key in enumerate(self._grids):
            grid_elecs = map((lambda x:getattr(x, 'snap_coords')), 
                self._grids[key])

            self.gs_glyphs[key] = glyph = virtual_points3d(grid_elecs,
                scale_factor=0.3, color=self._colors[key], 
                name=key, figure=self.scene.mayavi_scene)

            set_discrete_lut(glyph, self._colors.values())
            scalar_color = self._colors.keys().index(key)

            glyph.mlab_source.dataset.point_data.scalars=(
                np.ones(len(self._grids[key])) * scalar_color)

        #set the surface unpickable
        for srf in brain.brains:
            srf._geo_surf.actor.actor.pickable=False

        #setup the node selection callback
        picker = self.scene.mayavi_scene.on_mouse_pick( self.selectnode_cb )
        picker.tolerance = .02

    def selectnode_cb(self, picker):
        '''
        Callback to move an node into the selected state
        '''
        from color_utils import change_single_glyph_color
        from mayavi import mlab

        if self.interactive_mode is None:
            return
        target = self.interactive_mode.name
        if target in ('', 'unsorted'):
            return

        for key,nodes in zip(self.gs_glyphs.keys(), self.gs_glyphs.values()):
            if picker.actor in nodes.actor.actors:
                pt = int(picker.point_id/nodes.glyph.glyph_source.
                    glyph_source.output.points.to_array().shape[0])
                x,y,z = nodes.mlab_source.points[pt]
                elec = self._all_electrodes[(x,y,z)]
                current_key = elec.grid_name
                break

        #import pdb
        #pdb.set_trace()

        if elec in self._grids[target]:
            if (x,y,z) in self._points_to_unsorted:
                del self._points_to_unsorted[(x,y,z)]
                change_single_glyph_color(nodes, pt, 
                    self._colors.keys().index(current_key))
                elec.grid_transition_to = ''
            else:
                self._points_to_unsorted[(x,y,z)] = elec
                change_single_glyph_color(nodes, pt, 
                    self._colors.keys().index('unsorted'))
                elec.grid_transition_to = 'unsorted'
        else:
            if (x,y,z) in self._points_to_cur_grid:
                del self._points_to_cur_grid[(x,y,z)]
                change_single_glyph_color(nodes, pt, 
                    self._colors.keys().index(current_key))
                elec.grid_transition_to = ''
            else:
                self._points_to_cur_grid[(x,y,z)] = elec
                change_single_glyph_color(nodes, pt, 
                    self._colors.keys().index(target))
                elec.grid_transition_to = target

        mlab.draw()
                
class InteractivePanel(HasPrivateTraits):
    model = Instance(ElectrodePositionsModel)

    ct_scan = DelegatesTo('model')
    t1_scan = DelegatesTo('model')
    run_pipeline_button = Button('Extract electrodes to surface')

    subjects_dir = DelegatesTo('model')
    subject = DelegatesTo('model')
    fsdir_writable = DelegatesTo('model')

    electrode_geometry = DelegatesTo('model')
    hemisphere = DelegatesTo('model')

    _grid_named_objects = DelegatesTo('model')

    #interactive_mode = Instance(NameHolder)
    interactive_mode = DelegatesTo('model')
    find_best_fit_button = Button('Fit changes')
    shell = Dict

    viz = Instance(SurfaceVisualizerPanel)

    traits_view = View(
        HGroup(
            VGroup(
                Item('ct_scan'),
            ),
            VGroup(
                Item('electrode_geometry', editor=ListEditor(
                    editor=CSVListEditor(), rows=2), ), 
            ), 
            VGroup(
                Item('run_pipeline_button', show_label=False),
            ),
        ),
        HGroup(
                Item('subjects_dir'),
                Item('subject'),
                Item('hemisphere')
        ),
        HGroup(
            VGroup(
                Item('interactive_mode', 
                    editor=InstanceEditor(name='_grid_named_objects'),
                    style='custom', label='Add/remove electrodes from'),
            ),
            VGroup(
                Item('find_best_fit_button', show_label=False),
            ),
        ),

                Item('shell', show_label=False, editor=ShellEditor()),
        height=300, width=500
    )

    def __init__(self, model, viz=None, **kwargs):
        super(InteractivePanel, self).__init__(**kwargs)
        self.model = model
        self.viz = viz

    def _run_pipeline_button_fired(self):
        self.model._run_pipeline()

    def _find_best_fit_button_fired(self):
        self.model._fit_changes()

class iEEGCoregistrationFrame(HasTraits):
    interactive_panel = Instance(InteractivePanel)
    surface_visualizer_panel = Instance(SurfaceVisualizerPanel)

    traits_view = View(
        Group(
            Item('surface_visualizer_panel', editor=InstanceEditor(), 
                style='custom' ),
            Item('interactive_panel', editor=InstanceEditor(), style='custom'),
        show_labels=False),

        title=('llanfairpwllgwyngyllgogerychwyrndrobwllllantysiliogogogoch is '
            'nice this time of year'),
        height=800, width=700
    )

    def __init__(self, **kwargs):
        super(iEEGCoregistrationFrame, self).__init__(**kwargs)
        model = ElectrodePositionsModel()
        self.surface_visualizer_panel = SurfaceVisualizerPanel(model)
        self.interactive_panel = InteractivePanel(model, 
            self.surface_visualizer_panel)

iEEGCoregistrationFrame().configure_traits()

