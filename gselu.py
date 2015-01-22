from __future__ import division
import os
import numpy as np
from mayavi.core.ui.api import MayaviScene, SceneEditor, MlabSceneModel
from traits.api import (Bool, Button, cached_property, File, HasTraits,
    Instance, on_trait_change, Str, Property, Directory, Dict, DelegatesTo,
    HasPrivateTraits, Any, List, Enum, Int, Event, Float, Tuple)
from traitsui.api import (View, Item, Group, OKCancelButtons, ShellEditor,
    HGroup, VGroup, InstanceEditor, TextEditor, ListEditor, CSVListEditor,
    Handler, Label, OKCancelButtons)
from traitsui.message import error as error_dialog

from electrode import Electrode
from utils import virtual_points3d, NameHolder, GeometryNameHolder
from utils import crash_if_freesurfer_is_not_sourced, gensym
from geometry import load_affine

class ElectrodePositionsModel(HasPrivateTraits):
    ct_scan = File
    t1_scan = File
    subjects_dir = Directory
    subject = Str
    fsdir_writable = Bool

    use_ct_mask = Bool(True)
    disable_erosion = Bool(False)

    electrode_geometry = List(List(Int), [[8,8]]) # Gx2 list

    _electrodes = List(Electrode)
    interactive_mode = Instance(NameHolder)
    _grids = Dict # Str -> List(Electrode)
    _grid_named_objects = Property(depends_on='_grids')
    #_grid_named_objects = List(NameHolder)

    _sorted_electrodes = Dict # Tuple -> Electrode
    _interpolated_electrodes = Dict # Tuple -> Electrode
    _unsorted_electrodes = Dict # Tuple -> Electrode
    _all_electrodes = Dict # Tuple -> Electrode
        # dictionary from surface coordinates (as hashable) to reused
        # electrode objects

    _surf_to_ct_map = Dict
    _ct_to_surf_map = Dict

    _ct_to_grid_ident_map = Dict # Tuple -> Str
    
    _points_to_cur_grid = Dict
    _points_to_unsorted = Dict

    _single_glyph_to_recolor = Tuple
    _new_glyph_color = Any
    _new_ras_positions = Dict

    _interactive_mode_snapshot = Str

    _rebuild_vizpanel_event = Event
    _rebuild_guipanel_event = Event
    _update_glyph_lut_event = Event
    _update_single_glyph_event = Event
    _reorient_glyph_event = Event

    _visualization_ready = Bool(False)

    _colors = Any # OrderedDict(Grid -> Color)
    _color_scheme = Any #Generator returning 3-tuples
    _grid_geom = Dict # Grid -> Gx2 list

    ct_registration = File

    ct_threshold = Float(2500.)
    delta = Float(0.5)
    epsilon = Float(10.)
    rho = Float(35.)
    rho_strict = Float(20.)
    rho_loose = Float(50.)

    delta_recon = Float(0.65)
    epsilon_recon = Float(10.)
    rho_recon = Float(40.)
    rho_strict_recon = Float(30.)
    rho_loose_recon = Float(55.)

    snapping_enabled = Bool(True)
    nr_steps = Int(2500)
    deformation_constant = Float(1.)

    #state-storing interactive labeling windows
    mlaw = Instance(HasTraits)
    alaw = Instance(HasTraits)
    raw = Instance(HasTraits)
    
    #def __grid_named_objects_default(self):
    #    return self._get__grid_named_objects()

    #grid named objects is broken, does not update on reload and does not
    #fully update on grid adding
    @cached_property
    def _get__grid_named_objects(self):
        from color_utils import mayavi2traits_color
        grid_names = [NameHolder(name=''), 
            GeometryNameHolder(name='unsorted',
                geometry='n/a',
                #TODO dont totally hardcode this color
                color=mayavi2traits_color((1,0,0)))]

        #for key in sorted(self._grids.keys()):
        #use the canonical order as the order to appear in the list
        if self._colors is not None:
            for key in self._colors.keys():
                if key in ('unsorted','selection'):
                    continue
                grid_names.append(GeometryNameHolder(
                    name=key, 
                    geometry=str(self._grid_geom[key]), 
                    color=mayavi2traits_color(self._colors[key])))

        #if len(self._grids) > 0:
        #import pdb
        #pdb.set_trace()

        return grid_names

    def _interactive_mode_changed(self):
        self._commit_grid_changes()

    def _commit_grid_changes(self):
        for p in (self._points_to_cur_grid, self._points_to_unsorted):
            for loc in p:
                elec = p[loc]
                
                old = elec.grid_name
                new = elec.grid_transition_to

                elec.grid_name = new
                elec.grid_transition_to = ''
        
                if old not in ('','unsorted','selection'):
                    self._grids[old].remove(elec)
                    self._ct_to_grid_ident_map[elec.asct()] = 'unsorted'
                if new not in ('','unsorted','selection'):
                    self._grids[new].append(elec)
                    self._ct_to_grid_ident_map[elec.asct()] = new

        self._points_to_cur_grid = {}
        self._points_to_unsorted = {}
    
    def acquire_affine(self):
        import pipeline as pipe
        if self.ct_registration not in (None, ''):
            aff = load_affine(self.ct_registration)
        else:
            aff = pipe.register_ct_to_mr_using_mutual_information(
                self.ct_scan, subjects_dir=self.subjects_dir, 
                subject=self.subject)

        return aff

    def run_pipeline(self):
        #setup
        if self.subjects_dir is None or self.subjects_dir=='':
            self.subjects_dir = os.environ['SUBJECTS_DIR']
        if self.subject is None or self.subject=='':
            self.subject = os.environ['SUBJECT']

        self._grids = {}
        #self._grid_named_objects = self._get__grid_named_objects()
        self.interactive_mode = self._grid_named_objects[0]
        #manually handle property


        self._electrodes = []
        self._all_electrodes = {}
        self._unsorted_electrodes = {}
        self._sorted_electrodes = {}
        self._interpolated_electrodes = {}

        self._ct_to_surf_map = {}
        self._surf_to_ct_map = {}

        #self._visualization_ready = False

        #pipeline
        import pipeline as pipe
        
        #adjust the brainmask creation to use the existing affine if provided,
        #requires us to be "clever" and create an LTA for that
        #TODO
        if self.use_ct_mask:
            ct_mask = pipe.create_brainmask_in_ctspace(self.ct_scan,
                subjects_dir=self.subjects_dir, subject=self.subject)
        else:
            ct_mask = None

        self._electrodes = pipe.identify_electrodes_in_ctspace(
            self.ct_scan, mask=ct_mask, threshold=self.ct_threshold,
            use_erosion=(not self.disable_erosion)) 

        aff = self.acquire_affine()

        pipe.create_dural_surface(subjects_dir=self.subjects_dir, 
            subject=self.subject)

        #initial sorting
        #self._grids, self._colors = pipe.classify_electrodes(
        try:
            self._colors, self._grid_geom, self._grids, self._color_scheme = (
                pipe.classify_electrodes(self._electrodes,
                                         self.electrode_geometry,
                                         delta = self.delta,
                                         epsilon = self.epsilon,
                                         rho = self.rho,
                                         rho_strict = self.rho_strict,
                                         rho_loose = self.rho_loose
                                        ))
        except ValueError as e:
            error_dialog(str(e))
            raise

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
        if self.snapping_enabled:
            pipe.snap_electrodes_to_surface(
                self._electrodes, subjects_dir=self.subjects_dir,
                subject=self.subject, max_steps=self.nr_steps)

        # Store the sorted/interpolated points in separate maps for access
        for key in self._grids:
            for elec in self._grids[key]:
                if elec.is_interpolation:
                    self._interpolated_electrodes[elec.asct()] = elec
                else:
                    self._sorted_electrodes[elec.asct()] = elec

                #save each electrode's grid identity
                self._ct_to_grid_ident_map[elec.asct()] = key

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
                self._unsorted_electrodes[elec.asct()] = elec

                self._ct_to_grid_ident_map[elec.asct()] = 'unsorted'

        self._all_electrodes.update(self._interpolated_electrodes)
        self._all_electrodes.update(self._unsorted_electrodes)
        self._all_electrodes.update(self._sorted_electrodes)
    
        #save mapping from ct to surface coords
        for elec in self._all_electrodes.values():
            if self.snapping_enabled:
                surf_coord = elec.astuple()
            else:
                surf_coord = elec.asras()

            self._ct_to_surf_map[elec.asct()] = surf_coord
            self._surf_to_ct_map[surf_coord] = elec.asct()

        self._visualization_ready = True
        self._rebuild_vizpanel_event = True
        self._rebuild_guipanel_event = True

    def add_grid(self):
        name = 'usergrid%s'%gensym()

        #force self._grids to update (GUI depends on cached property)
#        temp_grids = self._grids
#        self._grids = {}
#        self._grids.update(temp_grids)

        #geometry and color data should be defined first so that when grids
        #grids is updated the GUI does not error out looking for this info
        self._grid_geom[name] = 'user-defined'
        self._colors[name] = self._color_scheme.next()

        #testing GUI update bug
        #temp_grids = self._grids.copy()
        #temp_grids[name] = []
        #self._grids = temp_grids

        self._grids[name] = []
        #self._grid_named_objects = self._get__grid_named_objects()
        self.interactive_mode = self._grid_named_objects[0]
        
        self._update_glyph_lut_event = True
        self._update_guipanel_event = True

    def change_single_glyph(self, xyz, elec, target, current_key):
        if elec in self._grids[target]:
            if xyz in self._points_to_unsorted:
                del self._points_to_unsorted[xyz]
                elec.grid_transition_to = ''
                self._new_glyph_color = self._colors.keys().index(current_key)
            else:
                self._points_to_unsorted[xyz] = elec
                elec.grid_transition_to = 'unsorted'
                self._new_glyph_color = self._colors.keys().index('unsorted')
        else:
            if xyz in self._points_to_cur_grid:
                del self._points_to_cur_grid[xyz]
                elec.grid_transition_to = ''
                self._new_glyph_color = self._colors.keys().index(current_key)
            else:
                self._points_to_cur_grid[xyz] = elec
                elec.grid_transition_to = target
                self._new_glyph_color = self._colors.keys().index(target)

        self._single_glyph_to_recolor = xyz
        self._update_single_glyph_event = True

    def open_registration_window(self):
        cur_grid = self.interactive_mode
        #TODO WHY IS REGISTRATION DONE BY GRIDS? Shouldnt we adjust the
        #entire image registration?
        if cur_grid is None:
            error_dialog('Registration is done gridwise')
            return
        if cur_grid.name in ('', 'unsorted'):
            error_dialog('Regsitration is done gridwise')
            return

        from utils import RegistrationAdjustmentWindow
        self.raw = RegistrationAdjustmentWindow(
            model = self,
            cur_grid = cur_grid.name)
        self.raw.edit_traits()

    def reorient_glyph(self, target, matrix):
        self._commit_grid_changes()

        old_locs = map(lambda x:x.asras(), self._grids[target])

        from geometry import apply_affine
        new_locs = apply_affine(old_locs, matrix)

        for oloc, nloc in zip(old_locs, new_locs):
            self._new_ras_positions[tuple(oloc)] = tuple(nloc)

        self._interactive_mode_snapshot = target

        #draw the changes, no underlying state has yet been altered
        self._reorient_glyph_event = True

        #for unknown reason the operation causes the LUT to be destroyed
        #we can recreate it easily
        self._update_glyph_lut_event = True

        #import pdb
        #pdb.set_trace()

        for oloc,nloc,elec in zip(old_locs, new_locs, self._grids[target]):

            if tuple(oloc) == tuple(nloc):
                continue

            elec.surf_coords = tuple(nloc)
            self._ct_to_surf_map[elec.asct()] = tuple(nloc)
            del self._surf_to_ct_map[tuple(oloc)]
            self._surf_to_ct_map[tuple(nloc)] = elec.asct()

    def fit_changes(self):
        #maybe this should be a different call which evaluates a single
        #grid

        #currently we dont use this
        _, _, self._grids = pipe.classify_electrodes(
            self._electrodes, self.electrode_geometry,
            fixed_points=self._grids.values())

    def assign_manual_labels(self):
        self._commit_grid_changes()
        cur_grid = self.interactive_mode

        if cur_grid is None:
            error_dialog('Select a grid to assign labels')
            return
        if cur_grid.name in ('','unsorted'):
            error_dialog('Select a grid to assign labels')
            return

        from utils import ManualLabelAssignmentWindow
        self.mlaw = ManualLabelAssignmentWindow(
            model = self,
            cur_grid = cur_grid.name,
            electrodes = self._grids[cur_grid.name])
        self.mlaw.edit_traits()

    def assign_automated_labels(self):
        self._commit_grid_changes()
        cur_grid = self.interactive_mode

        if cur_grid is None:
            error_dialog('Select a grid to assign labels')
            return
        if cur_grid.name in ('','unsorted'):
            error_dialog('Select a grid to assign labels')
            return

        from utils import AutomatedAssignmentWindow
        self.alaw = AutomatedAssignmentWindow(
            model = self,
            cur_grid = cur_grid.name,
            electrodes = self._grids[cur_grid.name])
        self.alaw.edit_traits()

    def reconstruct_all_geometry(self):
        import pipeline as pipe

        for key in self._grids:
            pipe.classify_single_fixed_grid(key, self._grids, self._grid_geom,
                self._colors, 
                delta=self.delta, 
                epsilon=self.epsilon,
                rho=self.rho, 
                rho_loose=self.rho_loose,
                rho_strict=self.rho_strict)

    def reconstruct_geometry(self):
        self._commit_grid_changes()
        key = self.interactive_mode.name

        import pipeline as pipe
        success, interpolated_points = pipe.classify_single_fixed_grid(key, 
            self._grids, self._grid_geom, self._colors,
            delta = self.delta,
            epsilon = self.epsilon,
            rho = self.rho,
            rho_loose = self.rho_loose,
            rho_strict = self.rho_strict)

        print success, 'nimmo'

    
        #after geom reconstruction the interpolated points need to
        #be translated to the surface and *all points* must be
        #snapped again
        for elec in interpolated_points:
            self._grids[key].append(elec)

        aff = self.acquire_affine() 

        pipe.translate_electrodes_to_surface_space(
            self._grids[key], aff, subjects_dir=self.subjects_dir,
            subject=self.subject)

        #at this point its safe to snap only the electrodes that we have
        #isolated -- that is, just snap this individual grid only
        
        #if the individual grid is not snapped alone, we should do that
        #even if the user doesnt reconstruct the grid
        if self.snapping_enabled:
            pipe.snap_electrodes_to_surface(
                self._grids[key], subjects_dir=self.subjects_dir,
                subject=self.subject, max_steps=self.nr_steps)

        #add electrode to grid data structures
        for elec in interpolated_points:
            if self.snapping_enabled:
                surf_coord = elec.astuple()
            else:
                surf_coord = elec.asras()
            self._ct_to_surf_map[elec.asct()] = surf_coord
            self._surf_to_ct_map[surf_coord] = elec.asct()

        #do something to update the visualization with the new points
        self._rebuild_vizpanel_event = True

    def save_label_files(self):
        self._commit_grid_changes()

        if self.interactive_mode is None:
            print "select a grid to save labels from"
            return
        target = self.interactive_mode.name
        if target in ('unsorted',):
            print "select a grid to save labels from"
            return

        #if not self._geom_reconstructed:
        #    error_dialog('Finish reconstructing geometry first')
        #    return

        #this is the saving part

        from file_dialog import save_in_directory
        labeldir = save_in_directory()

        if os.path.exists(labeldir) and not os.path.isdir(labeldir):
            error_dialog('Cannot write labels to a non-directory')
            raise ValueError('Cannot write labels to a non-directory')

        # if the empty string is returned, the user cancelled the operation
        if labeldir == '':
            return

        try:
            os.makedirs(labeldir)
        except OSError:
            #potentially handle errno further
            pass

        from mne.label import Label

        #import pdb
        #pdb.set_trace()

        #only save label files for the current grid
        key = self.interactive_mode.name

        if self.snapping_enabled:
            import pipeline as pipe
            pipe.snap_electrodes_to_surface(
                self._grids[key], subjects_dir=self.subjects_dir,
                subject=self.subject, max_steps=self.nr_steps)

        #for key in self._grids:
        for j,elec in enumerate(self._grids[key]):
            if elec.name != '':
                label_name = elec.name
            else:
                elec_id = elec.geom_coords
                elec_2dcoord = ('unsorted%i'%j if len(elec_id)==0 else
                    str(elec_id))
                label_name = '%s_elec_%s'%(key, elec_2dcoord)

            if self.snapping_enabled:
                pos = [elec.pial_coords.tolist()]
                vertices = [elec.vertno]
                hemi = elec.hemi
            else:
                pos = [tuple(elec.surf_coords)]
                vertices = [0]
                hemi = 'lh'

            label = Label(vertices=vertices, pos=pos, hemi=hemi,
                          subject=self.subject,
                          name=label_name)
            label.save( os.path.join( labeldir, label_name ))

class ParamsPanel(HasTraits):
    model = Instance(ElectrodePositionsModel)

    ct_threshold = DelegatesTo('model')
    delta = DelegatesTo('model')
    epsilon = DelegatesTo('model')
    rho = DelegatesTo('model')
    rho_loose = DelegatesTo('model')
    rho_strict = DelegatesTo('model')
    delta_recon = DelegatesTo('model')
    epsilon_recon = DelegatesTo('model')
    rho_recon = DelegatesTo('model')
    rho_loose_recon = DelegatesTo('model')
    rho_strict_recon = DelegatesTo('model')
    #visualize_in_ctspace = DelegatesTo('model')
    nr_steps = DelegatesTo('model')
    snapping_enabled = DelegatesTo('model')
    deformation_constant = DelegatesTo('model')
    use_ct_mask = DelegatesTo('model')
    disable_erosion = DelegatesTo('model')

    traits_view = View(
        Group(
        HGroup(
        VGroup(
            Label('The threshold above which electrode clusters will be\n'
                'extracted from the CT image'),
            Item('ct_threshold'),
            Label('Snap to surface space (turn off for depth electrodes)'),
            Item('snapping_enabled'),
            Label('Number of steps before convergence in snap-to-surface\n'
                'algorithm'),
            Item('nr_steps', enabled_when='snapping_enabled'),
            Label('Weight given to the deformation term in the snapping\n'
                'algorithm, reduce if snapping error is very high.'),
            Item('deformation_constant', enabled_when='snapping_enabled'),
            Label('Try to extract the brain from the CT image and mask\n'
                'extracranial noise -- takes several minutes'),
            Item('use_ct_mask'),
            Label('Disable binary erosion procedure to reduce CT noise'),
            Item('disable_erosion'),
        ),
        VGroup(
            Label('Delta controls the distance between electrodes. That is,\n'
                'electrode distances must be between c*(1-d) and c*(1+d),\n'
                'where c is an estimate of the correct distance.'),
            Item('delta'),
            Label('Epsilon controls the tolerance of the initial angle\n'
                'difference from 90 degrees (or 180 degrees in rare cases).'),
            Item('epsilon'),
            Label('Rho controls the maximum allowable discrepancy between\n'
                'angles relative to their position to already fitted\n'
                'electrodes mostly as the difference |rho-90|.\n'
                'Rho_strict and Rho_loose are used in similar cases,\n' 
                'which demand slightly greater or smaller constraints.'),
            Item('rho'),
            Item('rho_strict'),
            Item('rho_loose'),
        ),
        ),),
    title='Edit parameters',
    buttons=OKCancelButtons,
    )

class SurfaceVisualizerPanel(HasTraits):
    scene = Instance(MlabSceneModel,())
    model = Instance(ElectrodePositionsModel)

    subject = DelegatesTo('model')
    subjects_dir = DelegatesTo('model')
    _colors = DelegatesTo('model')

    _grids = DelegatesTo('model')
    interactive_mode = DelegatesTo('model')

    _points_to_unsorted = DelegatesTo('model')
    _points_to_cur_grid = DelegatesTo('model')

    _all_electrodes = DelegatesTo('model')
    _unsorted_electrodes = DelegatesTo('model')
    _ct_to_surf_map = DelegatesTo('model')
    _surf_to_ct_map = DelegatesTo('model')

    visualize_in_ctspace = Bool(False)
    _viz_coordtype = Property#(depends_on='visualize_in_ctspace')
    def _get__viz_coordtype(self):
        if self.visualize_in_ctspace:
            return 'ct_coords'
        elif self.model.snapping_enabled:
            return 'snap_coords'
        else:
            return 'surf_coords'

    brain = Any
    gs_glyphs = Dict

    traits_view = View(
        Item('scene', editor=SceneEditor(scene_class=MayaviScene),
            show_label=False),
        height=500, width=500)

    def __init__(self, model, **kwargs):
        super(SurfaceVisualizerPanel, self).__init__(**kwargs)
        self.model = model

    @on_trait_change('scene:activated')
    def setup(self):
        #import pdb
        #pdb.set_trace()
        if self.model._visualization_ready:
            self.show_grids_on_surface()

    def show_grids_on_surface(self):

        from mayavi import mlab
        #mlab.clf(figure = self.scene.mayavi_scene)

        #there is a bug in mlab.clf which causes the callbacks to become
        #disconnected in such a way that they cannot be reattached to the
        #scene. I tracked this bug to the VTK picker before giving up.

        #To avoid this, we use a workaround -- discard the scene every
        #single time we want to use mlab.clf and replace it with an
        #entirely new MlabSceneModel instance. This has the added benefit
        #of (according to my tests) avoiding memory leaks.

        #from utils import clear_scene
        #clear_scene(self.scene.mayavi_scene)

        from color_utils import set_discrete_lut

        import surfer
        if not self.visualize_in_ctspace:
            brain = self.brain = surfer.Brain( 
                self.subject, subjects_dir=self.subjects_dir,
                surf='pial', curv=False, hemi='both',
                figure=self.scene.mayavi_scene)

            brain.toggle_toolbars(True)

            #set the surface unpickable
            for srf in brain.brains:
                srf._geo_surf.actor.actor.pickable=False
                srf._geo_surf.actor.property.opacity = 0.4

            scale_factor = 3.
        else:
            scale_factor = 5.
    
        unsorted_elecs = map((lambda x:getattr(x, self._viz_coordtype)),
            self._unsorted_electrodes.values())
        self.gs_glyphs['unsorted'] = glyph = virtual_points3d( 
            unsorted_elecs, scale_factor=scale_factor, name='unsorted',
            figure=self.scene.mayavi_scene, color=self._colors['unsorted'])  

        set_discrete_lut(glyph, self._colors.values())
        glyph.mlab_source.dataset.point_data.scalars=(
            np.zeros(len(unsorted_elecs)))

        for i,key in enumerate(self._grids):
            grid_elecs = map((lambda x:getattr(x, self._viz_coordtype)), 
                self._grids[key])

            if len(grid_elecs)==0:
                continue

            self.gs_glyphs[key] = glyph = virtual_points3d(grid_elecs,
                scale_factor=scale_factor, color=self._colors[key], 
                name=key, figure=self.scene.mayavi_scene)

            set_discrete_lut(glyph, self._colors.values())
            scalar_color = self._colors.keys().index(key)

            glyph.mlab_source.dataset.point_data.scalars=(
                np.ones(len(self._grids[key])) * scalar_color)

        #setup the node selection callback
        picker = self.scene.mayavi_scene.on_mouse_pick( self.selectnode_cb )
        picker.tolerance = .02

    def redraw_single_grid(self, key):
        #this function is never called
        #it is always easier to redraw everything, and avoids memory leaks

        from mayavi import mlab
        from color_utils import set_discrete_lut

        #removing glyphs may cause memory leaks
        self.gs_glyphs[key].remove()

        grid_elecs = map((lambda x:getattr(x, self._viz_coordtype)),
            self._grids[key])

        if len(grid_elecs)==0:
            return

        self.gs_glyphs[key] = glyph = virtual_points3d(grid_elecs,
            scale_factor=scale_factor, color=self._colors[key],
            name=key, figure=self.scene.mayavi_scene)
      
        set_discrete_lut(glyph, self._colors.values())
        scalar_color = self._colors.keys().index(key)

        glyph.mlab_source.dataset.point_data.scalars=(
            np.ones(len(self._grids[key])) * scalar_color)

    def selectnode_cb(self, picker):
        '''
        Callback to move an node into the selected state
        '''
        #import pdb
        #pdb.set_trace()

        if self.interactive_mode is None:
            return
        target = self.interactive_mode.name
        if target in ('', 'unsorted', 'selection'):
            return

        current_key = None

        for key,nodes in zip(self.gs_glyphs.keys(), self.gs_glyphs.values()):
            if picker.actor in nodes.actor.actors:
                pt = int(picker.point_id/nodes.glyph.glyph_source.
                    glyph_source.output.points.to_array().shape[0])
                x,y,z = nodes.mlab_source.points[pt]

                #translate from CT to surf coords if necessary
                if not self.visualize_in_ctspace:
                    x,y,z = self._surf_to_ct_map[(x,y,z)]

                elec = self._all_electrodes[(x,y,z)]

                current_key = elec.grid_name
                break

        #if the user did not click on anything interesting, do nothing
        if current_key is None:
            return

        self.model.change_single_glyph((x,y,z), elec, target, current_key)
    
    @on_trait_change('model:_update_single_glyph_event')
    def update_single_glyph(self):

        if len(self.gs_glyphs)==0:
            #this callback is still hooked to the SurfaceVisualizerPanel 
            #instance which has been disconnected from the scene
            return

        from color_utils import change_single_glyph_color
        from mayavi import mlab
        #import pdb
        #pdb.set_trace()
        xyz = self.model._single_glyph_to_recolor
        if not self.visualize_in_ctspace:
            xyz = self._ct_to_surf_map[xyz]

        for nodes in self.gs_glyphs.values():
            pt, = np.where( np.all(nodes.mlab_source.points == xyz, axis=1 ))
            if len(pt) > 0:
                break

        if len(pt)==0:
            #why does this error sometimes get thrown when a correct point
            #was clicked and the visualization changes
            raise ValueError('Error in figuring out what point was clicked')

        change_single_glyph_color(nodes, int(pt), self.model._new_glyph_color)

        mlab.draw(figure=self.scene.mayavi_scene)

    @on_trait_change('model:_update_glyph_lut_event')
    def update_glyph_lut(self):
        from color_utils import set_discrete_lut
        for glyph in self.gs_glyphs.values():
            set_discrete_lut(glyph, self._colors.values())

    @on_trait_change('model:_reorient_glyph_event')
    def update_orientation(self):
        #only do this for surface viz, do not adjust CT electrode positions
        if self.visualize_in_ctspace:
            return

        target = self.model._interactive_mode_snapshot

        for glyph in self.gs_glyphs.values():
            points = np.array(glyph.mlab_source.dataset.points)

            for i,point in enumerate(points):
                surf_point = tuple(self.model._surf_to_ct_map[tuple(point)])
                if self.model._ct_to_grid_ident_map[surf_point] == target:
                    points[i] = self.model._new_ras_positions[tuple(point)] 

            glyph.mlab_source.dataset.points = points

        #glyph = self.gs_glyphs[self._model._interactive_mode_snapshot]

        #not all of the desired points will be in this glyph.
        #it is necessary to either
            #1. redraw the scene (we are not doing this)
            #2. determine the electrode identities, by iterating through
            #   each glyphs and somehow figuring out the electrode
            #   identity, perhaps by maintaining a table in the model

            #   and then storing and updating the locations of
            #   *all* glyphs

            #   note, all the electrodes we want are in one Grid
            #   but not necessarily one glyph.
        
            #   so it does not make sense to calculate all the electrode
            #   locations and leave them unordered, because we will need to 
            #   replace the positions array for each glyph individually
            #   after we figure out the membership of each electrode

            #   so it is probably best to just to access each electrode
            #   directly from the model to get its updated coordinates
                
class InteractivePanel(HasPrivateTraits):
    model = Instance(ElectrodePositionsModel)

    ct_scan = DelegatesTo('model')
    t1_scan = DelegatesTo('model')
    run_pipeline_button = Button('Run pipeline')

    subjects_dir = DelegatesTo('model')
    subject = DelegatesTo('model')
    fsdir_writable = DelegatesTo('model')

    ct_registration = DelegatesTo('model')

    electrode_geometry = DelegatesTo('model')

    _grid_named_objects = DelegatesTo('model')

    #interactive_mode = Instance(NameHolder)
    interactive_mode = DelegatesTo('model')
    add_grid_button = Button('Add new grid')
    shell = Dict

    save_label_files_button = Button('Save freesurfer labels')

    edit_parameters_button = Button('Edit Fitting Parameters')
    
    reconstruct_geom_button = Button('Reconstruct geometry')
    assign_manual_labels_button = Button('Label manually')
    assign_automated_labels_button = Button('Label automatically')
    adjust_registration_button = Button('Adjust registration')

    #we retain a reference to easily reference the visualization in the shell
    viz = Instance(SurfaceVisualizerPanel)
    ctviz = Instance(SurfaceVisualizerPanel)

    traits_view = View(
        HGroup(
            VGroup(
                Item('ct_scan'),
                #Item('ct_registration', label='reg matrix\n(optional)')
                Item('adjust_registration_button', show_label=False),
            ),
            VGroup(
                Item('electrode_geometry', editor=ListEditor(
                    editor=CSVListEditor(), rows=2), ), 
            ), 
            VGroup(
                Item('run_pipeline_button', show_label=False),
                Item('edit_parameters_button', show_label=False),
                Item('save_label_files_button', show_label=False),
            ),
        ),
        HGroup(
                Item('subjects_dir'),
                Item('subject'),
        ),
        HGroup(
            VGroup(
                Item('interactive_mode', 
                    editor=InstanceEditor(name='_grid_named_objects'),
                    style='custom', label='Edit electrodes\nfrom grid'),
            ),
            VGroup(
                Item('add_grid_button', show_label=False),
                Item('reconstruct_geom_button', show_label=False),
            ),
            VGroup(
                Item('assign_manual_labels_button', show_label=False),
                Item('assign_automated_labels_button', show_label=False),
            ),
        ),

                Item('shell', show_label=False, editor=ShellEditor()),
        height=300, width=500
    )

    def __init__(self, model, viz=None, ctviz=None, **kwargs):
        super(InteractivePanel, self).__init__(**kwargs)
        self.model = model
        self.viz = viz
        self.ctviz = ctviz

    def _run_pipeline_button_fired(self):
        self.model.run_pipeline()

    def _add_grid_button_fired(self):
        self.model.add_grid()

    def _find_best_fit_button_fired(self):
        self.model.fit_changes()

    def _save_label_files_button_fired(self):
        self.model.save_label_files()

    def _edit_parameters_button_fired(self):
        ParamsPanel(model=self.model).edit_traits()

    def _reconstruct_geom_button_fired(self):
        self.model.reconstruct_geometry()

    def _assign_manual_labels_button_fired(self):
        self.model.assign_manual_labels()

    def _assign_automated_labels_button_fired(self):
        self.model.assign_automated_labels()

    def _adjust_registration_button_fired(self):
        self.model.open_registration_window()

class iEEGCoregistrationFrame(HasTraits):
    model = Instance(ElectrodePositionsModel)
    interactive_panel = Instance(InteractivePanel)
    surface_visualizer_panel = Instance(SurfaceVisualizerPanel)
    ct_visualizer_panel = Instance(SurfaceVisualizerPanel)
    #viz_interface = Instance(IntermediateVizInterface)

    traits_view = View(
        Group(
            HGroup(
                Item('surface_visualizer_panel', editor=InstanceEditor(), 
                    style='custom', resizable=True ),
                Item('ct_visualizer_panel', editor=InstanceEditor(),
                    style='custom', resizable=True ),
            show_labels=False),
            Item('interactive_panel', editor=InstanceEditor(), style='custom',
                resizable=True),
        show_labels=False),
        title=('llanfairpwllgwyngyllgogerychwyrndrobwllllantysiliogogogoch is'
            ' nice this time of year'),
        height=800, width=800, resizable=True
    )

    def __init__(self, **kwargs):
        super(iEEGCoregistrationFrame, self).__init__(**kwargs)
        model = self.model = ElectrodePositionsModel()
        self.surface_visualizer_panel = SurfaceVisualizerPanel(model)
        self.ct_visualizer_panel = SurfaceVisualizerPanel(model,
            visualize_in_ctspace=True)
        self.interactive_panel = InteractivePanel(model,
            viz=self.surface_visualizer_panel,
            ctviz=self.ct_visualizer_panel)

    @on_trait_change('model:_rebuild_vizpanel_event')
    def _rebuild_vizpanel(self):
        self.surface_visualizer_panel = SurfaceVisualizerPanel(self.model)
        self.interactive_panel.viz = self.surface_visualizer_panel

        self.ct_visualizer_panel = SurfaceVisualizerPanel(self.model,
            visualize_in_ctspace=True)
        self.interactive_panel.ctviz = self.ct_visualizer_panel

    @on_trait_change('model:_rebuild_guipanel_event')
    def _rebuild_guipanel(self):
        self.interactive_panel = InteractivePanel(self.model,
            viz=self.surface_visualizer_panel,
            ctviz=self.ct_visualizer_panel)

crash_if_freesurfer_is_not_sourced()
iEEGCoregistrationFrame().configure_traits()

