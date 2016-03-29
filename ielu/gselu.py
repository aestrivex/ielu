from __future__ import division
import os
import sys
import numpy as np
from mayavi.core.ui.api import MayaviScene, SceneEditor, MlabSceneModel
from traits.api import (Bool, Button, cached_property, File, HasTraits,
    Instance, on_trait_change, Str, Property, Directory, Dict, DelegatesTo,
    HasPrivateTraits, Any, List, Enum, Int, Event, Float, Tuple, Range,
    Color)
from traitsui.api import (View, Item, Group, OKCancelButtons, ShellEditor,
    HGroup, VGroup, InstanceEditor, TextEditor, ListEditor, CSVListEditor,
    Handler, Label, OKCancelButtons, VSplit)
from traitsui.message import error as error_dialog
from traitsui.api import MenuBar, Menu, Action

from custom_list_editor import CustomListEditor

from electrode import Electrode
from name_holder import NameHolder, GeometryNameHolder, NameHolderDisplayer
from utils import (virtual_points3d, crash_if_freesurfer_is_not_sourced,
    gensym, get_subjects_dir, intize)
from color_utils import mayavi2traits_color
from geometry import load_affine
from functools import partial
import nibabel as nib

class ElectrodePositionsModel(HasPrivateTraits):
#class ElectrodePositionsModel(HasTraits):
    ct_scan = File
    t1_scan = File
    subjects_dir = Directory
    def _subjects_dir_default(self):
        if 'SUBJECTS_DIR' in os.environ:
            return os.environ['SUBJECTS_DIR']
        else:
            return ''
    subject = Str
    def _subject_default(self):
        if 'SUBJECT' in os.environ:
            return os.environ['SUBJECT']
        else:
            return ''
    fsdir_writable = Bool

    use_ct_mask = Bool(False)
    disable_erosion = Bool(False)
    overwrite_xfms = Bool(False)
    registration_procedure = Enum('experimental shape correction',
        'uncorrected MI registration', 'no registration')
    shapereg_slice_diff = Float(5.0)

    #this was previously marked as transient which did not cause any problem
    #is there a reason it needed to be transient?
    zoom_factor_override = List(Float, [1.0, 1.0, 1.4] )

    electrode_geometry = List(List(Int), [[8,8]]) # Gx2 list

    electrodes = List(Electrode)
    #interactive_mode = Instance(NameHolder)
    interactive_mode = Property
    def _get_interactive_mode(self):
        return self.interactive_mode_displayer.interactive_mode
    #def _set_interactive_mode(self, val):
    #    self.interactive_mode_displayer.interactive_mode = val

    interactive_mode_displayer = Instance(NameHolderDisplayer, (),
        transient=False)

    _grids = Dict # Str -> List(Electrode)
    #_grid_named_objects = Property(depends_on='_grids')
    #_grid_named_objects = List(NameHolder)
    
    #@on_trait_change('_grid_named_objects')
    #def _update_interactive_mode_displayer(self):
    #    self.interactive_mode_displayer.name_holders = (
    #        self._grid_named_objects)
    #def _update_grid_named_objects_force(self):
    #    swap = self._grid_named_objects
    #    self._grid_named_objects = []
    #    self._grid_named_objects = swap

    surface_opacity = Float(0.4)
        
    _sorted_electrodes = Dict() # Tuple -> Electrode
    _interpolated_electrodes = Dict() # Tuple -> Electrode
    _unsorted_electrodes = Dict() # Tuple -> Electrode
    _all_electrodes = Dict() # Tuple -> Electrode
        # dictionary from surface coordinates (as hashable) to reused
        # electrode objects

    _surf_to_iso_map = Dict()
    _iso_to_surf_map = Dict()

    _iso_to_grid_ident_map = Dict() # Tuple -> Str
    
    _points_to_cur_grid = Dict()
    _points_to_unsorted = Dict()

    _single_glyph_to_recolor = Tuple()
    _new_glyph_color = Any()
    _new_ras_positions = Dict()

    _interactive_mode_snapshot = Str()

    _rebuild_vizpanel_event = Event
    _rebuild_guipanel_event = Event
    _update_glyph_lut_event = Event
    _update_single_glyph_event = Event
    _reorient_glyph_event = Event

    _visualization_ready = Bool(False, )

    _add_annotation_event = Event
    _add_label_event = Event
    _remove_labels_event = Event 
    _label_file = Str()
    _label_borders = Bool(True, )
    _label_opacity = Range(0., 1., 1., )
    _label_color = Color()
    _label_hemi = Enum('both','lh','rh', )

    _hide_noise_event = Event
    _noise_hidden = Bool(False, )

    _colors = Any() # OrderedDict(Grid -> Color)
    _color_scheme = Any(transient=True) #Generator returning 3-tuples
    _grid_geom = Dict() # Grid -> Gx2 list
    _grid_types = Dict() # Grid -> Str

    ct_registration = File

    ct_threshold = Float(2500.)
    dilation_iterations = Int(25)

    critical_percentage = Range(0., 1., 0.75)

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

    isotropize = Enum('By header', 'By voxel', 'Manual override',
        'Isotropization off')

    #this was previously marked as transient which caused a problem with
    #adding new points after load. is there a reason it needs to be transient?
    isotropization_override = List(Float, [1.0, 1.0, 2.5] )

    roi_parcellation = Str('aparc')
    roi_error_radius = Float(4.)

    coronal_dpi = Float(125.)
    coronal_size = List(Float)
    def _coronal_size_default(self):
        return [450., 450.]

    _snapping_completed = Bool(False, )
    deformation_constant = Float(1.)
    sa_steps_break = Int(2500)
    sa_steps_total = Int(2500)
    sa_init_temp = Float(1e-3)
    sa_exp = Float(1.)

    #state-storing interactive labeling windows
    ews = Dict(transient=True) #str -> Instance(HasTraits)
    alw = Instance(HasTraits, transient=True)
    #raw = Instance(HasTraits)
    
    panel2d = Instance(HasTraits, transient=True)
    _cursor_tracker = Instance(Electrode, transient=True)
    
    def _create_default_name_holders(self):
    #    return self._get__grid_named_objects()
        name_holders = [NameHolder(name=''), 
            GeometryNameHolder(name='unsorted',
                geometry='n/a',
                #TODO dont totally hardcode this color
                color=mayavi2traits_color((1,0,0)))]

        # put additional grids in
        if self._colors is not None:
            for key in self._colors.keys():
                if key in ('unsorted','selection'):
                    continue
                name_holders.append( self._new_grid_name_holder(key) )

        return name_holders

    def _rebuild_interactive_mode_displayer(self, previous_holder=None):
        if previous_holder is not None:
            holder_index = self.interactive_mode_displayer.name_holders.index(
                previous_holder)

        self.interactive_mode_displayer = NameHolderDisplayer()

        self.interactive_mode_displayer.name_holders = (
            self._create_default_name_holders())

        #self.interactive_mode_displayer.interactive_mode = (
        #    self.interactive_mode)

        if previous_holder is not None:
            self.interactive_mode_displayer.interactive_mode = (
                self.interactive_mode_displayer.name_holders[holder_index])

        self._rebuild_guipanel_event = True

    #grid named objects is broken, does not update on reload and does not
    #fully update on grid adding
    #@cached_property
#    def _get__grid_named_objects(self):
#        from color_utils import mayavi2traits_color
#        grid_names = [NameHolder(name=''), 
#            GeometryNameHolder(name='unsorted',
#                geometry='n/a',
#                #TODO dont totally hardcode this color
#                color=mayavi2traits_color((1,0,0)))]
#
#        #for key in sorted(self._grids.keys()):
#        #use the canonical order as the order to appear in the list
#        if self._colors is not None:
#            for key in self._colors.keys():
#                if key in ('unsorted','selection'):
#                    continue
#                grid_names.append(GeometryNameHolder(
#                    name=key, 
#                    geometry=str(self._grid_geom[key]), 
#                    color=mayavi2traits_color(self._colors[key])))
#
#        #if len(self._grids) > 0:
#        #import pdb
#        #pdb.set_trace()
#
#        return grid_names

    def _new_grid_name_holder(self, key):
        gnh = GeometryNameHolder(
            name=key,
            previous_name=key,
            geometry=str(self._grid_geom[key]),
            color=mayavi2traits_color(self._colors[key])) 

        #gnh.on_trait_change(partial(self._change_grid_name, gnh),name='name')
        gnh.on_trait_change( lambda:self._change_grid_name(gnh), name='name')

        return gnh

    def _change_grid_name(self, holder):
        self._commit_grid_changes()

        import copy
        old_name = holder.previous_name
        new_name = holder.name

        if old_name == new_name:
            return

        if new_name in self._grids:
            error_dialog('That name already exists. Pick a different one')  
            return

        if new_name == '':
            error_dialog('Please specify a non-empty name')

        self._grids[new_name] = copy.copy(self._grids[old_name])
        del self._grids[old_name]

        #make sure colors as ordereddict stays in correct order
        #otherwise visualizations will get confused
        from collections import OrderedDict

        colors_index = self._colors.keys().index(old_name)
        new_colors_dict = OrderedDict()
        for i, key in enumerate(self._colors):
            if i == colors_index:
                new_colors_dict[new_name] = self._colors[key]
            else:
                new_colors_dict[key] = self._colors[key]
        self._colors = new_colors_dict 

        self._grid_geom[new_name] = copy.copy(self._grid_geom[old_name])
        del self._grid_geom[old_name]

        self._grid_types[new_name] = copy.copy(self._grid_types[old_name])
        del self._grid_types[old_name]

        for electrode in self._grids[new_name]:
            electrode.grid_name = new_name

        holder.previous_name = new_name

        #self._update_glyph_lut_event = True

        self._rebuild_interactive_mode_displayer(previous_holder=holder)

    def _interactive_mode_changed(self):
        self._commit_grid_changes()

    @on_trait_change('interactive_mode_displayer:_mode_changed_event')
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
                    self._iso_to_grid_ident_map[ intize(elec.asiso()) ] = ( 
                        'unsorted')
                elif old=='unsorted':
                    del self._unsorted_electrodes[ intize(elec.asiso()) ]

                if new not in ('','unsorted','selection'):
                    self._grids[new].append(elec)
                    self._iso_to_grid_ident_map[ intize(elec.asiso()) ] = new
                elif new=='unsorted':
                    self._unsorted_electrodes[ intize(elec.asiso()) ] = elec

        self._points_to_cur_grid = {}
        self._points_to_unsorted = {}
    
    def acquire_affine(self):
        import pipeline as pipe

        overwrite = self.overwrite_xfms and not self._visualization_ready

        if self.ct_registration not in (None, ''):
            aff = load_affine(self.ct_registration)

        elif self.registration_procedure == 'experimental shape correction':
            aff = pipe.register_ct_using_zoom_correction(
                self.ct_scan, subjects_dir=self.subjects_dir,
                subject=self.subject, overwrite=overwrite,
                zf_override=self.zoom_factor_override)

        elif self.registration_procedure == 'uncorrected MI registration':
            aff = pipe.register_ct_to_mr_using_mutual_information(
                self.ct_scan, subjects_dir=self.subjects_dir, 
                subject=self.subject, overwrite=overwrite)

        #recommend that the user never do this
        elif self.registration_procedure == 'no registration':
            #aff = np.eye(4)

            aff = pipe.get_rawavg_to_orig_xfm(
                subjects_dir=self.subjects_dir,
                subject=self.subject)

            #aff = np.array(((-1., 0., 0., 128.),
            #                (0., 0., 1., -128.),
            #                (0., -1., 0., 128.),
            #                (0., 0., 0., 1.)))

            #aff = np.array(((1., 0., 0., 0.),
            #                (0., 0., -1., 0.),
            #                (0., 1., 0., 0.),
            #                (0., 0., 0., 1.)))

            from scipy.linalg import inv
            aff = inv(aff)

        else:
            raise ValueError("Bad registration procedure type")

        return aff

    def run_pipeline(self):
        #setup
        if self.subjects_dir is None or self.subjects_dir=='':
            self.subjects_dir = os.environ['SUBJECTS_DIR']
        if self.subject is None or self.subject=='':
            self.subject = os.environ['SUBJECT']

        #get rid of any existing grid changes
        self._commit_grid_changes()
        self._grids = {}
        self._grid_types = {}
        
        #manually handle property


        self._electrodes = []
        self._all_electrodes = {}
        self._unsorted_electrodes = {}
        self._sorted_electrodes = {}
        self._interpolated_electrodes = {}

        self._iso_to_surf_map = {}
        self._surf_to_iso_map = {}

        self._snapping_completed = False
        self._noise_hidden = False
        self._visualization_ready = False

        #pipeline
        import pipeline as pipe
        
        #if self.use_ct_mask:
        if False:
            ct_mask = pipe.create_brainmask_in_ctspace(self.ct_scan,
                subjects_dir=self.subjects_dir, subject=self.subject,
                overwrite=self.overwrite_xfms)
        else:
            ct_mask = None

        self._electrodes = pipe.remove_large_negative_values_from_ct(
            self.ct_scan, subjects_dir=self.subjects_dir, 
            subject=self.subject)

        self._electrodes = pipe.identify_electrodes_in_ctspace(
            self.ct_scan, mask=ct_mask, threshold=self.ct_threshold,
            use_erosion=(not self.disable_erosion),
            isotropization_type=self.isotropize,
            iso_vector_override=self.isotropization_override) 

        pipe.linearly_transform_electrodes_to_isotropic_coordinate_space(
            self._electrodes, self.ct_scan, 
            isotropization_direction_off = 'copy_to_iso',
            isotropization_direction_on = 'deisotropize',
            isotropization_strategy = self.isotropize,
            iso_vector_override=self.isotropization_override)

        self._electrodes = np.unique(self._electrodes).tolist()

        #I considered allowing the user to manually specify a different
        #registration but we don't currently do this
        aff = self.acquire_affine()

        pipe.create_dural_surface(subjects_dir=self.subjects_dir, 
            subject=self.subject)

        # make the electrodes translated way outside of the brain go away
        # and not be included in the classification at all
        pipe.translate_electrodes_to_surface_space(
            self._electrodes, aff, subjects_dir=self.subjects_dir,
            subject=self.subject)

        if self.use_ct_mask:
            print 'eliminating extracranial noise'
            removals=(
                pipe.identify_extracranial_electrodes_in_freesurfer_space(
                self._electrodes, 
                dilation_iterations=self.dilation_iterations,
                subjects_dir=self.subjects_dir, subject=self.subject))

            print 'removed %i electrodes' % len(removals)

            for e in removals:
                self._electrodes.remove(e)

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
                                         rho_loose = self.rho_loose,
                                         crit_pct = self.critical_percentage
                                        ))
        except ValueError as e:
            error_dialog(str(e))
            raise

        # add grid labels to electrodes
        for key in self._grids:
            for elec in self._grids[key]:
                elec.grid_name = key

        # add interpolated points to overall list
        interps = []
        for key in self._grids:
            for elec in self._grids[key]:
                if elec.is_interpolation:
                    interps.append(elec)

        self._electrodes.extend(interps)

        # add isotropization for interpolated points
        pipe.linearly_transform_electrodes_to_isotropic_coordinate_space(
            interps, self.ct_scan,
            isotropization_direction_off = 'copy_to_ct',
            isotropization_direction_on = 'deisotropize',
            isotropization_strategy = self.isotropize,
            iso_vector_override = self.isotropization_override)

        # translate any new electrodes to surface space as well
        pipe.translate_electrodes_to_surface_space(
            self._electrodes, aff, subjects_dir=self.subjects_dir,
            subject=self.subject)

        # make the electrodes translated way outside of the brain go away
        if ct_mask:
            removals=(
                pipe.identify_extracranial_electrodes_in_freesurfer_space(
                    self._electrodes))

            for e in removals:
                self._electrodes.remove(e)
            
        # previously we snapped here

        # Store the sorted/interpolated points in separate maps for access
        for key in self._grids:
            for elec in self._grids[key]:
                if elec.is_interpolation:
                    self._interpolated_electrodes[ intize(elec.asiso()) ]=elec
                else:
                    self._sorted_electrodes[ intize( elec.asiso()) ] = elec

                #save each electrode's grid identity
                self._iso_to_grid_ident_map[ intize( elec.asiso()) ] = key


        #set the grid type to be subdural
        #any depth grids will not have been created by now since they have
        #a geometry
        for key in self._grids:
            self._grid_types[key] = 'subdural'

        # store the unsorted points in a separate map for access
        for elec in self._electrodes:
            is_sorted = False
            for key in self._grids:
                if is_sorted:
                    break
                for elec_other in self._grids[key]:
                    if elec is elec_other:
                        is_sorted = True
                        break
            if not is_sorted:
                self._unsorted_electrodes[ intize(elec.asiso()) ] = elec

                self._iso_to_grid_ident_map[intize(elec.asiso())] = 'unsorted'

        self._all_electrodes.update(self._interpolated_electrodes)
        self._all_electrodes.update(self._unsorted_electrodes)
        self._all_electrodes.update(self._sorted_electrodes)
    
        #save mapping from ct to surface coords
        for elec in self._all_electrodes.values():
            #snapping is never completed by this point anymore
            surf_coord = elec.asras()

            self._iso_to_surf_map[ intize(elec.asiso()) ] = surf_coord
            self._surf_to_iso_map[ intize(surf_coord) ] = elec.asiso()

        #manually trigger a change to grid_named_objects property
        #using an unlikely grid name
        #self._grids['test rice-a-roni'] = []
        #del self._grids['test rice-a-roni']

        self.interactive_mode_displayer.name_holders = (
            self._create_default_name_holders())
        
        self.interactive_mode_displayer.interactive_mode = (
            self.interactive_mode_displayer.name_holders[0])

        self._cursor_tracker = None

        #manually add the new grids to grid_named_objects
        #for key in self._grids:
            #self._grid_named_objects.append( self._new_grid_name_holder(key))
        #    self.interactive_mode_displayer.name_holders.append(
        #        self._new_grid_name_holder(key))

        #fire visualization events
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
        self._grid_types[name] = 'depth'

        #testing GUI update bug
        #temp_grids = self._grids.copy()
        #temp_grids[name] = []
        #self._grids = temp_grids

        self._grids[name] = []
        #self._grid_named_objects = self._get__grid_named_objects()
        #self.interactive_mode = self._grid_named_objects[0]
        #self._grid_named_objects.append( self._new_grid_name_holder( name ))
        self.interactive_mode_displayer.name_holders.append(
            self._new_grid_name_holder(name))
        
        self._update_glyph_lut_event = True

        self._rebuild_interactive_mode_displayer(previous_holder=
            self.interactive_mode_displayer.interactive_mode)

    def add_electrode_to_grid(self, elec, target):
        self._grids[target].append(elec)

        self._iso_to_surf_map[intize(elec.asiso())] = elec.asras()
        self._surf_to_iso_map[intize(elec.asras())] = elec.asiso()

        self._iso_to_grid_ident_map[ intize(elec.asiso()) ] = target
        self._interpolated_electrodes[ intize(elec.asiso()) ] = elec
        self._all_electrodes[ intize(elec.asiso()) ] = elec

        self._rebuild_vizpanel_event = True
        elec.special_name = ''

    def change_single_glyph(self, xyz, elec, target, current_key):
        if elec in self._grids[target]:
            if intize(xyz) in self._points_to_unsorted:
                del self._points_to_unsorted[ intize(xyz) ]
                elec.grid_transition_to = ''
                self._new_glyph_color = self._colors.keys().index(current_key)
            else:
                self._points_to_unsorted[ intize(xyz) ] = elec
                elec.grid_transition_to = 'unsorted'
                self._new_glyph_color = self._colors.keys().index('unsorted')
        else:
            if intize(xyz) in self._points_to_cur_grid:
                del self._points_to_cur_grid[ intize(xyz) ]
                elec.grid_transition_to = ''
                self._new_glyph_color = self._colors.keys().index(current_key)
            else:
                self._points_to_cur_grid[ intize(xyz) ] = elec
                elec.grid_transition_to = target
                self._new_glyph_color = self._colors.keys().index(target)

        self._single_glyph_to_recolor = xyz
        self._update_single_glyph_event = True

    def hide_noise(self):
        self._commit_grid_changes()

        if self._noise_hidden:
            #self._update_glyph_lut_event = True
            self._noise_hidden = False
            self._rebuild_vizpanel_event = True
        else:
            #self._hide_noise_event = True
            self._noise_hidden = True
            self._rebuild_vizpanel_event = True

        self._draw_event = True

#    def open_registration_window(self):
#        cur_grid = self.interactive_mode_displayer.interactive_mode
#        #TODO WHY IS REGISTRATION DONE BY GRIDS? Shouldnt we adjust the
#        #entire image registration?
#        if cur_grid is None:
#            error_dialog('Registration is done gridwise')
#            return
#        if cur_grid.name in ('', 'unsorted'):
#            error_dialog('Regsitration is done gridwise')
#            return
#
#        from utils import RegistrationAdjustmentWindow
#        self.raw = RegistrationAdjustmentWindow(
#            model = self,
#            cur_grid = cur_grid.name)
#        self.raw.edit_traits()
#
#    # this an operation for the manual registration window
#    def reorient_glyph(self, target, matrix):
#        self._commit_grid_changes()
#
#        old_locs = map(lambda x:x.asras(), self._grids[target])
#
#        from geometry import apply_affine
#        new_locs = apply_affine(old_locs, matrix)
#
#        for oloc, nloc in zip(old_locs, new_locs):
#            self._new_ras_positions[tuple(oloc)] = tuple(nloc)
#
#        self._interactive_mode_snapshot = target
#
#        #draw the changes, no underlying state has yet been altered
#        self._reorient_glyph_event = True
#
#        #for unknown reason the operation causes the LUT to be destroyed
#        #we can recreate it easily
#        self._update_glyph_lut_event = True
#
#        #import pdb
#        #pdb.set_trace()
#
#        for oloc,nloc,elec in zip(old_locs, new_locs, self._grids[target]):
#
#            if tuple(oloc) == tuple(nloc):
#                continue
#
#            elec.surf_coords = tuple(nloc)
#            self._iso_to_surf_map[elec.asiso()] = tuple(nloc)
#            del self._surf_to_iso_map[tuple(oloc)]
#            self._surf_to_iso_map[tuple(nloc)] = elec.asiso()
#
#    def fit_changes(self):
#        #maybe this should be a different call which evaluates a single
#        #grid
#
#        #ceurrently we dont use this
#        _, _, self._grids = pipe.classify_electrodes(
#            self._electrodes, self.electrode_geometry,
#            fixed_points=self._grids.values())

    def examine_electrodes(self):
        self._commit_grid_changes()
        cur_grid = self.interactive_mode_displayer.interactive_mode

        if cur_grid is None:
            error_dialog('Select a grid to assign labels')
            return
        if cur_grid.name in ('','unsorted'):
            error_dialog('Select a grid to assign labels')
            return

        if cur_grid.name in self.ews:
            error_dialog('This window is already open')
            return

        #from utils import AutomatedAssignmentWindow
        from electrode import ElectrodeWindow
        #self.ew = AutomatedAssignmentWindow(
        grid_type = self._grid_types[cur_grid.name]
        ew = ElectrodeWindow(
            model = self,
            cur_grid = cur_grid.name,
            name_stem = cur_grid.name,
            electrodes = self._grids[cur_grid.name],
            grid_type = grid_type,
            naming_convention = ('line' if grid_type == 'depth' 
                else 'grid_serial'),
            )
        self.ews[cur_grid.name] = ew
        ew.edit_traits()

    def open_add_label_window(self):
        if self.alw is None:
            from utils import AddLabelsWindow
            self.alw = AddLabelsWindow(model=self)
        self.alw.edit_traits()

#    def reconstruct_all_geometry(self):
#        import pipeline as pipe
#
#        for key in self._grids:
#            pipe.classify_single_fixed_grid(key, self._grids, self._grid_geom,
#                self._colors, 
#                delta=self.delta, 
#                epsilon=self.epsilon,
#                rho=self.rho, 
#                rho_loose=self.rho_loose,
#                rho_strict=self.rho_strict)

    def snap_all(self):
        self._commit_grid_changes()

        import pipeline as pipe

        snappable_electrodes = []

        for key in self._grids.keys():
            if self._grid_types[key] != 'subdural':
                continue

            snappable_electrodes.extend(self._grids[key])

        if len(snappable_electrodes) == 0:
            error_dialog("Found no subdural electrodes to snap")
            return

        pipe.snap_electrodes_to_surface(
            snappable_electrodes, subjects_dir=self.subjects_dir,
            subject=self.subject, 
            max_steps=self.sa_steps_total,
            giveup_steps=self.sa_steps_break,
            init_temp=self.sa_init_temp,
            temperature_exponent=self.sa_exp,
            )

        self._snapping_completed = True

        #update CT to surf mappings for clickability
        for key in self._grids.keys():
            if self._grid_types[key] == 'subdural':
                for elec in self._grids[key]: 
                    
                    # we snap to the pial surface in terms of returning results
                    # which is pial_coords. but we use the dural surface
                    # coords *specifically for visualization only*.
                    # so the map which is for visualization effects translating
                    # between the two has to use the dural coordinate.
    
                    #snap_coord = elec.astuple()
                    snap_coord = elec.snap_coords
                    surf_coord = elec.asras()
                    iso_coord = elec.asiso()

                    self._iso_to_surf_map[ intize(iso_coord) ] = snap_coord
                    self._surf_to_iso_map[ intize(snap_coord) ] = iso_coord
                    #TODO manage collisions in ct_to_surf mapping
                    
                    # could have been snapped before (in which case we should also get rid of the previous snapped coord in this map)
                    try:
                        del self._surf_to_iso_map[ intize(surf_coord) ]
                    except KeyError:
                        pass

        # we can update the visualization now
        self._rebuild_vizpanel_event = True
    
    def construct_panel2d(self):
        def build_panel():
            import panel2d
            self.panel2d = pd = panel2d.TwoDimensionalPanel()
            #pd.on_trait_change( self._create_new_electrode, 
            #    'add_electrode_event')
            pd.load_img(os.path.join(get_subjects_dir(subject=self.subject,
                subjects_dir=self.subjects_dir), 'mri', 'orig.mgz'), 
                image_name='t1')
            pd.load_img(self.ct_scan, image_name='ct')    

        if self.panel2d is None:
            build_panel()
        elif len(self.panel2d.images) < 2:
            expected_images = 0
            if self.subject and self.subjects_dir:
                expected_images += 1
            if self.ct_scan:
                expected_images += 1
            if expected_images == 2:
                build_panel()

        return self.panel2d

    def move_electrode(self, elec, new_coords, in_ras=False,
        as_postprocessing=False):
        '''
        Move the electrode to fix errors, or for postprocessing visualization.
        
        This is done when the user goes into the electrode window, selects an
        electrode, and clicks on the menu item "move this electrode."

        If done as postprocessing, ROIs are not reset and snapped coordinates
        are updated to the new values from the RAS (even if these RAS values 
        were just calculated

        This eliminates the effect of snapping and usually should be done
        before snapping, also resets ROIs.
        '''

        #remove old electrode position from dictionary data structures
        del self._iso_to_surf_map[ intize(elec.asiso()) ]
        del self._surf_to_iso_map[ intize(elec.asras()) ]

        target_grid = self._iso_to_grid_ident_map[ intize(elec.asiso()) ]
        del self._iso_to_grid_ident_map[ intize(elec.asiso()) ]
        del self._all_electrodes[ intize(elec.asiso()) ]

        if not in_ras:
            elec.ct_coords = new_coords
            aff = self.acquire_affine()

            import pipeline as pipe
            pipe.translate_electrodes_to_surface_space( [elec], aff,
                subjects_dir=self.subjects_dir, subject=self.subject) 
        else:
            #leave CT alone
            elec.surf_coords = new_coords

        new_ras_coords = elec.asras()

        if as_postprocessing:
            if self._snapping_completed:
                elec.snap_coords = new_ras_coords
                elec.pial_coords = new_ras_coords
        else:
            self._snapping_completed = False
            elec.snap_coords = None
            elec.pial_coords = None
            elec.roi_list = []

        #repopulate dictionaries with new updated electrode
        self._iso_to_surf_map[ intize( elec.asiso()) ] = elec.asras()
        self._surf_to_iso_map[ intize( elec.asras()) ] = elec.asiso()

        self._iso_to_grid_ident_map[ intize( elec.asiso() )] = target_grid
        self._all_electrodes[ intize(elec.asiso()) ] = elec

        self._rebuild_vizpanel_event = True

    @on_trait_change('panel2d:add_electrode_event')
    def _create_new_electrode(self):
        self._commit_grid_changes()

        if len(self._all_electrodes) == 0:
            error_dialog('No electrodes loaded')
            return

        pd = self.panel2d
        image_name = pd.currently_showing.name
        px, py, pz = pd.cursor

        #px,py,pz,_ = pd.pins[image_name][pd.current_pin]

        from electrode import Electrode
        elec = Electrode(ct_coords = (px,py,pz))

        if image_name == 't1':
            #rx,ry,rz = pd.map_cursor((px,py,pz), pd.images['t1'][2])
            error_dialog('Adding electrodes only allowed from CT reference')
            return
    
        elif image_name == 'ct':
            aff = self.acquire_affine()
            import pipeline as pipe

            pipe.translate_electrodes_to_surface_space( [elec], aff,
                subjects_dir=self.subjects_dir, subject=self.subject)

            pipe.linearly_transform_electrodes_to_isotropic_coordinate_space(
                [elec], self.ct_scan,
                isotropization_direction_off = 'copy_to_iso',
                isotropization_direction_on = 'isotropize',
                isotropization_strategy = self.isotropize,
                iso_vector_override = self.isotropization_override)
        else:
            raise ValueError("Internal error: bad image type")

        #self._grids['unsorted'] = elec
        #self._grids['unsorted'].append(elec)

        self._iso_to_surf_map[ intize(elec.asiso()) ] = elec.asras()
        self._surf_to_iso_map[ intize(elec.asras()) ] = elec.asiso()

        self._iso_to_grid_ident_map[ intize(elec.asiso()) ] = 'unsorted'
        self._all_electrodes[ intize(elec.asiso()) ] = elec
        self._unsorted_electrodes[ intize( elec.asiso()) ] = elec

        self._rebuild_vizpanel_event = True

    @on_trait_change('panel2d:untrack_cursor_event')
    def _remove_tracked_cursor(self):
        self._cursor_tracker = None

    @on_trait_change('panel2d:track_cursor_event')
    def _add_tracked_cursor(self):
        self._commit_grid_changes()

        #if there is already a tracker and the user hits this button, it
        #becomes an undo function. remove the existing tracker and exit
        if self._cursor_tracker is not None:
            self._cursor_tracker = None
            self._rebuild_vizpanel_event = True
            return

        pd = self.panel2d
        image_name = pd.currently_showing.name
        px, py, pz = pd.cursor

        if image_name == 't1':
            error_dialog('Cursor can only be tracked from CT reference')
            return

        elif image_name == 'ct':
            elec = Electrode(ct_coords = (px,py,pz))
            self._cursor_tracker = elec
            
            aff = self.acquire_affine()
            import pipeline as pipe
            pipe.linearly_transform_electrodes_to_isotropic_coordinate_space(
                [elec], self.ct_scan,
                isotropization_direction_off = 'copy_to_iso',
                isotropization_direction_on = 'isotropize',
                isotropization_strategy = self.isotropize,
                iso_vector_override = self.isotropization_override)

            pipe.translate_electrodes_to_surface_space( [elec], aff,
                subjects_dir=self.subjects_dir, subject=self.subject)

        else:
            raise ValueError("Internal error: bad image type")

        self._rebuild_vizpanel_event = True

    @on_trait_change('panel2d:panel2d_closed_event')
    def _removed_tracked_cursor(self):
        self._commit_grid_changes()

        if self._cursor_tracker is not None:
            self._rebuild_vizpanel_event = True

    def _ask_user_for_savefile(self, title=None):
        from utils import ask_user_for_savefile
        return ask_user_for_savefile(title=title)

    def _ask_user_for_loadfile(self, title=None):
        from utils import ask_user_for_loadfile
        return ask_user_for_loadfile(title=title)

    def add_annotation(self, annot_name, hemi='both', border=True, opacity=1.):
        self._label_file = annot_name
        self._label_borders = border
        self._label_opacity = opacity
        self._label_hemi = hemi
        self._add_annotation_event = True

    def add_label(self, label_file, border=True, opacity=1., color='blue'):
        self._label_file = label_file
        self._label_borders = border
        self._label_opacity = opacity
        self._label_color = color
        self._add_label_event = True

    def remove_labels(self):
        self._remove_labels_event = True

    def get_electrodes_from_grid(self, target=None, electrodes=None):
        if target is None:
            self._commit_grid_changes()
            if self.interactive_mode_displayer.interactive_mode is None:
                print "select a grid to save labels from"
                return
            target = self.interactive_mode_displayer.interactive_mode.name
            if target in ('unsorted',):
                print "select a grid to save labels from"
                return

        #for now only save label files for the current grid, may change
        #in principle this is not what we want

        # get electrodes from current state if not passed as argument
        key = target
        if electrodes is None:
            key = self.interactive_mode_displayer.interactive_mode.name
            electrodes = self._grids[key]

        return sorted(electrodes)

    def get_electrodes_all(self):
        return sorted(filter(lambda e:e.grid_name != 'unsorted', 
            self._all_electrodes.values()))

class ExtractionRegistrationSortingPanel(HasTraits):
    model = Instance(ElectrodePositionsModel)

    ct_threshold = DelegatesTo('model')
    critical_percentage = DelegatesTo('model')
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
    sa_steps_break = DelegatesTo('model')
    sa_steps_total = DelegatesTo('model')
    sa_init_temp = DelegatesTo('model')
    sa_exp = DelegatesTo('model')
    deformation_constant = DelegatesTo('model')
    use_ct_mask = DelegatesTo('model')
    disable_erosion = DelegatesTo('model')
    overwrite_xfms = DelegatesTo('model')
    registration_procedure = DelegatesTo('model')
    shapereg_slice_diff = DelegatesTo('model')
    zoom_factor_override = DelegatesTo('model')
    dilation_iterations = DelegatesTo('model')
    isotropize = DelegatesTo('model')
    isotropization_override = DelegatesTo('model')

    traits_view = View(
        Group(
        HGroup(
        VGroup(
            Label('The threshold above which electrode clusters will be\n'
                'extracted from the CT image'),
            Item('ct_threshold'),
            Label('Weight given to the deformation term in the snapping\n'
                'algorithm, reduce if snapping error is very high.'),
            Item('deformation_constant'),
            Label('Mask extracranial noise'),
            HGroup(
                Item('use_ct_mask', show_label=False),
                Item('dilation_iterations', show_label=True, 
                    label='iterations', enabled_when='use_ct_mask'),
            ),
            Label('Overwrite existing transformations'),
            Item('overwrite_xfms'),
            Label('Disable binary erosion procedure to reduce CT noise'),
            Item('disable_erosion'),
            HGroup(
                VGroup(
                Label('Type of registration'),
                Item('registration_procedure', show_label=False),
                ),
                VGroup(
                Label('Override zoom factor'),
                Item('zoom_factor_override', editor =CSVListEditor(),
                enabled_when='registration_procedure==\'experimental shape '
                'correction\'', show_label=False),
                ),
            ),
            Label('Convert electrode locations to isotropic coordinate '
                'space\nbefore sorting'),
            HGroup(
                Item('isotropize', show_label=False),
                Item('isotropization_override', editor=CSVListEditor(),
                    enabled_when='isotropize==\'Manual override\'',
                    show_label=False),
            ),
        show_labels=False),
        VGroup(
            Label('The percentage of electrodes to find in sorting'),
            Item('critical_percentage'),
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
            Label('Simulated annealing parameters'),
            HGroup(
                Item('sa_steps_break', label='steps before convergence'),
                Item('sa_steps_total', label='steps total'),
            ),
            HGroup(
                Item('sa_init_temp', label='initial temperature'),
                Item('sa_exp', label='exponential term'),
            ),
        ),
        ),
        ),
        
    title='Adjust extraction registration and sorting parameters',
    buttons=OKCancelButtons,
    )

class VisualizationsOutputsPanel(HasTraits):
    model = Instance(ElectrodePositionsModel)

    roi_parcellation = DelegatesTo('model')
    roi_error_radius = DelegatesTo('model')

    coronal_dpi = DelegatesTo('model')
    coronal_size = DelegatesTo('model')

    traits_view = View(
        HGroup(
        VGroup(
            Label('Identification of ROIs near electrodes'),
            Item('roi_parcellation'),
            Item('roi_error_radius'),
        ),
        VGroup(
            Label('Coronal slices'),
            Item('coronal_dpi', label='dpi'),
            Item('coronal_size', label='size', editor=CSVListEditor()),
        ),
        ),
        
    title='Adjust clinical visualization and output parameters',
    buttons=OKCancelButtons,
    )

class SurfaceVisualizerPanel(HasTraits):
    scene = Instance(MlabSceneModel,())
    model = Instance(ElectrodePositionsModel)

    subject = DelegatesTo('model')
    subjects_dir = DelegatesTo('model')
    _colors = DelegatesTo('model')

    _grids = DelegatesTo('model')
    #interactive_mode = DelegatesTo('model')
    interactive_mode_displayer = DelegatesTo('model')

    _points_to_unsorted = DelegatesTo('model')
    _points_to_cur_grid = DelegatesTo('model')

    _all_electrodes = DelegatesTo('model')
    _unsorted_electrodes = DelegatesTo('model')
    _iso_to_surf_map = DelegatesTo('model')
    _surf_to_iso_map = DelegatesTo('model')
    _grid_types = DelegatesTo('model')

    _cursor_tracker = DelegatesTo('model')

    visualize_in_ctspace = Bool(False)
    _viz_coordtype = Property#(depends_on='visualize_in_ctspace')
    def _get__viz_coordtype(self):
        if self.visualize_in_ctspace:
            #return 'ct_coords'
            return 'iso_coords'
        elif self.model._snapping_completed:
            #plot points on dural surface better for visualization than pial
            return 'snap_coords'
            #return 'pial_coords'
        else:
            return 'surf_coords'

    brain = Any
    gs_glyphs = Dict
    tracking_glyph = Any

    _lh_pysurfer_offset = Float
    _rh_pysurfer_offset = Float
    def __lh_pysurfer_offset_default(self):
        if self.visualize_in_ctspace:
            return 0
        lh_pia, _ = nib.freesurfer.read_geometry(
            os.path.join(self.subjects_dir, self.subject, 'surf', 'lh.pial'))
        return np.max(lh_pia[:, 0])
    def __rh_pysurfer_offset_default(self):
        if self.visualize_in_ctspace:
            return 0
        rh_pia, _ = nib.freesurfer.read_geometry(
            os.path.join(self.subjects_dir, self.subject, 'surf', 'rh.pial'))
        return np.min(rh_pia[:, 0])

    traits_view = View(
        Item('scene', editor=SceneEditor(scene_class=MayaviScene),
            show_label=False, resizable=True),
        height=500, width=500, resizable=True)

    def __init__(self, model, **kwargs):
        super(SurfaceVisualizerPanel, self).__init__(**kwargs)
        self.model = model

    @on_trait_change('scene:activated')
    def setup(self):
        #import pdb
        #pdb.set_trace()
        if self.model._visualization_ready:
            #print self.model._cursor_tracker
            #print self._cursor_tracker
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
                srf._geo_surf.actor.property.opacity = (
                    self.model.surface_opacity)

            scale_factor = 3.
        else:
            scale_factor = 5.

        def fix_elec( elec, coordtype=None ):
            init_coord = np.array(getattr(elec, coordtype))

            if init_coord[0] < 0: 
                init_coord[0] -= self._lh_pysurfer_offset
            else:
                init_coord[0] -= self._rh_pysurfer_offset
            return init_coord

        unsorted_coordtype = (self._viz_coordtype if 
            self._viz_coordtype not in ('pial_coords', 'snap_coords')
            else 'surf_coords')

        #unsorted
        if not self.model._noise_hidden:
        
            unsorted_elecs = map(
                partial(fix_elec, coordtype=unsorted_coordtype),
                 self._unsorted_electrodes.values() )
            self.gs_glyphs['unsorted'] = glyph = virtual_points3d( 
                unsorted_elecs, scale_factor=scale_factor, name='unsorted',
                figure=self.scene.mayavi_scene, 
                color=self._colors['unsorted'])  

            set_discrete_lut(glyph, self._colors.values())
            glyph.mlab_source.dataset.point_data.scalars=(
                np.zeros(len(unsorted_elecs)))

        #grids
        for i,key in enumerate(self._grids):

            grid_coordtype = (self._viz_coordtype if
                (self._viz_coordtype not in ('snap_coords', 'pial_coords') or
                 self._grid_types[key]=='subdural') else 
                'surf_coords')

            grid_elecs = map(
                partial(fix_elec, coordtype=grid_coordtype),
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

        #panel tracking
        if self._cursor_tracker is not None:
            tracker_elec = fix_elec(self._cursor_tracker,
                coordtype=unsorted_coordtype)

            tracking_scale_factor = 6. if self.visualize_in_ctspace else 4.

            self.tracking_glyph = virtual_points3d( [tracker_elec], 
                scale_factor=tracking_scale_factor, name='tracking_glyph',
                figure=self.scene.mayavi_scene, 
                color=self._colors['selection'])

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

        if self.interactive_mode_displayer.interactive_mode is None:
            return
        target = self.interactive_mode_displayer.interactive_mode.name
        if target in ('', 'unsorted', 'selection'):
            return

        current_key = None

        for key,nodes in zip(self.gs_glyphs.keys(), self.gs_glyphs.values()):
            if picker.actor in nodes.actor.actors:
                pt = int(picker.point_id/nodes.glyph.glyph_source.
                    glyph_source.output.points.to_array().shape[0])
                #x,y,z = np.around(nodes.mlab_source.points[pt], 4)
                x,y,z = nodes.mlab_source.points[pt]

                #translate from CT to surf coords if necessary
                if not self.visualize_in_ctspace:
                    if x < 0:
                        x += self._lh_pysurfer_offset
                    else:
                        x += self._rh_pysurfer_offset
                    x,y,z = self._surf_to_iso_map[intize((x,y,z)) ]
                    #x,y,z = self._iso_to_surf_map[intize((x,y,z)) ]

                elec = self._all_electrodes[ intize((x,y,z))]

                current_key = elec.grid_name
                break

        #if the user did not click on anything interesting, do nothing
        if current_key is None:
            return

        self.model.change_single_glyph((x,y,z), elec, target, current_key)
    
    @on_trait_change('model:_update_single_glyph_event')
    def update_single_glyph(self):

        #to be safe
        #self.update_glyph_lut()

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
            xyz = np.array( self._iso_to_surf_map[ intize(xyz) ] )
            if xyz[0] < 0:
                xyz[0] -= self._lh_pysurfer_offset
            else:
                xyz[0] -= self._rh_pysurfer_offset

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

    @on_trait_change('model:_draw_event')
    def force_render(self):
        from plotting_utils import force_render as fr
        fr( figure=self.scene.mayavi_scene )

    @on_trait_change('model:_update_glyph_lut_event')
    def update_glyph_lut(self):
        from color_utils import set_discrete_lut
        for glyph in self.gs_glyphs.values():
            set_discrete_lut(glyph, self._colors.values())

    #not used. has the problem of crescent moon shaped electrodes,
    #transparent electrodes are still occlusive despite being transparent
    @on_trait_change('model:_hide_noise_event')
    def hide_unsorted_electrodes(self):
        from color_utils import make_transparent
        for glyph in self.gs_glyphs.values():
            make_transparent(glyph, 0) 

    @on_trait_change('model:_add_annotation_event')
    def add_annotation(self):
        if self.visualize_in_ctspace:
            return
        
        if self.brain is None:
            error_dialog("Run pipeline first")
            return

        #for hemi in ('lh', 'rh'):
        if self.model._label_hemi in ('lh','rh'):
            self.brain.add_annotation(self.model._label_file,
                borders=self.model._label_borders,
                alpha=self.model._label_opacity,
                hemi=self.model._label_hemi,
                remove_existing=True)

        else:
            for hemi in ('lh','rh'):
                self.brain.add_annotation(self.model._label_file,
                    borders=self.model._label_borders,
                    alpha=self.model._label_opacity,
                    hemi=hemi,
                    remove_existing=(hemi=='lh'))
        
    @on_trait_change('model:_add_label_event')
    def add_label(self):
        if self.visualize_in_ctspace:
            return

        if self.brain is None:
            error_dialog('Run pipeline first')
            return

        import mne
        from color_utils import traits2mayavi_color
        self.brain.add_label(self.model._label_file,
            borders=self.model._label_borders,
            alpha=self.model._label_opacity,
            color=traits2mayavi_color(self.model._label_color),
            hemi=mne.read_label(self.model._label_file).hemi )

    @on_trait_change('model:_remove_labels_event')
    def remove_labels(self):
        if self.visualize_in_ctspace or self.brain is None:
            return

        self.brain.remove_labels(hemi='lh')
        self.brain.remove_labels(hemi='rh')
        for annot in self.brain.annot_list:
            annot['surface'].remove()
        self.brain.annot_list=[]
    
    #@on_trait_change('model:_reorient_glyph_event')
    #def update_orientation(self):
        #only do this for surface viz, do not adjust CT electrode positions
    #    if self.visualize_in_ctspace:
    #        return
    #
    #    target = self.model._interactive_mode_snapshot
    #
    #    for glyph in self.gs_glyphs.values():
    #        points = np.array(glyph.mlab_source.dataset.points)
    #
    #        for i,point in enumerate(points):
    #            surf_point = tuple(self.model._surf_to_iso_map[tuple(point)])
    #            if self.model._iso_to_grid_ident_map[surf_point] == target:
    #                points[i] = self.model._new_ras_positions[tuple(point)] 
    #
    #        glyph.mlab_source.dataset.points = points

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

    @on_trait_change('model:surface_opacity')
    def update_surface_opacity(self):
        if not self.visualize_in_ctspace:
            opc = self.model.surface_opacity
            for srf in self.brain.brains:
                srf._geo_surf.actor.property.opacity = opc
            

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
    #interactive_mode = DelegatesTo('model')
    interactive_mode_displayer = DelegatesTo('model')

    add_grid_button = Button('Add new grid')
    add_label_button = Button('Add labels')
    shell = Dict

    save_montage_button = Button('Save montage')
    save_csv_button = Button('Save csv')

    edit_parameters_button = Button('Edit Fitting Parameters')
    hide_noise_button = Button('Hide noise')
    
    reconstruct_vizpanel_button = Button('Rebuild viz')
    examine_electrodes_button = Button('Examine electrodes')
    snap_electrodes_button = Button('Snap electrodes')
    #adjust_registration_button = Button('Adjust registration')
    visualize_ct_button = Button('Examine CT')

    #we retain a reference to easily reference the visualization in the shell
    viz = Instance(SurfaceVisualizerPanel)
    ctviz = Instance(SurfaceVisualizerPanel)

    traits_view = View(
        HGroup(
            VGroup(
                Item('ct_scan'),
                #Item('ct_registration', label='reg matrix\n(optional)')
                #Item('adjust_registration_button', show_label=False),
                #HGroup(
                #    Item('visualize_ct_button', show_label=False),
                #    Item('hide_noise_button', show_label=False),
                #),
            ),
            VGroup(
                Item('electrode_geometry', editor=CustomListEditor(
                    editor=CSVListEditor(), rows=2), ), 
            ), 
            VGroup(
                Item('run_pipeline_button', show_label=False),
                #Item('edit_parameters_button', show_label=False),
                #HGroup(
                #    Item('save_montage_button', show_label=False),
                #    Item('save_csv_button', show_label=False),
                #),
            ),
        ),
        HGroup(
                Item('subjects_dir'),
                Item('subject'),
        ),
        HGroup(
            VGroup(
                #Item('interactive_mode', 
                #    editor=InstanceEditor(name='_grid_named_objects'),
                #    style='custom', label='Edit electrodes\nfrom grid'),
                Item('interactive_mode_displayer',
                    editor=InstanceEditor(), style='custom',
                    label='Edit electrodes\nfrom grid'),
            ),
            VGroup(
                Item('add_grid_button', show_label=False),
                #Item('reconstruct_vizpanel_button', show_label=False),
                #Item('add_label_button', show_label=False)
            #),
            #VGroup(
                Item('examine_electrodes_button', show_label=False),
            #    Item('snap_electrodes_button', show_label=False),
            ),
        ),

                Item('shell', show_label=False, editor=ShellEditor()),

        height=300, width=500, resizable=True
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
#
    def _edit_parameters_button_fired(self):
        ExtractionRegistrationSortingPanel(model=self.model).edit_traits()

    def _hide_noise_button_fired(self):
        self.model.hide_noise() 

    def _add_label_button_fired(self):
        self.model.open_add_label_window()

    def _reconstruct_vizpanel_button_fired(self):
        self.model._reconstruct_vizpanel_event = True

    def _snap_electrodes_button_fired(self):
        self.model.snap_all()

    def _examine_electrodes_button_fired(self):
        self.model.examine_electrodes()

    def _adjust_registration_button_fired(self):
        self.model.open_registration_window()

#    def _visualize_ct_button_fired(self):
#        import panel2d
#        self.model.panel2d = pd = panel2d.TwoDimensionalPanel()
#        #pd.on_trait_change(self.model._create_new_electrode, 
#        #    'add_electrode_event')
#        pd.load_img(self.ct_scan, image_name='ct')
#        pd.edit_traits()

class iEEGCoregistrationFrame(HasTraits):
    model = Instance(ElectrodePositionsModel)
    interactive_panel = Instance(InteractivePanel)
    surface_visualizer_panel = Instance(SurfaceVisualizerPanel)
    ct_visualizer_panel = Instance(SurfaceVisualizerPanel)

    load_pickle_action = Action(name='Load pickle', action='do_load_pickle' )
    save_pickle_action = Action(name='Save pickle', action='do_save_pickle' )
    quit_action = Action(name='Quit', action='do_quit')

    run_pipeline_action = Action( name='Run pipeline', 
        action='do_run_pipeline')
    snap_action = Action( name='Snap electrodes to surface', action='do_snap')
    hide_noise_action = Action( name='Hide noise', action='do_hide_noise')
    examine_ct_action = Action( name='Examine CT', action='do_examine_ct')
    display_pysurfer_labels_action = Action(name='Show labels or annotation',
        action='do_display_pysurfer_labels')

    find_rois_action = Action(name='Find neighboring ROIs', 
        action='do_find_rois')
    coronal_slices_action = Action(name='Save coronal slices (depths)',
        action='do_coronal_slices')
    subdural_opaque_images_action = Action(
        name='Save snapshots (grids)',
        action='do_save_opaque')
    save_montage_action = Action(name='Save montage', 
        action='do_save_montage')
    save_csv_action = Action(name='Save CSV', action='do_save_csv')

    edit_extraction_sorting_parameters_action = Action(
        name = 'Extraction/Registration/Sorting',
        action = 'do_extraction_params')
    edit_visualization_output_parameters_action = Action(
        name = 'Visualization/Output',
        action = 'do_visualization_params')

    traits_view = View(
        Group(
            HGroup(
                Item('surface_visualizer_panel', editor=InstanceEditor(), 
                    style='custom', resizable=True ),
                Item('ct_visualizer_panel', editor=InstanceEditor(),
                    style='custom', resizable=True ),
            show_labels=False, layout='split'),
            Item('interactive_panel', editor=InstanceEditor(), style='custom',
                resizable=True),
        show_labels=False, layout='split'),
        title=('llanfairpwllgwyngyllgogerychwyrndrobwllllantysiliogogogoch is'
            ' nice this time of year'),

        menubar = MenuBar(
            Menu( load_pickle_action,
                  save_pickle_action,
                  #quit_action,
                name = 'File'),
            Menu ( run_pipeline_action,
                   examine_ct_action,
                   hide_noise_action,
                   display_pysurfer_labels_action,
                name = 'Tools'),
            Menu( find_rois_action,
                  snap_action, 
                  coronal_slices_action,
                  subdural_opaque_images_action,
                  save_montage_action,
                  save_csv_action,
                name = 'Electrode'),
            Menu ( edit_extraction_sorting_parameters_action,
                   edit_visualization_output_parameters_action,
                name = 'Settings'),
        ),

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

    def do_save_pickle(self):
        self.model._commit_grid_changes()

        savefile = self.model._ask_user_for_savefile(title='save pkl file')

        from pickle import dump
        with open(savefile, 'w') as fd:
            dump(self.model, fd)

    def do_load_pickle(self):
        loadfile = self.model._ask_user_for_loadfile(title='load pkl file')

        from pickle import load
        try:
            with open(loadfile) as fd:
                self.model = load(fd)
        except (KeyError, AttributeError) as e:
            error_dialog('Failed to load ElectrodePositionsModel object from\n'
                'provided pickle file.\n\n'
                'Are you sure that was a correct pickle file?')
            return

        #color scheme is a generator and therefore can't be imported properly
        #but we can recreate the correct generator state easily enough
        from utils import get_default_color_scheme
        colors = get_default_color_scheme()
        #get rid of the right number of colors
        for i in xrange( len( self.model._grids)):
            colors.next()

        self.model._color_scheme = colors
    
        #the trait notifiers set up on the displayer are not reproduced
        #after pickling since those are not defined in the NameHolder 
        #initializer but by private method _new_grid_name_holder(grid_name)
        #maybe they could be defined in the initializer instead but whatever
        #here we just manually call our tools to rebuild the displayer safely
        self.model._rebuild_interactive_mode_displayer()

        self.model._rebuild_guipanel_event = True
        self.model._rebuild_vizpanel_event = True
    
    #def do_quit(self):
    #    sys.exit(0)
    # user can quit with window manager. makes less likely to misclick and
    # exit accidentally

    def do_run_pipeline(self):
        self.model.run_pipeline()

    def do_snap(self):
        self.model.snap_all()

    def do_examine_ct(self):
        import panel2d

        pd = self.model.construct_panel2d()

        #self.model.panel2d = pd = panel2d.TwoDimensionalPanel()
        #pd.load_img(self.model.ct_scan, image_name='ct')
        pd.edit_traits()

    def do_hide_noise(self):
        self.model.hide_noise() 

        if self.model._noise_hidden:
            self.hide_noise_action.name = 'Unhide noise'
        else:
            self.hide_noise_action.name = 'Hide noise'

    def do_display_pysurfer_labels(self):
        self.model.open_add_label_window()

    def do_find_rois(self):
        self.model._commit_grid_changes()

        from electrode_group import get_nearby_rois_all
        get_nearby_rois_all( self.model._grids,
                             subjects_dir=self.model.subjects_dir,
                             subject=self.model.subject,
                             parcellation=self.model.roi_parcellation,
                             error_radius=self.model.roi_error_radius )

    def do_coronal_slices(self):
        self.model._commit_grid_changes()

        from electrode_group import coronal_slice_all
        coronal_slice_all( self.model._grids,
                           self.model._grid_types,
                           subjects_dir=self.model.subjects_dir,
                           subject=self.model.subject,
                           dpi=self.model.coronal_dpi )

    def do_save_opaque(self):
        self.model._commit_grid_changes()

        savefile = self.model._ask_user_for_savefile()

        if savefile is None:
            return

        self.model.surface_opacity = 1.0
        if not self.model._noise_hidden:
            self.model.hide_noise()

        from plotting_utils import save_opaque_clinical_sequence
        save_opaque_clinical_sequence( savefile, 
            self.surface_visualizer_panel.scene.mayavi_scene )

        self.model.surface_opacity = 0.4
        if self.model._noise_hidden:
            self.model.hide_noise()

    def do_save_montage(self):
        self.model._commit_grid_changes()

        electrodes = self.model.get_electrodes_all()

        from electrode_group import save_coordinates
        save_coordinates( electrodes, self.model._grid_types,
            snapping_completed=self.model._snapping_completed,
            file_type='montage')
    
    def do_save_csv(self):
        self.model._commit_grid_changes()

        electrodes = self.model.get_electrodes_all()

        from electrode_group import save_coordinates
        save_coordinates( electrodes, self.model._grid_types,
            snapping_completed=self.model._snapping_completed,
            file_type='csv')

    def do_extraction_params(self):
        ExtractionRegistrationSortingPanel(model=self.model).edit_traits()

    def do_visualization_params(self):
        VisualizationsOutputsPanel(model=self.model).edit_traits()

    @on_trait_change('model:_rebuild_vizpanel_event')
    def _rebuild_vizpanel(self):

        #throw away old vizpanel listeners to remove annoying error messages
        self.surface_visualizer_panel.model = None
        self.ct_visualizer_panel.model = None

        #create new panels
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

if __name__ == '__main__':
    #force Qt to relay ctrl+C
    from main import main
    main()
