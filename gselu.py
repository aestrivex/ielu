from mayavi.core.ui.api import MayaviScene, SceneEditor, MlabSceneModel
from traits.api import (Bool, Button, cached_property, File, HasTraits,
    Instance, on_trait_change, Str, Property, Directory, Dict, DelegatesTo,
    HasPrivateTraits, Any, List, Enum, Int, Event)
from traitsui.api import (View, Item, Group, OKCancelButtons, ShellEditor,
    HGroup,VGroup, InstanceEditor, TextEditor, ListEditor, CSVListEditor)

class ElectrodePositionsModel(HasPrivateTraits):
    ct_scan = File
    t1_scan = File
    subjects_dir = Directory
    subject = Str
    fsdir_writable = Bool
    hemisphere = Enum('rh','lh')

    electrode_geometry = List(List(Int), [[8,8]]) # Gx2 list

    _ct_electrodes = Any #np.ndarray Nx3
    _surf_electrodes = List #List(np.ndarray Nx3)

    _visualize_event = Event

    color_scheme = Any #None or generator function

    def _run_pipeline(self):
        import pipeline as pipe
        
        ct_mask = pipe.create_brainmask_in_ctspace(self.ct_scan,
            subjects_dir=self.subjects_dir, subject=self.subject)

        self._ct_electrodes = pipe.identify_electrodes_in_ctspace(
            self.ct_scan, mask=ct_mask) 

        grids, colors = pipe.classify_electrodes(self._ct_electrodes, 
            self.electrode_geometry)
        self.color_scheme = colors
        self._surf_electrodes = []

        aff = pipe.register_ct_to_mr_using_mutual_information(self.ct_scan,
            subjects_dir=self.subjects_dir, subject=self.subject)

        pipe.create_dural_surface(subjects_dir=self.subjects_dir, 
            subject=self.subject)

        for key in grids:
            grid_points = grids[key]

            surf_points = pipe.translate_electrodes_to_surface_space(
                grid_points, aff, subjects_dir=self.subjects_dir,
                subject=self.subject)
    
            snapped_points = pipe.snap_electrodes_to_surface(
                surf_points, self.hemisphere, subjects_dir=self.subjects_dir,
                subject=self.subject, max_steps=5000)

            self._surf_electrodes.append(snapped_points)

        self._visualize_event = True

class SurfaceVisualizerPanel(HasTraits):
    scene = Instance(MlabSceneModel,())
    model = Instance(ElectrodePositionsModel)

    _visualize_event = DelegatesTo('model')
    subject = DelegatesTo('model')
    subjects_dir = DelegatesTo('model')
    hemisphere = DelegatesTo('model')
    _surf_electrodes = DelegatesTo('model')
    color_scheme = DelegatesTo('model')

    brain = Any

    traits_view = View(
        Item('scene', editor=SceneEditor(scene_class=MayaviScene),
            show_label=False),
            #),
        height=500, width=500)

    def __init__(self, model, **kwargs):
        super(SurfaceVisualizerPanel, self).__init__(**kwargs)
        self.model = model

    @on_trait_change('model:_visualize_event')
    def show_grids_on_surface(self):
        from mayavi import mlab
        mlab.clf(figure = self.scene.mayavi_scene)

        import surfer
        brain = self.brain = surfer.Brain( 
            self.subject, subjects_dir=self.subjects_dir,
            surf='pial', curv=False, hemi=self.hemisphere,
            figure=self.scene.mayavi_scene)

        brain.brains[0]._geo_surf.actor.property.opacity = 0.35

        for elecs in self._surf_electrodes:
            brain.add_foci( elecs, scale_factor=0.3,
                color=self.color_scheme.next() )

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

    find_best_fit_button = Button('Reconfigure electrode configuration')
    shell = Dict

    viz = Instance(SurfaceVisualizerPanel)

    traits_view = View(
        #VGroup(
            HGroup(
                Item('ct_scan'),
                Item('electrode_geometry', editor=ListEditor(
                    editor=CSVListEditor())),
            ),
            HGroup(
                Item('subjects_dir'),
                Item('subject'),
                Item('hemisphere')
            ),
            HGroup(
                Item('run_pipeline_button', show_label=False),
                Item('find_best_fit_button', show_label=False),
            ),
        #),
            Item('shell', show_label=False, editor=ShellEditor()),
        height=300, width=500
    )

    def __init__(self, model, viz=None, **kwargs):
        super(InteractivePanel, self).__init__(**kwargs)
        self.model = model
        self.viz = viz

    def _run_pipeline_button_fired(self):
        self.model._run_pipeline()

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

