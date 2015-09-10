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

class NameHolder(HasTraits):
    name = Str
    traits_view = View()

    def __str__(self):
        return 'Grid: %s'%self.name

class GeometryNameHolder(NameHolder):
    geometry = Str
    color = Color
    previous_name = Str
    traits_view = View( 
        HGroup(
            Item('name', show_label=False, 
                editor=TextEditor(auto_set=False, enter_set=True), ),
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

class NameHolderDisplayer(Handler):
    name_holders = List(Instance(NameHolder))
    interactive_mode = Instance(NameHolder)
    _mode_changed_event = Event

    @on_trait_change('interactive_mode')
    def fire_event(self):
        self._mode_changed_event = True

    traits_view = View(
        Item('interactive_mode', editor=InstanceEditor(name='name_holders'),
            style='custom', show_label=False),
    )
