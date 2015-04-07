import os
from traits.trait_base import ETSConfig
_tk = ETSConfig.toolkit

from functools import partial
from traits.api import (Property, Instance, Bool, Str, BaseTraitHandler,
    Range)
from traitsui.api import CustomEditor

if _tk in ('null','',None):
    _tk = os.environ['ETS_TOOLKIT']
    #there's no situation other than debugging this file directly
    #where this won't be set. To debug this file, set the environment var.
else:
    print _tk

CustomEditorKlass = __import__('traitsui.%s.custom_editor'%_tk,
    fromlist=['CustomEditor'], ).CustomEditor

def mkeditor(*args, **kwargs):
    return (getattr(__import__('custom_list_editor'),'%s_editor_factory'%_tk)
        (*args,**kwargs))

def wx_editor_factory(*args, **kwargs):
    pass

def qt4_editor_factory(parent, editor, *args, **kwargs):
    from pyface.qt import QtCore, QtGui
    trait_handler = editor.factory.trait_handler
    if trait_handler is None:
        trait_handler = editor.object.base_trait( editor.name ).handler
    editor._trait_handler = trait_handler

    editor.control = panel = QtGui.QScrollArea()
    editor.control.setFrameShape(QtGui.QFrame.NoFrame)
    editor.control.setWidgetResizable(True)

    editor.mapper = QtCore.QSignalMapper(panel)

    editor._list_pane = QtGui.QWidget()
    editor._list_pane.setSizePolicy(QtGui.QSizePolicy.Expanding,
                                  QtGui.QSizePolicy.Expanding)
    layout = QtGui.QGridLayout(editor._list_pane)
    layout.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop) 
    layout.setContentsMargins(0, 0, 0, 0)

    _editor = editor.factory.editor
    if _editor is None:
        _editor = trait_handler.item_trait.get_editor()
    editor._editor = getattr( _editor, editor.kind )

    extended_name = editor.extended_name.replace('.', ':')
    editor.context_object.on_trait_change( editor.update_editor_item,
                           extended_name + '_items?', dispatch = 'ui' )
    editor.set_tooltip()

    return panel

#custom editor on toolkit level
class CustomQtListEditorKlass(CustomEditorKlass):

    from pyface.qt import QtCore, QtGui

    #trait definitions copied from file
    mapper = Instance(QtCore.QSignalMapper)
    mutable = Bool(True)
    kind = Str
    
    single_row = True

    list_menu = """
       Add &Before     [_menu_before]: self.add_before()
       Add &After      [_menu_after]:  self.add_after()
       ---
       &Delete         [_menu_delete]: self.delete_item()
       ---
       Move &Up        [_menu_up]:     self.move_up()
       Move &Down      [_menu_down]:   self.move_down()
       Move to &Top    [_menu_top]:    self.move_top()
       Move to &Bottom [_menu_bottom]: self.move_bottom()
    """

    empty_list_menu = """
       Add: self.add_empty()
    """

    def update_editor(self): 
        from pyface.qt import QtCore, QtGui
        self.mapper = QtCore.QSignalMapper(self.control)
        # Disconnect the editor from any control about to be destroyed:
        self._dispose_items()

        list_pane = self._list_pane
        layout = list_pane.layout()

        # Create all of the list item trait editors:
        trait_handler = self._trait_handler
        resizable     = ((trait_handler.minlen != trait_handler.maxlen) and
                         self.mutable)
        item_trait    = trait_handler.item_trait

        is_fake = (resizable and (len( self.value ) == 0))
        if is_fake:
            self.empty_list()
        else:
            # Asking the mapper to send the sender to the callback method
            self.mapper.mapped[QtCore.QObject].connect(self.popup_menu)

        editor = self._editor
        for index, value in enumerate(self.value):
            row, column = divmod(index, self.factory.columns)

            # Account for the fact that we have <columns> number of
            # pairs
            column = column * 2

            if resizable:
                # Connecting the new button to the mapper
                from traitsui.qt4.helper import IconButton

                from pyface.qt import qt_api

                # fix bug in pyqt, reverts back to traitsui 4.3 version
                if qt_api == 'pyside':
                    control = IconButton('list_editor.png', self.mapper.map)
                elif qt_api == 'pyqt':
                    control = IconButton('list_editor.png',
                        lambda : self.popup_menu(
                            self._list_pane.layout().sender() ))

                # Setting the mapping and asking it to send the index of the
                # sender to the callback method
                self.mapper.setMapping(control, control)

                layout.addWidget(control, row, column)

            from traitsui.editors.list_editor import ListItemProxy
            proxy = ListItemProxy( self.object, self.name, index, item_trait,
                                   value )
            if resizable:
                control.proxy = proxy
            peditor = editor( self.ui, proxy, 'value', self.description,
                              list_pane ).set( object_name = '' )
            peditor.prepare( list_pane )
            pcontrol = peditor.control
            pcontrol.proxy = proxy

            if isinstance(pcontrol, QtGui.QWidget):
                layout.addWidget(pcontrol, row, column+1)
            else:
                layout.addLayout(pcontrol, row, column+1)

        # QScrollArea can have problems if the widget being scrolled is set too
        # early (ie. before it contains something).
        if self.control.widget() is None:
            self.control.setWidget(list_pane)

    def update_editor_item ( self, event ):
        """ Updates the editor when an item in the object trait changes
        externally to the editor.
        """
        # If this is not a simple, single item update, rebuild entire editor:
        from pyface.qt import QtCore, QtGui
        if (len( event.removed ) != 1) or (len( event.added ) != 1):
            self.update_editor()
            return

        # Otherwise, find the proxy for this index and update it with the
        # changed value:
        for control in self.control.widget().children():
            if isinstance(control, QtGui.QLayout):
                continue

            proxy = control.proxy
            if proxy.index == event.index:
                proxy.value = event.added[0]
                break

    def dispose ( self ):
        """ Disposes of the contents of an editor.
        """
        self._dispose_items()

        extended_name = self.extended_name.replace('.', ':')
        self.context_object.on_trait_change( self.update_editor_item,
                                 extended_name + '_items?', remove = True )

        super( CustomQtListEditorKlass, self ).dispose()

    def empty_list ( self ):
        """ Creates an empty list entry (so the user can add a new item).
        """
        # Connecting the new button to the mapper
        from pyface.qt import QtCore, QtGui
        from pyface.qt import qt_api
        from traitsui.qt4.helper import IconButton

        #control = IconButton('list_editor.png', self.mapper.map)
        if qt_api == 'pyside':
            control = IconButton('list_editor.png', self.mapper.map)
        elif qt_api == 'pyqt':
            control = IconButton('list_editor.png',
                lambda : self.popup_empty_menu(
                    self._list_pane.layout().sender() ))

        # Setting the mapping and asking it to send the sender to the
        # callback method
        self.mapper.setMapping(control, control)
        self.mapper.mapped[QtCore.QObject].connect(self.popup_empty_menu)
        control.is_empty = True
        self._cur_control = control

        from traitsui.editors.list_editor import ListItemProxy
        proxy    = ListItemProxy( self.object, self.name, -1, None, None )
        pcontrol = QtGui.QLabel('   (Empty List)')
        pcontrol.proxy = control.proxy = proxy

        layout = self._list_pane.layout()
        layout.addWidget(control, 0, 0)
        layout.addWidget(pcontrol, 0, 1)

    def get_info ( self ):
        """ Returns the associated object list and current item index.
        """
        proxy = self._cur_control.proxy
        return ( proxy.list, proxy.index )

    def popup_empty_menu ( self , sender):
        """ Displays the empty list editor popup menu.
        """
        from traitsui.qt4.menu import MakeMenu
        from pyface.qt import QtCore, QtGui
        self._cur_control = control = sender
        menu = MakeMenu( self.empty_list_menu, self, True, control ).menu
        menu.exec_(control.mapToGlobal(QtCore.QPoint(0, 0)))

    def popup_menu ( self , sender):
        """ Displays the list editor popup menu.
        """
        from traitsui.qt4.menu import MakeMenu
        from pyface.qt import QtCore, QtGui
        self._cur_control = sender

        proxy    = sender.proxy
        index    = proxy.index
        menu     = MakeMenu( self.list_menu, self, True, sender ).menu
        len_list = len( proxy.list )
        not_full = (len_list < self._trait_handler.maxlen)

        self._menu_before.enabled( not_full )
        self._menu_after.enabled(  not_full )
        self._menu_delete.enabled( len_list > self._trait_handler.minlen )
        self._menu_up.enabled(  index > 0 )
        self._menu_top.enabled( index > 0 )
        self._menu_down.enabled(   index < (len_list - 1) )
        self._menu_bottom.enabled( index < (len_list - 1) )

        menu.exec_(sender.mapToGlobal(QtCore.QPoint(0, 0)))

    def add_item ( self, offset ):
        """ Adds a new value at the specified list index.
        """
        list, index = self.get_info()
        index      += offset
        item_trait  = self._trait_handler.item_trait
        value       = item_trait.default_value_for( self.object, self.name )
        self.value = list[:index] + [ value ] + list[index:]
        self.update_editor()

    def add_before ( self ):
        """ Inserts a new item before the current item.
        """
        self.add_item( 0 )

    def add_after ( self ):
        """ Inserts a new item after the current item.
        """
        self.add_item( 1 )

    def add_empty ( self ):
        """ Adds a new item when the list is empty.
        """
        list, index = self.get_info()
        self.add_item( 0 )

    def delete_item ( self ):
        """ Delete the current item.
        """
        list, index = self.get_info()
        self.value  = list[:index] + list[index+1:]
        self.update_editor()

    def move_up ( self ):
        """ Move the current item up one in the list.
        """
        list, index = self.get_info()
        self.value  = (list[:index-1] + [ list[index], list[index-1] ] +
                       list[index+1:])
        self.update_editor()

    def move_down ( self ):
        """ Moves the current item down one in the list.
        """
        list, index = self.get_info()
        self.value  = (list[:index] + [ list[index+1], list[index] ] +
                       list[index+2:])
        self.update_editor()

    def move_top ( self ):
        """ Moves the current item to the top of the list.
        """
        list, index = self.get_info()
        self.value  = [ list[index] ] + list[:index] + list[index+1:]
        self.update_editor()

    def move_bottom ( self ):
        """ Moves the current item to the bottom of the list.
        """
        list, index = self.get_info()
        self.value  = list[:index] + list[index+1:] + [ list[index] ]
        self.update_editor()

    def _dispose_items ( self ):
        """ Disposes of each current list item.
        """
        layout = self._list_pane.layout()
        child = layout.takeAt(0)
        while child is not None:
            control = child.widget()
            if control is not None:
                editor = getattr( control, '_editor', None )
                if editor is not None:
                    editor.dispose()
                    editor.control = None
                control.deleteLater()
            child = layout.takeAt(0)
        del child

    def _kind_default(self):
        """ Returns a default value for the 'kind' trait.
        """
        return self.factory.style + '_editor'

    def _mutable_default(self):
        """ Trait handler to set the mutable trait from the factory.
        """
        return self.factory.mutable

if _tk == 'wx':
    from traitsui.editors.list_editor import ListEditor as CustomListEditor
elif _tk == 'qt4':

    from traitsui.editor_factory import EditorFactory
    from traitsui.ui_traits import style_trait

    #custom edior on traitsui level
    class CustomListEditor(CustomEditor):
        '''
        Traitsui level abstraction layer for the Custom List Editor
        '''
        factory = Property #Callable

        editor = Instance( EditorFactory )

        mutable = Bool(True)

        style = style_trait

        trait_handler = Instance( BaseTraitHandler )

        rows = Range(1, 50, 5, desc = 'the number of list rows to display')
        columns = Range(1, 10, 1, dec = 'the number of list cols to display')

        use_notebook = Bool(False)
        show_notebook_menu = Bool(False)

        def _get_klass(self):
            #tell the editor to use the klass defined in this file
            return CustomQtListEditorKlass

        def _get_factory(self):
            return partial(mkeditor)
        
if __name__=='__main__':
    import signal
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    from traits.api import List, HasTraits
    from traitsui.api import View, Item 

    class Dog(HasTraits):
        dog = List

        def _dog_default(self):
            return [4,5,7]

        view = View(Item('dog', editor=CustomListEditor()))

    Dog().configure_traits()
