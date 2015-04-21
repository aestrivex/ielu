from __future__ import division
import os
import numpy as np
import nibabel as nib

from traits.api import HasTraits, Float, Int, Tuple
from traitsui.api import View, Item, CSVListEditor

def coronal_slice(elecs, start=None, end=None, outfile=None, 
    subjects_dir=None,
    subject=None, reorient2std=True, dpi=150, size=(200,200)): 
    '''
    create an image of a coronal slice which serves as a guesstimate of a
    depth lead inserted laterally and nonvaryingly in the Y axis

    plot the electrodes from the lead overlaid on the slice in the X and Z
    directions

    Paramaters
    ----------
    elecs : List( Electrode )
        list of electrode objects forming this depth lead
    start : Electrode
        Electrode object at one end of the depth lead
    end : Electrode
        Electrode object at the other end of the depth lead
    outfile : Str
        Filename to save the image to
    subjects_dir : Str | None
        The freesurfer subjects_dir. If this is None, it is assumed to be the
        $SUBJECTS_DIR environment variable. If this folder is not writable,
        the program will crash.
    subject : Str | None
        The freesurfer subject. If this is None, it is assumed to be the
        $SUBJECT environment variable.
    reorient2std : Bool
        Apply a matrix to rotate orig.mgz to the standard MNI orientation
        emulating fslreorient2std. Pretty much always true here.
    dpi : Int
        Dots per inch of output image
    size : Tuple
        Specify a 2-tuple to control the image size, default is (200,200)
    '''
    print 'creating coronal slice with start electrodes %s' % str(start)

    if subjects_dir is None or subjects_dir=='':
        subjects_dir = os.environ['SUBJECTS_DIR']
    if subject is None or subject=='':
        subject = os.environ['SUBJECT']

    orig = os.path.join(subjects_dir, subject, 'mri', 'orig.mgz')

    import panel2d
    pd = panel2d.TwoDimensionalPanel()
    pd.load_img(orig, reorient2std=reorient2std)

    #identify coronal slice positions -- 
        #stupid version - use middle slice
        #smart version bit later -- replicate cash lab procedure to
            #create slice along coordinate frame of lead geometry

    #starty = pd.map_cursor( start.asras(), pd.current_affine, invert=True)[1]
    #endy = pd.map_cursor( end.asras(), pd.current_affine, invert=True )[1]
    #midy = (starty+endy)/2
    #pd.move_cursor(128, midy, 128)

    electrodes = np.array([pd.map_cursor(e.asras(), pd.current_affine, 
        invert=True) for e in elecs])

    if start is not None and end is not None:
        start_coord = pd.map_cursor( start.asras(), pd.current_affine, 
            invert=True)
        end_coord = pd.map_cursor( end.asras(), pd.current_affine, 
            invert=True )
        
        x_size, _, z_size = pd.current_image.shape
        slice = np.zeros((z_size, x_size))
        
        m = (start_coord[1]-end_coord[1])/(start_coord[0]-end_coord[0])
        b = start_coord[1]-m*start_coord[0]

        rnew = np.arange(x_size)
        anew = m*rnew+b
        alower = np.floor(anew)
        afrac = np.mod(anew, 1)

        vol = pd.current_image
        for rvox in rnew:
            slice[:, rvox] = (vol[rvox, alower[rvox], :] * 
                (1-afrac[rvox])+vol[rvox, alower[rvox], :] *
                afrac[rvox])
    else:
        slice_nr = np.mean(electrodes[:,1])
        slice = pd.current_image[:, slice_nr, :].T
    
    pd.xz_plane.data.set_data('imagedata', slice)

    #determine electrode positions
    #electrodes = np.random.random((8,2))*150+50


    #add data to coronal plane
    pd.xz_plane.data.set_data('electrodes_x', electrodes[:,0])
    pd.xz_plane.data.set_data('electrodes_z', electrodes[:,2])

    pd.xz_plane.plot(('electrodes_x','electrodes_z'), type='scatter',
        color='red', marker='dot', size=8, name='electrodes')

    pd.xz_plane.delplot('cursor')
    #pd.edit_traits()

    if outfile is not None:
        from chaco.api import PlotGraphicsContext
        pd.xz_plane.do_layout(force=True)
        pd.xz_plane.outer_bounds = size
        gc = PlotGraphicsContext(size, dpi=dpi)
        gc.render_component(pd.xz_plane)
        gc.save(outfile)

    return pd
