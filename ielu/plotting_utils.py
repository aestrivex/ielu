from __future__ import division
import os
import numpy as np
import nibabel as nib

from traits.api import HasTraits, Float, Int, Tuple
from traitsui.api import View, Item, CSVListEditor

from geometry import get_vox2rasxfm, apply_affine, get_std_orientation
from utils import get_subjects_dir

def force_render( figure=None ):
    from mayavi import mlab
    figure.scene.render()
    mlab.draw(figure=figure)
    from pyface.api import GUI
    _gui = GUI()
    orig_val = _gui.busy
    _gui.set_busy(busy=True)
    _gui.process_events()
    _gui.set_busy(busy=orig_val)
    _gui.process_events()

def coronal_slice(elecs, start=None, end=None, outfile=None, 
    subjects_dir=None,
    subject=None, reorient2std=True, dpi=150, size=(200,200),
    title=None): 
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
    title : Str
        Specify a matplotlib title
    '''
    print 'creating coronal slice with start electrodes %s' % str(start)

    subjdir_subj = get_subjects_dir( subjects_dir=subjects_dir,
                                     subject=subject )
    orig = os.path.join(subjdir_subj, 'mri', 'orig.mgz')

    x_size, y_size, z_size = nib.load(orig).shape

    # vox2ras and ras2vox shouldnt have different procedures for 
    # getting the different dimensions. the matrix showing those
    # dimensions has the correct dimensions by inversion beforehand
    # in the complex 3-way case
    vox2ras = get_vox2rasxfm(orig, stem='vox2ras')
    ras2vox = np.linalg.inv(vox2ras)

    ras2vox[0:3,3] = (x_size/2, y_size/2, z_size/2)

    rd, ad, sd = get_std_orientation(ras2vox)

#    rd, = np.where(np.abs(ras2vox[:,0]) == np.max(np.abs(ras2vox[:,0])))
#    ad, = np.where(np.abs(ras2vox[:,1]) == np.max(np.abs(ras2vox[:,1])))
#    sd, = np.where(np.abs(ras2vox[:,2]) == np.max(np.abs(ras2vox[:,2])))

    r_size = [x_size, y_size, z_size][rd]
    a_size = [x_size, y_size, z_size][ad]
    s_size = [x_size, y_size, z_size][sd]

    #starty = pd.map_cursor( start.asras(), pd.current_affine, invert=True)[1]
    #endy = pd.map_cursor( end.asras(), pd.current_affine, invert=True )[1]
    #midy = (starty+endy)/2
    #pd.move_cursor(128, midy, 128)

    electrodes = np.squeeze([apply_affine([e.asras()], ras2vox) 
        for e in elecs])
    #electrodes = np.array([pd.map_cursor(e.asras(), ras2vox,
    #    invert=True) for e in elecs])

    vol = np.transpose( nib.load(orig).get_data(), (rd, ad, sd) )
    
    if start is not None and end is not None:
        start_coord = np.squeeze(apply_affine([start.asras()], ras2vox))
        end_coord = np.squeeze(apply_affine([end.asras()], ras2vox))

        if start_coord[rd] == end_coord[rd]:
            raise ValueError('This lead has no variation in the X axis. It shouldnt be displayed coronally')
        
        slice = np.zeros((s_size, r_size))
        
        m = (start_coord[ad]-end_coord[ad])/(start_coord[rd]-end_coord[rd])
        b = start_coord[ad]-m*start_coord[rd]

        rnew = np.arange(r_size)
        anew = m*rnew+b
        alower = np.floor(anew)
        afrac = np.mod(anew, 1)

        try:
            for rvox in rnew:
                slice[:, rvox] = (vol[rvox, alower[rvox], :] * 
                    (1-afrac[rvox])+vol[rvox, alower[rvox]+1, :] *
                    afrac[rvox])
        except IndexError:
            raise ValueError('This lead has minimal variation in the X axis. It shouldnt be displayed coronally')

    else:
        slice_nr = np.mean(electrodes[:,ad])
        slice = vol[:, slice_nr, :].T
    
    vox2pix = np.zeros((2,4))
    vox2pix[0, rd] = 1
    vox2pix[1, sd] = 1
    ras2pix = np.dot(vox2pix, ras2vox)

    pix = np.dot(ras2pix, 
        np.transpose([np.append(e.asras(), 1) for e in elecs]))

    #add data to coronal plane
    import pylab as pl

    fig = pl.figure()

    pl.imshow(slice, cmap='gray')
    pl.scatter(pix[0,:], pix[1,:], s=10, c='red', edgecolor='yellow',
        linewidths=0.4)

    if title is not None:
        pl.title(title)

    pl.axis('off')
    #pl.show()

    if outfile is not None:
        pl.savefig(outfile, dpi=dpi)

    return fig

def sequence_3d_images( figure ):
    from mayavi import mlab
    views = [lambda:mlab.view( azimuth=0, elevation=90, figure=figure ),
        lambda:mlab.view( azimuth=180, elevation=90, figure=figure ),
        lambda:mlab.view( azimuth=0, elevation=0, figure=figure ),
        lambda:mlab.view( azimuth=90, elevation=90, figure=figure ),
        lambda:mlab.view( azimuth=270, elevation=90, figure=figure )]

    for view in views:
        yield view

def save_opaque_clinical_sequence( savefile, mayavi_figure ):
    import pylab as pl
    from matplotlib.backends.backend_pdf import PdfPages
    from mayavi import mlab

    with PdfPages(savefile) as pdf:
        for angle in sequence_3d_images( mayavi_figure ):
            angle()
            force_render( figure=mayavi_figure )
            pixmap = mlab.screenshot( figure=mayavi_figure )
            mpl_figure = pl.figure()
            pl.imshow(pixmap, figure=mpl_figure)
            pdf.savefig(mpl_figure)

