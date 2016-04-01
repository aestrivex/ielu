#
# The code in identify_roi_from_aparc originally comes from Gio Piantoni
#

from __future__ import division
import os
import numpy as np
import nibabel as nib
import geometry as geo
import grid as gl
from utils import SortingLabelingError
from electrode import Electrode
from scipy.spatial.distance import cdist

def create_brainmask_in_ctspace(ct, subjects_dir=None, subject=None, 
    overwrite=False):
    '''
    Calculate the reverse transformation from the MR space to the CT space,
    translate the brainmask into the space of the CT image

    Parameters
    ----------
    ct : str
        The filename of the CT image to use
    subjects_dir : Str | None
        The freesurfer subjects_dir. If this is None, it is assumed to be the
        $SUBJECTS_DIR environment variable. If this folder is not writable,
        the program will crash.
    subject : Str | None
        The freesurfer subject. If this is None, it is assumed to be the
        $SUBJECT environment variable.
    overwrite : Bool
        When true, will do the computation and not search for a saved value.
        Defaults to false.

    Returns
    -------
    brainmask : str
        The location of the textfile where the brainmask is located,
        which is currently $SUBJECTS_DIR/mri/brain_ct.nii.gz
    '''
    print 'converting brainmask to CT image space'

    if subjects_dir is None or subjects_dir=='':
        subjects_dir = os.environ['SUBJECTS_DIR']
    if subject is None or subject=='':
        subject = os.environ['SUBJECT']

    xfms_dir = os.path.join(subjects_dir, subject, 'mri', 'transforms')
    if not os.path.exists(xfms_dir):
        os.mkdir(xfms_dir)
    lta = os.path.join(xfms_dir,'mr2ct.lta')
    #_,lta = tempfile.mkstemp()
    
    rawavg = os.path.join(subjects_dir, subject, 'mri', 'rawavg.mgz')
    brain = os.path.join(subjects_dir, subject, 'mri', 'brain.mgz')
    nas_brain = os.path.join(subjects_dir, subject, 'mri', 'brain_nas.nii.gz')
    ct_brain = os.path.join(subjects_dir, subject, 'mri', 'brain_ct.nii.gz')

    if os.path.exists(lta) and not overwrite:
        print 'brainmask in ct space already exists, using it'
        return ct_brain

    import tempfile
    import subprocess

    mri_vol2vol_cmd = ['mri_vol2vol','--mov',brain,'--targ',rawavg,
        '--regheader','--o',nas_brain,'--no-save-reg']
    r=subprocess.call(mri_vol2vol_cmd)

    mri_robustreg_cmd1 = ['mri_robust_register','--mov',rawavg,'--dst',ct,
        '--lta',lta,'--satit','--vox2vox','--cost','nmi']
    p=subprocess.call(mri_robustreg_cmd1)

    _,gbg = tempfile.mkstemp()

    mri_robustreg_cmd2 = ['mri_robust_register','--mov',nas_brain,'--dst',ct,
        '--lta',gbg,'--satit','--cost','nmi','--ixform',lta,'--mapmov',
        ct_brain]
    q=subprocess.call(mri_robustreg_cmd2)

    #os.unlink(lta)
    os.unlink(gbg)

    return ct_brain

def identify_electrodes_in_ctspace(ct, mask=None, threshold=2500, 
    use_erosion=True, isotropization_type=None, iso_vector_override=None):
    '''
    Given a CT image, identify the electrode locations in CT space.
    Includes locations of high image intensity that are not electrodes.

    Optionally performs binary erosion on the image using a spherical kernel
    of radius 1 to reduce noise (highly recommended)

    Parameters
    ----------
    ct : str
        The filename of the CT image to use
    mask : str | np.ndarray
        The filename of an image in CT space to use as a brainmask.
        Alternately, a matrix in the same shape as the CT image.
        The use of a mask is optional and not crucial for the algorithm to
        work well at the present time.
    threshold : float | int
        The threshold used to identify the electrodes from the image.
        The intensity of the electrodes should be above this threshold, while
        the rest of the image should mostly be below the threshold -- though
        it is ok if parts of the skull, mandibles, and image artifacts outside
        the brain exceed this threshold. The default value is 2500 which
        should be appropriate for many CT images.
    use_erosion : bool
        If true, before extracting electrodes, binary erosion is applied to 
        the image using a spherical kernel of radius 1. The default value is 
        true. Use of the binary erosion procedure is strongly recommended. 
        When using CT images of very high slice thickness, it may be
        necessary to turn off the binary erosion but these images are 
        probably not usable with our algorithm anyway.
    isotropization_type : str | None
        None : do nothing
        By voxel : Isotropize such that the image is aligned with the
            longest dimension in number of voxels 
        By header : Isotropize such that the image has
            isotropic coordinate space according to affine matrix
        Manual override : Provide a manual zoom vector
    iso_vector_override : List(Float)
        User specifies in manual override isotropization setting.

    Returns
    -------
    electrodes : List(Electrode)
        an list of Electrode objects with only the ct coords indicated.
    '''
    print 'identifying electrode locations from CT image'

    from scipy import ndimage
    import sys

    def get_centerofmass(isotropize=None):
        cti = nib.load(ct)   
        ctd = cti.get_data()

        if isotropization_type == 'By voxel':
            initial_shape = ctd.shape

            max_axis = np.max(ctd.shape)

            #WARNING: this is not the true isotropization
            #factor. To get the true isotropization factor we would have to
            #trust the image to tell us its correct slice thickness.
            #But this is usually a good approximation of the shitty slice
            #thickness scans we have been getting.

            zf = np.array([max_axis, max_axis, max_axis]) // ctd.shape
            print 'DOING THE ISOTROPIC LINEARIZATION'
            ctd = ndimage.interpolation.zoom(ctd, zf)
            print 'FINISHED ISOTROPIC LINEARIZATION'

            new_shape = ctd.shape

            print 'INITIAL SHAPE: {0}, NEW SHAPE {1}'.format(initial_shape,
                new_shape)

        elif isotropization_type == 'By header':
            initial_shape = ctd.shape

            #these are the same thing
            vox2ras = cti.get_affine()
            #vox2ras = geo.get_vox2rasxfm(ct, stem='vox2ras')

            #check orientation
            rd, ad, sd = geo.get_std_orientation(vox2ras)

            vox2ras_rstd = np.array(
                map( lambda ix: np.squeeze( vox2ras[ix, :3] ),
                     (rd, ad, sd) ))

            vox2ras_dg = np.abs(np.diag(vox2ras_rstd)[:3])

            min_axis = np.min( vox2ras_dg )
            zf = vox2ras_dg / min_axis

            if np.all(zf == 1):
                print 'IMAGE HEADER IS ISOTROPIC, NO LINEARIZATION TO DO'
            else:

                print 'DOING THE ISOTROPIC LINEARIZATION'
                ctd = ndimage.interpolation.zoom(ctd, zf)
                print 'FINISHED ISOTROPIC LINEARIZATION'
                new_shape = ctd.shape

                print 'INITIAL SHAPE: {0}, NEW SHAPE {1}'.format(
                    initial_shape, new_shape)

        elif isotropization_type == 'Manual override':
            initial_shape = ctd.shape
            zf = np.array(iso_vector_override)

            if np.all(zf == 1):
                print 'IMAGE HEADER IS ISOTROPIC, NO LINEARIZATION TO DO'
            else:
                print 'DOING THE ISOTROPIC LINEARIZATION'
                ctd = ndimage.interpolation.zoom(ctd, zf)
                print 'FINISHED ISOTROPIC LINEARIZATION'
                new_shape = ctd.shape

                print 'INITIAL SHAPE: {0}, NEW SHAPE {1}'.format(
                    initial_shape, new_shape)

        #istropization done

        print np.mean(ctd), 'CT MEAN'
        print np.std(ctd), 'CT STDEV'

        #threshold = np.mean(mask_test)+3*np.std(mask_test)
        print threshold, 'COMPROMISE'

        #supthresh_locs = np.where(np.logical_and(ctd > threshold, maskd))
        supthresh_locs = np.where( ctd > threshold )

        ecs = np.zeros(ctd.shape)
        ecs[supthresh_locs]=1

        if use_erosion:
            cte = ndimage.binary_erosion(ecs)
        else:
            cte = ecs

        ctpp = np.zeros(ctd.shape)
        ctpp[np.where(cte)] = ctd[np.where(cte)]

        class Component():
            def __init__(self):
                self.coor = []
                self.intensity = []
            def add(self, coor, intensity):
                self.coor.append(coor)
                self.intensity.append(intensity)
            def center_of_mass(self):
                M, Rx, Ry, Rz = 0,0,0,0
                for r,m in zip(self.coor, self.intensity):
                    M+=m
                    Rx+=m*r[0]
                    Ry+=m*r[1]
                    Rz+=m*r[2]
                return round(Rx/M), round(Ry/M), round(Rz/M)

        #in principle it is possible to alter the algorithm by removing
        #some of the subsequent categories of diagonals. in practice
        #this makes very little difference compared to choosing an
        #appropriate threshold and having good quality images
        def iter_bfs(x,y,z,im,c):
            from Queue import Queue
            queue = Queue()

            queue.put( (x,y,z) )

            while not queue.empty():

                cx, cy, cz = queue.get_nowait()

                try:
                    if im[cx,cy,cz]==0:
                        continue
                except IndexError:
                    continue

                if cx < 0 or cy < 0:
                    continue
                
                c.add((cx,cy,cz), im[cx,cy,cz])
                im[cx,cy,cz]=0
            
                queue.put((cx-1, cy, cz))
                queue.put((cx+1, cy, cz))
                queue.put((cx, cy-1, cz))
                queue.put((cx, cy+1, cz))
                queue.put((cx, cy, cz-1))
                queue.put((cx, cy, cz+1))

                queue.put((cx-1, cy-1, cz))
                queue.put((cx-1, cy+1, cz))
                queue.put((cx-1, cy, cz-1))
                queue.put((cx-1, cy, cz+1))
                queue.put((cx+1, cy-1, cz))
                queue.put((cx+1, cy+1, cz))
                queue.put((cx+1, cy, cz-1))
                queue.put((cx+1, cy, cz+1))
                queue.put((cx, cy-1, cz-1))
                queue.put((cx, cy-1, cz+1))
                queue.put((cx, cy+1, cz-1))
                queue.put((cx, cy+1, cz+1))
                
                queue.put((cx-1, cy-1, cz-1))
                queue.put((cx-1, cy-1, cz+1))
                queue.put((cx-1, cy+1, cz-1))
                queue.put((cx-1, cy+1, cz+1))
                queue.put((cx+1, cy-1, cz-1))
                queue.put((cx+1, cy-1, cz+1))
                queue.put((cx+1, cy+1, cz-1))
                queue.put((cx+1, cy+1, cz+1))

        #this recursive function occasionally caused memory problems
        #it also caused stack size problems
        def dfs(x,y,z,im,c):
            try:
                if im[x,y,z]==0:
                    return
            except IndexError:
                return
            if x<0 or y<0:
                return 

            c.add((x,y,z), im[x,y,z])
            im[x,y,z]=0
            dfs(x-1,y,z,im,c)
            dfs(x+1,y,z,im,c)
            dfs(x,y-1,z,im,c)
            dfs(x,y+1,z,im,c)
            dfs(x,y,z-1,im,c)
            dfs(x,y,z+1,im,c)

            dfs(x-1, y-1, z, im, c)
            dfs(x-1, y+1, z, im, c)
            dfs(x-1, y, z-1, im, c)
            dfs(x-1, y, z+1, im, c)
            dfs(x+1, y-1, z, im, c)
            dfs(x+1, y+1, z, im, c)
            dfs(x+1, y, z-1, im, c)
            dfs(x+1, y, z+1, im, c)
            dfs(x, y-1, z-1, im, c)
            dfs(x, y-1, z+1, im, c)
            dfs(x, y+1, z-1, im, c)
            dfs(x, y+1, z+1, im, c)

            dfs(x-1, y-1, z-1, im, c)
            dfs(x-1, y-1, z+1, im, c)
            dfs(x-1, y+1, z-1, im, c)
            dfs(x-1, y+1, z+1, im, c)
            dfs(x+1, y-1, z-1, im, c)
            dfs(x+1, y-1, z+1, im, c)
            dfs(x+1, y+1, z-1, im, c)
            dfs(x+1, y+1, z+1, im, c)

        def isolate_components(image):
            im = image.copy()

            clusters=[]
            #print np.shape(im), 'gabif'
            for x,y,z in zip(*np.where(im)):
                if im[x,y,z]==0:
                    continue
                else:
                    c = Component()
                    clusters.append(c)
                    iter_bfs(x,y,z,im,c) 
                
            return clusters

        recursionlimit = sys.getrecursionlimit()
        sys.setrecursionlimit(3000000)

        electrode_clusters = isolate_components(ctpp)

        sys.setrecursionlimit(recursionlimit)

        return [cluster.center_of_mass() for cluster in electrode_clusters]

    ret_elecs = []
    if isotropization_type!='Isotropization off':
        return [Electrode(iso_coords=i) for i in 
            get_centerofmass(isotropize=isotropization_type)]
    else:
        return [Electrode(ct_coords=c) for c in
            get_centerofmass(isotropize=isotropization_type)]

def linearly_transform_electrodes_to_isotropic_coordinate_space(electrodes,
    ct, isotropization_direction_on=None,
    isotropization_direction_off=None,
    isotropization_strategy=None,
    iso_vector_override=None):
    '''
    Execute a simple linear transformation to expand the electrode locations
    to an isotropic coordinate space of maximal size

    Parameters
    ----------
    electrodes : List(Electrode)
        A list of electrodes with ct_coords set
    ct : str
        The filename of the ct image to use
    disable_isotropization : Bool
        If true, copy CT locations instead of isotropizing
    isotropization_direction_off : 'copy_to_ct' | 'copy_to_iso'
        copy_to_ct: copies value in iso_coords to ct_coords
        copy_to_iso: copies value in ct_coords to iso_coords
    isotropization_direction_on : 'isotropize' | 'deisotropize'
        isotropize: convert nonisotropic ct to isotropic iso
        deisotropize: convert isotropic iso to nonisotropic ct
    isotropization_strategy : 'Header' | 'Voxel' | 'Manual' | 'Off'
    iso_vector_override: List(Float)
        provided for manual isotropization_strategy
        
    '''
    cti = nib.load(ct)

    for elec in electrodes:
        if isotropization_strategy == 'Isotropization off':
            if isotropization_direction_off == 'copy_to_iso':
                elec.iso_coords = elec.asct()
            elif isotropization_direction_off == 'copy_to_ct':
                elec.ct_coords = elec.asiso()

            continue

        elif isotropization_strategy == 'By voxel':
            cts_max = np.max(cti.shape)
            za, zb, zc = np.array([cts_max, cts_max, cts_max]) / cti.shape
        elif isotropization_strategy == 'By header':
            aff = cti.get_affine()
            aff_rstd = np.array( map( lambda ix: aff[ix, :3],
                                      geo.get_std_orientation(aff)))
            aff_dg = np.abs(np.diag(aff_rstd)[:3])
            za, zb, zc = aff_dg / np.min(aff_dg)
        elif isotropization_strategy == 'Manual override':
            za, zb, zc = np.array(iso_vector_override)
        else:
            raise ValueError('Invalid isotropization strategy')

        if isotropization_direction_on == 'isotropize':
            ca, cb, cc = elec.asct()
            elec.iso_coords = ( ca*za, cb*zb, cc*zc )
        elif isotropization_direction_on == 'deisotropize':
            ia, ib, ic = elec.asiso()
            elec.ct_coords = ( ia/za, ib/zb, ic/zc )
        else:
            raise ValueError('Invalid isotropization direction')

def identify_extracranial_electrodes_in_freesurfer_space(electrodes, 
    dilation_iterations=5, subjects_dir=None, subject=None):
    '''
    Identify the electrodes, following translation to MR space, which fall
    outside of the brain with default dilation parameters

    Parameters
    ----------
    electrodes : List(Electrode)
        A list of electrodes to either exclude or not
    dilation_iterations : int
        The number of iterations to dilate the brain mask. Default value 5

    Returns
    -------
    removals : List(Electrode)
        A list of electrodes not inside the mask
    '''
    if subjects_dir is None or subjects_dir=='':
        subjects_dir = os.environ['SUBJECTS_DIR']
    if subject is None or subject=='':
        subject = os.environ['SUBJECT']

    from scipy import ndimage
    brain = os.path.join(subjects_dir, subject, 'mri', 'brain.mgz')
    maski = nib.load(brain)
    maskd = maski.get_data()

    #from PyQt4.QtCore import pyqtRemoveInputHook
    #import pdb
    #pyqtRemoveInputHook()
    #pdb.set_trace()

    maskd = np.around(maskd) #eliminate noise, what noise? do it anyway
    maskd = ndimage.binary_dilation(maskd, iterations=dilation_iterations)

    removals = []

    #mask_aff = maski.get_affine()
    mask_aff = geo.get_vox2rasxfm(brain, stem='vox2ras-tkr')

    for e in electrodes:
        #find nearest voxel in voxel space
        voxel, = geo.apply_affine([e.asras()], np.linalg.inv(mask_aff))
        nx, ny, nz = map(int, (np.around(voxel)))

        #a value not in [0-255] index range means a value not in the image at all
        #these values are potentially possible for outlying noise but usually
        #are an indication of registration errors
        #so we remove them
        if not np.all([0 <= p <= 255 for p in (nx,ny,nz)]):
            removals.append(e)
            print ("Warning: Removed some extremely far outlying noise, "
                "likely sign of registration errors")
            continue

        if maskd[nx,ny,nz] == 0:
            removals.append(e)

    return removals

def classify_electrodes(electrodes, known_geometry,
    delta=.35, rho=35, rho_strict=20, rho_loose=50, color_scheme=None,
    epsilon=10, mindist=0, maxdist=36, crit_pct=.75):
    '''
    Sort the given electrodes (generally in the space of the CT scan) into
    grids and strips matching the specified geometry.

    Parameters
    ----------
    electrodes : List(electrodes)
        A list of electrode locations. The CT coordinate attribute of the
        electrodes is used as the position.
        It is the caller's responsibility to filter the electrodes list as
        appropriate.
    known_geometry : list of 2-tuples | Gx2 np.ndarray
        A list of the form [(8,8),(8,2)] describing the geometry of the
        grids and strips. There should be G entries, where G is the number of
        grids and/or strips. The list [(8,8),(8,2)] indicates one 8x8 grid,
        and one 8x2 strip. (8,2) and (2,8) are interchangeable.
    fixed_points : None
        currently does nothing. Later functionality will allow the user to
        interactively force points to certain strips and automatically
        determine strip geometry.
    color_scheme : None | Generator
        A generator which returns colors of the form tuple(R,G,B), where
        R,G,B are floats between 0 and 1 rather than hex numbers. If no
        generator is specified, a predefined generator is used which uses 12
        distinct colors and thereafter generates random (potentially
        nonunique) colors.
    delta : Float
        A fitting parameter that controls the relative distance between
        grid points. A grid point cannot be farther than delta*c from its
        orthogonal neighbors, where c is an estimate of the distance between
        grid neighbors, assuming a roughly square grid (Later, this should be
        a rectangular grid). The default value is .35
    rho : Float
        A fitting parameter controlling the distance from which successive
        angles can diverge from 90. The default value is 35
    rho_strict : Float
        A fitting parameter similar to rho but used in different geometric
        circumstances. The default value is 20.
    rho_loose : Float
        A fitting parameter similar to rho but used in different geometric
        circumstances. The default value is 50.
    epsilon : Float
        A fitting parameter controlling the acceptable deviation from 90
        degrees for the starting point of a KxM grid where K>1,M>1. A
        larger parameter means the algorithm will try a larger range of
        starting positions before giving up. The default value is 10.
    mindist : Float
        A fitting parameter controlling the minimum distance between starting
        points to fit at a 90 degree angle. The default value is 0.
    maxdist : Float
        A fitting parameter controlling the maximum distance between starting
        points to fit at a 90 degree angle. The scale of this distance is in
        voxel space of the CT image and therefore a bit arbitrary, so a
        wide range is used by default: the default value is 36.
    crit_pct : Float
        The critical percentage of electrodes to find before returning.
        Default value 0.75

    Returns
    -------
    found_grids : Dict(Grid : List)
        a dictionary of grid objects mapping to the sorted and interpolated 
        points for these grid objects
    color_scheme : None | Generator
        The color scheme to be used, described above
    '''
    from collections import OrderedDict    

    if color_scheme is None:
        from utils import get_default_color_scheme as color_scheme

    def name_generator():
        i=0
        while True:
            i+=1
            yield 'grid%i'%i

    colors = color_scheme()
    names = name_generator()

    #electrode_arr = map((lambda x:getattr(x, 'ct_coords')), electrodes)
    electrode_arr = map((lambda x:getattr(x, 'iso_coords')), electrodes)

    found_grids = {}
    grid_colors = OrderedDict()
    grid_colors['unsorted'] = (1,0,0)
    grid_colors['selection'] = (1,1,1)
    grid_geom = {}
    used_points = []

    for dims in known_geometry:
        new_elecs = geo.rm_pts(np.reshape(used_points, (-1,3)), 
            np.array(electrode_arr))

        #TODO mindist and maxdist settable parameters
        angles, _, neighbs = gl.find_init_angles(new_elecs, mindist=mindist, 
            maxdist=maxdist)

        ba = np.squeeze(sorted(zip(*np.where(np.abs(90-angles)<epsilon)),
                key=lambda v:np.abs(90-angles[v])))

        if ba.shape==():
            ba=[ba]
        elif len(ba)==0:
            raise SortingLabelingError("Could not find any good angles")

        for j,k in enumerate(ba):
            p0,p1,p2 = neighbs[k]
            pog = gl.Grid(p0,p1,p2,new_elecs, delta=delta,
                rho=rho, rho_strict=rho_strict, rho_loose=rho_loose,
                name=names.next(), critical_percentage=crit_pct)
            pog.extend_grid_arbitrarily()

            try:
                sp, corners, final_connectivity = pog.extract_strip(*dims)
            except SortingLabelingError as e:
                print 'Rejected this initialization'
                if j==len(ba)-1:
                    print ('No suitable strip found. Returning an empty '
                        'strip in its place')
                    found_grids[pog.name] = []
                    grid_colors[pog.name] = colors.next()
                    grid_geom[pog.name] = dims
                continue

            sp = np.reshape(sp, (-1,3))

            found_grids[pog.name] = []
            grid_colors[pog.name] = colors.next()
            grid_geom[pog.name] = dims
            for p in sp:
                used_points.append(p)
                #from PyQt4.QtCore import pyqtRemoveInputHook
                #pyqtRemoveInputHook()
                #import pdb
                #pdb.set_trace()
                if tuple(p.tolist()) in electrode_arr:
                    ix, = np.where(np.logical_and(np.logical_and( 
                        np.array(electrode_arr)[:,0]==p[0], 
                        np.array(electrode_arr)[:,1]==p[1]),
                        np.array(electrode_arr)[:,2]==p[2]))
                    try:
                        elec = electrodes[ix]
                        found_grids[pog.name].append(elec)
                    except (IndexError, TypeError) as e:
                        print ix
                        print p
                        raise SortingLabelingError(
                            "multiple electrodes at same point")
                else:
                    #elec = Electrode(ct_coords=tuple(p), 
                    #    is_interpolation=True)
                    elec = Electrode(iso_coords=tuple(p),
                        is_interpolation=True)
                    found_grids[pog.name].append(elec)

                #add corner information
                for corner in corners:
                    if np.all(corner==np.array(elec.asiso())):
                        elec.corner = ['corner 1']

                #add experimental full geometry information

                try:
                    elec.geom_coords = list(final_connectivity[
                        elec.asiso()])
                except KeyError:
                    pass
                    
            break

    #return found_grids, grid_colors
    return grid_colors, grid_geom, found_grids, colors


def remove_large_negative_values_from_ct(ct, subjects_dir=None,
    subject=None, threshold=-200):
    '''
    Opens the CT, checks for values less than threshold. Sets all values less than
    threshold to threshold instead. This is helpful for registration as extremely 
    large negative values present in the CT but not in the MR skew the mutual
    information algorithm.

    Parameters
    ----------
    ct : Str
        The filename containing the CT scan
    subjects_dir : Str | None
        The freesurfer subjects_dir. If this is None, it is assumed to be the
        $SUBJECTS_DIR environment variable.
    subject : Str | None
        The freesurfer subject. If this is None, it is assumed to be the
        $SUBJECT environment variable.
    '''
    print 'removing negative values from CT'

    if subjects_dir is None or subjects_dir=='':
        subjects_dir = os.environ['SUBJECTS_DIR']
    if subject is None or subject=='':
        subject = os.environ['SUBJECT']

    cti = nib.load(ct)   
    ctd = cti.get_data()

    if np.min(ctd) > threshold:
        print 'No large negative values in CT image'
        return

    ct_unaltered = os.path.join(subjects_dir, subject, 'mri', 
        'ct_unaltered.nii.gz')

    if os.path.exists(ct_unaltered):
        return
    else:
        nib.save(cti, ct_unaltered)

    ctd[ctd < threshold] = threshold

    ct_new = nib.Nifti1Image(ctd, header=cti.get_header(), 
        affine=cti.get_affine())
    nib.save(ct_new, ct)

def register_ct_to_mr_using_mutual_information(ct, subjects_dir=None,
    subject=None, overwrite=False):
    '''
    Performs the registration between CT and MR using the normalized mutual
    information cost option in freesurfer's mri_robust_register. Saves the
    output to a temporary file which is subsequently examined and the
    linear registration is returned.

    Freesurfer should already be sourced.

    Parameters
    ----------
    ct : Str
        The filename containing the CT scan
    subjects_dir : Str | None
        The freesurfer subjects_dir. If this is None, it is assumed to be the
        $SUBJECTS_DIR environment variable.
    subject : Str | None
        The freesurfer subject. If this is None, it is assumed to be the
        $SUBJECT environment variable.
    overwrite : Bool
        When true, will do the computation and not search for a saved value.
        Defaults to false.

    Returns
    -------
    affine : 4x4 np.ndarray
        The matrix containing the affine transformation from CT to MR space.
    '''
    print 'registering CT to MR'

    if subjects_dir is None or subjects_dir=='':
        subjects_dir = os.environ['SUBJECTS_DIR']
    if subject is None or subject=='':
        subject = os.environ['SUBJECT']

    #_,lta = tempfile.mkstemp()
    xfms_dir = os.path.join(subjects_dir, subject, 'mri', 'transforms')
    if not os.path.exists(xfms_dir):
        os.mkdir(xfms_dir)
    lta = os.path.join(xfms_dir,'ct2mr.lta')

    if os.path.exists(lta) and not overwrite:
        print 'using existing CT to MR transformation'
        return geo.get_lta(lta)

    #import tempfile
    import subprocess

    rawavg = os.path.join(subjects_dir, subject, 'mri', 'rawavg.mgz')
    out = os.path.join(subjects_dir, subject, 'mri', 'ct_nas.nii.gz')

    mri_robustreg_cmd = ['mri_robust_register','--mov',ct,'--dst',rawavg,
        '--lta',lta,'--satit','--vox2vox','--cost','nmi','--mapmov',out]
    p=subprocess.call(mri_robustreg_cmd)

    affine=geo.get_lta(lta)
    #os.unlink(lta)

    return affine

def register_ct_using_zoom_correction(ct, subjects_dir=None, subject=None,
    cm_dist=5, overwrite=False, zf_override=None):
    '''
    Performs a sophisticated and somewhat specific hack to register a
    high resolution CT image with an awkward slice thickness and skewed
    shape to the structural MRI.

    This hack expects a high resolution CT image with equal X,Y dimensions,
    and some arbitrary slice thickness. An example size is 512x512x319.

    Such images have skewed shape which prevents the mutual information
    criterion from getting a good registration. Instead, we resample the
    image in the Z dimension only to have a more uniform shape.

    The algorithm to figure out the extent to which to resample the image
    is as follows: pick 2 arbitrary axial slices. We use the immediate
    midpoint of the image, and a slice 5 cm away from this slice.
    Freesurfer is good at registering the MRI to these slices.

    Measure the translation difference between these slices, from which
    we can determine the true ratio between the distance shown in the CT
    image, and the actual distance in millimeters. We then build a newly
    resampled CT image using this sampling, with approximately isotropic 
    voxels instead of subsampled voxels (as freesurfer tries to do).

    Clearly, this hack is pretty specific to the type of CT image we are
    collecting at our center. But maybe it can be of more general use.
    '''
    print 'registering CT to MR with manual resampling and hacky tricks'

    if subjects_dir is None or subjects_dir=='':
        subjects_dir = os.environ['SUBJECTS_DIR']
    if subject is None or subject=='':
        subject = os.environ['SUBJECT']

    import subprocess

    xfms_dir = os.path.join(subjects_dir, subject, 'mri', 'transforms')
    if not os.path.exists(xfms_dir):
        os.mkdir(xfms_dir)

    ct_register_dir = os.path.join( subjects_dir, subject, 'mri', 'ct_reg' )
    if not os.path.exists(ct_register_dir):
        os.mkdir(ct_register_dir)

    #lta = os.path.join(xfms_dir, 'ct2mr.lta')

    skewed_lta = os.path.join(ct_register_dir,'skewed_ct2mr.lta')
    lta = os.path.join(ct_register_dir,'true_ct2mr.lta')

    if os.path.exists(lta) and not overwrite:
        print 'using existing CT to MR transformation'

        #perform some temporary corrections
#        zf = 1.43
#        lta = geo.get_lta(lta)
#        skew_mat = np.eye(4)
#        skew_mat[2,2] = zf
#
#        return np.dot(lta, skew_mat)

        #return geo.get_lta(lta)
        return np.loadtxt(lta)

    #determine the file extension
    file_ext = os.path.splitext(ct)[-1]
    if file_ext in ('.gz', '.nii'):
        image_factory = nib.Nifti1Image
    else:
    #elif file_ext in ('.mgz', '.mgh'):
        #image_factory = nib.MGHImage

    #use a hack to convert the file back to NIFTI. Even the MGH file type seems
    #to be behaving erratically, always has the wrong slice thickness in the
    #header and so on
        image_factory = nib.Nifti1Image
        new_ct = os.path.join(ct_register_dir, 'ct_as_nifti.nii.gz')
        mri_convert_cmd = ['mri_convert', ct, new_ct]
        subprocess.call(mri_convert_cmd)

        ct = new_ct

    #else:
    #    raise ValueError('CT image has invalid type, must be NIFTI or MGH')

    # pick 2 slices an arbitrary distance apart
    cti = nib.load(ct)
    z = cti.shape[2]
    
    center_z = z//2
    hdr = cti.get_header()

    vox_sz = hdr.get_zooms()
    iso_sz = vox_sz[0]
    slice_thickness = vox_sz[2]
    hdr.set_zooms((iso_sz,)*3)

    ct_slices = cm_dist * 10 / slice_thickness
    upper_z = center_z - ct_slices

    #import pdb
    #pdb.set_trace()

    print 'loading CT data'
    ctd = cti.get_data()

    orig = os.path.join(subjects_dir, subject, 'mri', 'orig.mgz')
    rawavg = os.path.join(subjects_dir, subject, 'mri', 'rawavg.mgz')

    if zf_override == None:
        center_slice = np.zeros(cti.shape)
        center_slice[:,:,center_z] = ctd[:,:,center_z]
        upper_slice = np.zeros(cti.shape)
        upper_slice[:,:,upper_z] = ctd[:,:,upper_z] 

        center_img = nib.Nifti1Image(center_slice, affine=cti.get_affine(),
            header=hdr)
        upper_img = nib.Nifti1Image(upper_slice, affine=cti.get_affine(),
            header=hdr)

        center_fname = os.path.join( ct_register_dir, 
            'center_ct_slice.nii.gz')
        upper_fname = os.path.join( ct_register_dir, 'upper_ct_slice.nii.gz')

        print 'saving CT slices'
        nib.save(center_img, center_fname)
        nib.save(upper_img, upper_fname)

        #register orig independently to these two slices
        import tempfile

        center_reg_orig = os.path.join( ct_register_dir, 
            'center_orig.nii.gz')
        upper_reg_orig = os.path.join( ct_register_dir, 'upper_orig.nii.gz')

        center_to_orig_lta = os.path.join( ct_register_dir, 'c2o.lta')

        _,gbg = tempfile.mkstemp()
        print 'registering orig to slices'

        mri_robustreg_cslice_cmd = ['mri_robust_register', '--mov', orig, 
            '--dst',
            center_fname, '--lta', center_to_orig_lta, '--satit', '--cost',
            'mi',
            '--nosym', '--mapmovhdr', center_reg_orig]
        p = subprocess.call(mri_robustreg_cslice_cmd)
         
        mri_robustreg_uslice_cmd = ['mri_robust_register', '--mov', orig, 
            '--dst',
            upper_fname, '--lta', gbg, '--satit', '--cost', 'mi', '--nosym',
            '--mapmovhdr', upper_reg_orig, '--ixform', center_to_orig_lta,
            '--maxsize', '128']
        q = subprocess.call(mri_robustreg_uslice_cmd) 

        os.unlink(gbg)

        #register the two translated origs to each other
        print 'registering slices to each other'
        translate_lta = os.path.join( ct_register_dir, 'u2c_translation.lta')
        
        mri_robustreg_trans_cmd = ['mri_robust_register', '--mov', 
            upper_reg_orig,
            '--dst', center_reg_orig, '--lta', translate_lta, '--satit', 
            '--nosym',
            ]
        r = subprocess.call(mri_robustreg_trans_cmd)

        #calculate the desired zoom factor
        #import pdb
        #pdb.set_trace()
        #translate_affine = geo.get_lta(translate_lta)
        #n = np.abs( translate_affine[2,3] )

        #calculate n
        center_orig_f = nib.load(center_reg_orig)
        upper_orig_f = nib.load(upper_reg_orig)
        coa = center_orig_f.get_affine()
        uoa = upper_orig_f.get_affine()

        from scipy.linalg import inv
        uoai = inv(uoa)

        pOrigin = (128, 128, 128, 1)

        pShifted = np.dot(uoai, np.dot(coa, pOrigin))

        n = np.abs(pOrigin[1]-pShifted[1])

        #calculate x, the distance covered in isotropic coordinates of 50
        #millimeters

        #we can prove that x is equal to this number by finding the RAS
        #coordinates of the translation using the qform matrix
        x = iso_sz * cm_dist * 10

        zoom_factor = (1, 1, (x+n)/x )

    else:
        zoom_factor = zf_override

    #resample the ct image and do the final registration

    from scipy.ndimage.interpolation import zoom
    print 'resampling image with zoom_factor {0}'.format(zoom_factor)
    #ct_zoom = zoom( ctd, (1,1,zoom_factor))
    ct_zoom = zoom( ctd, 1/np.array(zoom_factor) )

    resampled_ct = os.path.join(ct_register_dir, 'ct_resampled_zf.nii.gz')

    #resamp_img = nib.Nifti1Image(ct_zoom, affine=cti.get_affine(), 
    #    header=hdr)
    resamp_img = image_factory(ct_zoom, affine=cti.get_affine(),
        header=hdr)
    nib.save(resamp_img, resampled_ct)

    ct_final = os.path.join(ct_register_dir, 'ct_final_resamp_reg.nii.gz')

    mri_robustreg_resampled_cmd = ['mri_robust_register', '--mov', 
        resampled_ct, '--dst', rawavg, '--lta', skewed_lta, '--satit',
        '--vox2vox', '--mapmov', ct_final, '--cost', 'mi']
    s = subprocess.call(mri_robustreg_resampled_cmd)

    skew_aff = geo.get_lta(skewed_lta)
    unskew_aff = np.eye(4)
    for j in xrange(3):
        unskew_aff[j,j] = zoom_factor[j]
    aff = np.dot(skew_aff, unskew_aff)
    np.savetxt(lta, aff)

    return aff

def create_dural_surface(subjects_dir=None, subject=None):
    '''
    Creates the dural surface in the specified subjects_dir. This is done
    using a standalone script derived from the Freesurfer tools which actually
    use the dural surface.

    The caller is responsible for providing a correct subjects_dir, i.e., one
    which is writable. The higher-order logic should detect an unwritable
    directory, and provide a user-sanctioned space to write the new fake
    subjects_dir to.

    Parameters
    ----------
    subjects_dir : Str | None
        The freesurfer subjects_dir. If this is None, it is assumed to be the
        $SUBJECTS_DIR environment variable. If this folder is not writable,
        the program will crash.
    subject : Str | None
        The freesurfer subject. If this is None, it is assumed to be the
        $SUBJECT environment variable.
    '''
    print 'create dural surface step'
    if subjects_dir is None or subjects_dir=='':
        subjects_dir = os.environ['SUBJECTS_DIR']
    if subject is None or subject=='':
        subject = os.environ['SUBJECT']

    scripts_dir = os.path.dirname(__file__)
    os.environ['SCRIPTS_DIR'] = scripts_dir

    print scripts_dir

    if (os.path.exists(os.path.join(subjects_dir,subject,'surf','lh.dural'))
            and os.path.exists(os.path.join(subjects_dir, subject,'surf',
            'rh.dural'))):
        print 'dural surfaces already exist'
        return

    import subprocess

    curdir = os.getcwd()
    os.chdir(os.path.join(subjects_dir, subject, 'surf'))

    for hemi in ('lh','rh'):
        make_dural_surface_cmd = [os.path.join(scripts_dir, 
            'make_dural_surface.csh'),'-i','%s.pial'%hemi]
        p=subprocess.call(make_dural_surface_cmd)

    os.chdir(curdir)

def get_rawavg_to_orig_xfm(subjects_dir=None, subject=None, 
        skip_rawavg_to_orig=False):
    '''
    Collect the transformation

    Parameters
    ----------
    subjects_dir : Str | None
        The freesurfer subjects_dir. If this is None, it is assumed to be the 
        $SUBJECTS_DIR environment variable.
    subject : Str | None
        The freesurfer subject. If this is None, it is assumed to be the
        $SUBJECT environment variable.
    skip_rawavg_to_orig : bool
        Do not register rawavg to orig. Suitable for manual registration
        directly to orig, excluding rawavg as intermediate step. Not
        recommended in automated registration.
    '''
    if subjects_dir is None or subjects_dir=='':
        subjects_dir = os.environ['SUBJECTS_DIR']
    if subject is None or subject=='':
        subject = os.environ['SUBJECT']

    rawavg = os.path.join(subjects_dir, subject, 'mri', 'rawavg.mgz')
    orig = os.path.join(subjects_dir, subject, 'mri', 'orig.mgz')
    lta = os.path.join(subjects_dir, subject, 'mri', 'transforms',
        'raw2orig.lta')

    if not os.path.exists(lta) and not skip_rawavg_to_orig:
        import subprocess
        mri_robustreg_cmd = ['mri_robust_register','--mov',rawavg,'--dst',
            orig,'--lta',lta,'--satit','--vox2vox']
        p = subprocess.call(mri_robustreg_cmd)
    elif not os.path.exists(lta) and skip_rawavg_to_orig:
        raise NotImplementedError("Please contact the developer with this error")

    rawavg2orig = geo.get_lta(lta)
    return rawavg2orig

def translate_electrodes_to_surface_space(electrodes, ct2mr,
    subjects_dir=None, subject=None, affine=None):
    '''
    Translates electrodes from CT space to orig space, and then from
    orig space to surface space.

    Parameters
    ----------
    electrodes : List(Electrode)
        List of electrodes in CT space
    ct2mr : 4x4 np.ndarray
        Matrix containing the ct2mr affine transformation
    subjects_dir : Str | None
        The freesurfer subjects_dir. If this is None, it is assumed to be the 
        $SUBJECTS_DIR environment variable.
    subject : Str | None
        The freesurfer subject. If this is None, it is assumed to be the
        $SUBJECT environment variable.
    skip_rawavg_to_orig : bool
        Do not register rawavg to orig. Suitable for manual registration
        directly to orig, excluding rawavg as intermediate step. Not
        recommended in automated registration.

    There is no return value. The 'surf_coords' attribute will be used to
    store the surface locations of the electrodes
    '''
    print 'translating electrodes to surface space'

    if subjects_dir is None or subjects_dir=='':
        subjects_dir = os.environ['SUBJECTS_DIR']
    if subject is None or subject=='':
        subject = os.environ['SUBJECT']

    if len(electrodes) == 0:
        raise ValueError('No electrodes to translate to surface space')

    electrode_arr = map((lambda x:getattr(x, 'ct_coords')), electrodes)
    orig_elecs = geo.apply_affine(electrode_arr, ct2mr)

    orig = os.path.join(subjects_dir, subject, 'mri', 'orig.mgz')

    #if skip_rawavg_to_orig:
    #    nas2ras = np.eye(4)
    #else:
    if True:
        nas2ras = get_rawavg_to_orig_xfm(subject=subject, 
                                         subjects_dir=subjects_dir)

    nas_locs = geo.apply_affine(orig_elecs, nas2ras)

    tkr = geo.get_vox2rasxfm(orig, stem='vox2ras-tkr')

    surf_locs = geo.apply_affine(nas_locs, tkr)
    for elec, loc in zip(electrodes, surf_locs):
        elec.surf_coords = loc

def snap_electrodes_to_surface(electrodes, subjects_dir=None, 
    subject=None, max_steps=40000, giveup_steps=10000, 
    init_temp=1e-3, temperature_exponent=1,
    deformation_constant=1.):
    '''
    Transforms electrodes from surface space to positions on the surface
    using a simulated annealing "snapping" algorithm which minimizes an
    objective energy function as in Dykstra et al. 2012

    Parameters
    ----------
    electrodes : List(Electrode)
        List of electrodes with the surf_coords attribute filled. Caller is
        responsible for filtering these into grids if desired.
    subjects_dir : Str | None
        The freesurfer subjects_dir. If this is None, it is assumed to be the 
        $SUBJECTS_DIR environment variable. Needed to access the dural
        surface.
    subject : Str | None
        The freesurfer subject. If this is None, it is assumed to be the
        $SUBJECT environment variable. Needed to access the dural surface.
    max_steps : Int
        The maximum number of steps for the Simulated Annealing algorithm.
        Adding more steps usually causes the algorithm to take longer. The
        default value is 40000. max_steps can be smaller than giveup_steps,
        in which case giveup_steps is ignored
    giveup_steps : Int
        The number of steps after which, with no change of objective function,
        the algorithm gives up. A higher value may cause the algorithm to
        take longer. The default value is 10000.
    init_temp : Float
        The initial annealing temperature. Default value 1e-3
    temperature_exponent : Float
        The exponentially determined temperature when making random changes.
        The value is Texp0 = 1 - Texp/H where H is max_steps
    deformation_constant : Float
        A constant to weight the deformation term of the energy cost. When 1,
        the deformation and displacement are weighted equally. When less than
        1, there is assumed to be considerable deformation and the spring
        condition is weighted more highly than the deformation condition.

    There is no return value. The 'snap_coords' attribute will be used to
    store the snapped locations of the electrodes
    '''
    if subjects_dir is None or subjects_dir=='':
        subjects_dir = os.environ['SUBJECTS_DIR']
    if subject is None or subject=='':
        subject = os.environ['SUBJECT']

    from scipy.spatial.distance import cdist

    n = len(electrodes)
    electrode_arr = map((lambda x:getattr(x, 'surf_coords')), electrodes)
    e_init = np.array(electrode_arr)

    # first set the alpha parameter exactly as described in Dykstra 2012.
    # this parameter controls which electrodes have virtual springs connected.
    # this may not matter but doing it is fast and safe
    alpha = np.zeros((n,n))
    init_dist = cdist(e_init, e_init)

    neighbors = []

    for i in xrange(n):
        neighbor_vec = init_dist[:,i]
        #take 5 highest neighbors
        h5, = np.where(np.logical_and(neighbor_vec<np.sort(neighbor_vec)[5],
            neighbor_vec != 0))

        neighbors.append( h5 )

    neighbors = np.squeeze(neighbors)

    # get distances from each neighbor pairing
    neighbor_dists = []
    for i in xrange(n):
        neighbor_dists.append( init_dist[i, neighbors[i]] )

    neighbor_dists = np.hstack(neighbor_dists)

    #collect distance into histogram of resolution 0.2
    max = np.max( np.around(neighbor_dists) )
    min = np.min( np.around(neighbor_dists) )

    hist,_ = np.histogram(neighbor_dists, bins=(max-min)/2, range=(min, max))

    fundist = np.argmax(hist)*2 + min + 1

    #apply fundist to alpha matrix
    alpha_tweak = 1.75

    for i in xrange(n):
        neighbor_vec = init_dist[:,i]
        neighbor_vec[i] = np.inf

        neighbors = np.where( neighbor_vec < fundist*alpha_tweak )

        if len(neighbors) > 5:
            neighbors = np.where( neighbor_vec < np.sort(neighbor_vec)[5] )

        if len(neighbors) == 0:
            closest = np.argmin( neighbors )
            neighbors = np.where( neighbor_vec < closest*alpha_tweak )

        alpha[i,neighbors]=1

        for j in xrange(i):
            if alpha[j,i]==1:
                alpha[i,j]=1 
            if alpha[i,j]==1:
                alpha[j,i]=1

    # alpha is set, now do the annealing
    def energycost(e_new, e_old, alpha):
        n = len(alpha)

        dist_new = cdist(e_new, e_new)
        dist_old = cdist(e_old, e_old)

        H=0

        for i in xrange(n):
            H += deformation_constant*float(cdist( [e_new[i]], [e_old[i]] ))

            for j in xrange(i):
                H += alpha[i,j] * (dist_new[i,j] - dist_old[i,j])**2

        return H

    #load the dural surface locations
    lh_dura, _ = nib.freesurfer.read_geometry(
        os.path.join(subjects_dir, subject, 'surf', 'lh.dural'))

    rh_dura, _ = nib.freesurfer.read_geometry(
        os.path.join(subjects_dir, subject, 'surf', 'rh.dural'))

    #lh_dura[:, 0] -= np.max(lh_dura[:, 0])
    #rh_dura[:, 0] -= np.min(rh_dura[:, 0])

    #align the surfaces correctly
    #in the tkRAS space
    #orig = os.path.join( subjects_dir, subject, 'mri', 'orig.mgz' )
    #ras2vox = np.linalg.inv(geo.get_vox2rasxfm( orig ))
    #tkr = geo.get_vox2rasxfm(orig, 'vox2ras-tkr')
    #lh_dura = np.array( geo.apply_affine( geo.apply_affine(lh_dura, ras2vox), 
    #    tkr))
    #rh_dura = np.array( geo.apply_affine( geo.apply_affine(rh_dura, ras2vox), 
    #    tkr))
    #lh_dura = geo.apply_affine( geo.apply_affine(lh_dura, ras2vox), tkr)
    #rh_dura = geo.apply_affine( geo.apply_affine(rh_dura, ras2vox), tkr)

    dura = np.vstack((lh_dura, rh_dura))

    max_deformation = 3
    deformation_choice = 50

    #adjust annealing parameters
    # H determines maximal number of steps
    H = max_steps
    #Texp determines the steepness of temperateure gradient
    Texp=1-temperature_exponent/H
    #T0 sets the initial temperature and scales the energy term
    T0 = init_temp
    #Hbrk sets a break point for the annealing
    Hbrk = giveup_steps

    h=0; hcnt=0
    lowcost = mincost = 1e6

    #start e-init as greedy snap to surface
    e_snapgreedy = dura[np.argmin(cdist(dura, e_init), axis=0)]

    e = np.array(e_snapgreedy).copy()
    emin = np.array(e_snapgreedy).copy()

    #the annealing schedule continues until the maximum number of moves
    while h<H:
        h+=1; hcnt+=1
        #terminate if no moves have been made for a long time
        if hcnt>Hbrk:
            break

        #current temperature 
        T=T0*(Texp**h)

        #select a random electrode
        e1 = np.random.randint(n)
        #transpose it with a *nearby* point on the surface

        #find distances from this point to all points on the surface
        dists = np.squeeze(cdist(dura, [e[e1]]))
        #take a distance within the minimum 5X

        #mindist = np.min(dists) 
        mindist = np.sort(dists)[deformation_choice]
        candidate_verts, = np.where( dists < mindist*max_deformation )
        choice_vert = candidate_verts[np.random.randint(len(candidate_verts))]
        
        e_tmp = e.copy()
        #print choice_vert
        #print np.shape(candidate_verts)
        e_tmp[e1] = dura[choice_vert]

        cost = energycost(e_tmp, e_init, alpha)

        if cost < lowcost or np.random.random()<np.exp(-(cost-lowcost)/T):
            e = e_tmp 
            lowcost = cost

            if cost < mincost:
                emin = e
                mincost = cost
                print 'step %i ... current lowest cost = %f' % (h, mincost)
                hcnt = 0

            if mincost==0:
                break

        print 'step %i ... final lowest cost = %f' % (h, mincost)

    #return the emin coordinates
    for elec, loc in zip(electrodes, emin):
        elec.snap_coords = loc

    #return the nearest vertex on the pial surface 
    lh_pia, _ = nib.freesurfer.read_geometry(
        os.path.join(subjects_dir, subject, 'surf', 'lh.pial'))
    
    rh_pia, _ = nib.freesurfer.read_geometry(
        os.path.join(subjects_dir, subject, 'surf', 'rh.pial'))

    #expand the pial surfaces slightly to better visualize the electrodes
    #lh_pia =geo.expand_triangular_mesh(lh_pia, com_bias=(-2, 0, 0), offset=18)
    #rh_pia = geo.expand_triangular_mesh(rh_pia, com_bias=(2, 0, 0), offset=18)


    #adjust x-axis offsets as pysurfer illogically does as hard-coded step
    #lh_pia[:, 0] -= np.max(lh_pia[:, 0])
    #rh_pia[:, 0] -= np.min(rh_pia[:, 0])


    pia = np.vstack((lh_pia, rh_pia))

    e_pia = np.argmin(cdist(pia, emin), axis=0)

    for elec, soln in zip(electrodes, e_pia):
        elec.vertno = soln if soln<len(lh_pia) else soln-len(lh_pia)
        elec.hemi = 'lh' if soln<len(lh_pia) else 'rh'
        elec.pial_coords = pia[soln]


def fit_grid_to_line(electrodes, mindist=0, maxdist=36, epsilon=30, delta=.5,
    rho=35, rho_strict=20, rho_loose=50):
    '''
    Given a list of electrodes and two endpoints of a line, fit the electrodes
    onto the Nx1 or 1xN line using a greedy fitting procedure. Set the
    geom_coords attribute of the fitted electrodes.

    Parameters
    ----------
    electrodes : List(Electrodes)
        List of electrodes in the specified strip

    No return value
    '''
    electrode_arr = map((lambda x:getattr(x, 'iso_coords')), electrodes)

    #find most isolated point
    eadist = cdist(electrode_arr, electrode_arr)
    
    isol = np.argmax(np.sum(eadist, axis=0))
    next_isol = np.argmin(np.ma.masked_array(eadist[isol], 
        mask=(eadist[isol]==0)))

    c1 = np.array(electrodes[isol].asiso())
    c2 = np.array(electrodes[next_isol].asiso())

    pog = gl.Grid(c1, c2, None, np.array(electrode_arr), delta=delta,
        rho=rho, rho_strict=rho_strict, rho_loose=rho_loose, is_line=True)

    pog.extend_grid_arbitrarily()

    if len(pog.points) < len(electrodes):
        raise SortingLabelingError('Failed to fit all the electrodes')

    miny=0
    for elec in electrodes:
        y = pog.connectivity[gl.GridPoint(elec.iso_coords)][1]
        if y<miny:
            miny=y

    for elec in electrodes:
        conn = pog.connectivity[gl.GridPoint(elec.iso_coords)]
        elec.geom_coords = [0, conn[1]-miny]

def fit_grid_by_fixed_points(electrodes, geom, 
    delta=.35, rho=35, rho_strict=20, rho_loose=50, 
    epsilon=10, mindist=0, maxdist=36):
    '''
    Sort the given electrodes (generally in the space of the CT scan) into
    grids and strips matching the specified geometry.
    
    This function is not currently used. The idea was to have the user
    fix some errors of the algorithm and then send it back. But really this
    step is completely unnecessary because it is not appreciably more work for
    the user to do the entire job manually. As the software ages this
    function will be removed.

    Parameters
    ----------
    electrodes : List(Electrode)
        A list of electrode locations. The CT coordinate attribute of the
        electrodes is used as the position.
        It is the caller's responsibility to filter the electrodes list as
        appropriate.
    geom : List(2x1)
        A 2x1 vector describing the grid geometry
    delta : Float
        A fitting parameter that controls the relative distance between
        grid points. A grid point cannot be farther than delta*c from its
        orthogonal neighbors, where c is an estimate of the distance between
        grid neighbors, assuming a roughly square grid (Later, this should be
        a rectangular grid). The default value is .35
    rho : Float
        A fitting parameter controlling the distance from which successive
        angles can diverge from 90. The default value is 35
    rho_strict : Float
        A fitting parameter similar to rho but used in different geometric
        circumstances. The default value is 20.
    rho_loose : Float
        A fitting parameter similar to rho but used in different geometric
        circumstances. The default value is 50.
    epsilon : Float
        A fitting parameter controlling the acceptable deviation from 90
        degrees for the starting point of a KxM grid where K>1,M>1. A
        larger parameter means the algorithm will try a larger range of
        starting positions before giving up. The default value is 10.
    '''
    electrode_arr = map((lambda x:getattr(x, 'iso_coords')), electrodes)
    elecs = np.array(electrode_arr)


    angles, _, neighbs = gl.find_init_angles(elecs, mindist=mindist,
        maxdist=maxdist)

    ba = np.squeeze(sorted(zip(*np.where(np.abs(90-angles)<epsilon)),
            key=lambda v:np.abs(90-angles[v])))

    if ba.shape==():
        ba=[ba]
    elif len(ba)==0:
        raise SortingLabelingError("Could not find any good angles")

    for j,k in enumerate(ba):
        p0,p1,p2 = neighbs[k]
        pog = gl.Grid(p0, p1, p2, elecs, delta=delta, rho=rho,
            rho_strict=rho_strict, rho_loose=rho_loose)

        pog.critical_percentage = 1.
            
        try:
            pog.recreate_geometry( )
        except SortingLabelingError as e:
            print 'Could not recreate geometry with this initialization'
            continue

        try:
            sp, corners, final_connectivity = pog.extract_strip(*geom)
        except SortingLabelingError as e:
            print 'Rejected this choice'
            if j==len(ba)-1:
                raise ValueError("Could not incorporate fixed points")
            continue

        for p in sp:
            if tuple(p.tolist()) in electrode_arr:
                ix, = np.where(np.logical_and(np.logical_and( 
                    np.array(electrode_arr)[:,0]==p[0], 
                    np.array(electrode_arr)[:,1]==p[1]),
                    np.array(electrode_arr)[:,2]==p[2]))
                try:
                    elec = electrodes[ix]
                except IndexError:
                    raise SortingLabelingError(
                        "multiple electrodes at same point")
            else:
                raise SortingLabelingError(
                    'Electrodes in not same as electrodes out')

            for corner in corners:
                if np.all(corner==np.array(elec.asiso())):
                    elec.corner = ['corner 1']

            try:
                elec.geom_coords = list(final_connectivity[
                    elec.asiso()])
            except KeyError:
                pass

        break


def fit_grid_to_plane(electrodes, c1, c2, c3, geom, reverse_grid='check'):
    '''
    Given a list of electrodes and three corners of a plane, fit the
    electrodes onto the plane using a snapping algorithm minimizing a global
    cost function. Use a relatively rapid cooling schedule. Set the
    geom_coords attribute of the fitted electrodes.

    Parameters
    ----------
    electrodes : List(Electrode)
        List of electrodes in the specified grid
    c1, c2, c3 : Tuple
        Tuple containing coordinates (in CT space) of the corner electrodes
        the user selected
    geom : 2-Tuple
        The known geometry of this grid 
    reverse_grid : Bool | 'check'
        Bool to indicate that the X is the maximum geom value and Y is
        minimum. If False, Y is the maximum. If 'check', checks the grids
        current geometry information for this, which could conceivably fail
        if there are not enough points filled in.

    No return value
    '''
    #a,b,c,d = geo.find_plane_from_corners(c1, c2, c3)
    from scipy.spatial.distance import cdist, pdist

    c1 = np.array(c1)
    c2 = np.array(c2)
    c3 = np.array(c3)

    v1 = c2-c1
    v2 = c3-c1

    #import pdb
    #pdb.set_trace()

    c4 = c2+c3-c1

    plane = {}
    reverse_plane = {}
    xg = max(geom)-1
    ng = min(geom)-1
    for i in xrange(max(geom)):
        for j in xrange(min(geom)):
            if np.sum(v1**2) >= np.sum(v2**2):
                #longer side in direction of c2 
                s1 = c2*i/xg + c1*(xg-i)/xg
                s2 = c3*j/ng + c1*(ng-j)/ng
                pN = s1+s2-c1
            else:
                #longer side in direction of c3
                s1 = c3*i/xg + c1*(xg-i)/xg
                s2 = c2*j/ng + c1*(ng-j)/ng
                pN = s1+s2-c1
            plane[tuple(pN)]=(i,j)
            reverse_plane[(i,j)] = pN

    plane_points = np.array(plane.keys())

    #check transposition
    if reverse_grid == True:
        transpix = [1, 0]
    elif reverse_grid == False:
        transpix = [0, 1]
    elif reverse_grid == 'check':
        max_x = 0
        max_y = 0
        for elec in electrodes:
            if len(elec.geom_coords) == 0:
                continue

            x,y = elec.geom_coords
            if x > max_x:
                max_x = x
            if y > max_y:
                max_y = y

        if max_x >= max_y:
            transpix = [0, 1]
        else:
            transpix = [1, 0]
    else:
        raise BCTParamError('Invalid value of reverse_grid')

#    #assign the electrodes to the nearest plane point greedily
#    pp = {}
#    for elec in electrodes:
#        e_greedy = np.argmin(cdist([elec.asiso()], plane_points))
#        pp[elec.iso_coords] = plane_points[e_greedy]
#        plane_points = np.delete(plane_points, e_greedy, axis=0)

    #assign the electrodes to the nearest plane point depending on their
    #existing geometry prediction
    pp = {}
    for elec in electrodes:
        if len(elec.geom_coords) == 0:
            continue

        e_init_choice = reverse_plane[tuple(
            np.array(elec.geom_coords)[transpix])]
        pp[elec.asiso()] = e_init_choice
        e_init_ix = np.argmin(cdist([e_init_choice], plane_points))
        plane_points = np.delete(plane_points, e_init_ix, axis=0)

    #assign remaining electrodes greedily
    for elec in electrodes:
        if len(elec.geom_coords) != 0:
            continue

        e_greedy_choice = np.argmin(cdist([elec.asiso()], plane_points))
        pp[elec.asiso()] = plane_points[e_greedy_choice]
        plane_points = np.delete(plane_points, e_greedy_choice, axis=0)

#    pp_min = pp.copy()
#
#    deformation_constant = 0
#    adjacency_constant = 1
#
#    def globalcost(pp):
#        c=0
#    
#        #deformation term
#        for e in pp:
#            pe = pp[e]
#            c += deformation_constant*pdist((pe,e))**2
#
#        #adjacency term
#        for i in xrange(max(geom)):
#            for j in xrange(min(geom)):
#
#                pij = reverse_plane[(i,j)]
#                if i+1 < max(geom):
#                    px = reverse_plane[(i+1,j)]
#                    c += adjacency_constant*pdist((pij,px))**2
#
#                if j+1 > max(geom):
#                    py = reverse_plane[(i,j+1)]
#                    c += adjacency_constant*pdist((pij,py))**2
#
#        return c
#
#    # H determines maximal number of steps
#    H = 20000
#    #Texp determines the steepness of temperateure gradient
#    Texp=1-1/H
#    #T0 sets the initial temperature and scales the energy term
#    T0 = 1e-5
#    #Hbrk sets a break point for the annealing
#    Hbrk = 20000
#    
#    h=0; hcnt=0
#    lowcost = mincost = 1e16
#
#    while h<H:
#        h+=1; hcnt+=1
#
#        T=T0*(Texp**h)
#
#        #this doesnt even really allow for an search space
#        #since the set of possible changes is hopelessly constrained to 
#        #nearby swaps
#
#
#        e1 = np.random.randint(len(pp))
#        #cand_pt = np.argmin(cdist([electrodes[e1].asct()], pp.values()))
#        cand_pt = np.argmin(cdist([electrodes[e1].asiso()], pp.values()))
#
#        if cand_pt==e1:
#            continue
#        #eg = plane[tuple(pp[electrodes[e1].asct()])]
#        #cg = plane[tuple(pp[electrodes[cand_pt].asct()])]
#        eg = plane[tuple(pp[electrodes[e1].asiso()])]
#        cg = plane[tuple(pp[electrodes[cand_pt].asiso()])]
#        if np.abs(eg[0]-cg[0]) > 2 or np.abs(eg[1]-cg[1]) > 2:
#            continue
#
#        pp_tmp = pp.copy()
#        #old_e1 = pp_tmp[electrodes[e1].asct()]
#        #old_cp = pp_tmp[electrodes[cand_pt].asct()]
#
#        #pp_tmp[electrodes[cand_pt].asct()]=old_e1
#        #pp_tmp[electrodes[e1].asct()]=old_cp
#
#        old_e1 = pp_tmp[electrodes[e1].asiso()]
#        old_cp = pp_tmp[electrodes[cand_pt].asiso()]
#
#        pp_tmp[electrodes[cand_pt].asiso()]=old_e1
#        pp_tmp[electrodes[e1].asiso()]=old_cp
#
#        cost = globalcost(pp_tmp)
#        if cost < lowcost or np.random.random()<np.exp(-(cost-lowcost)/T):
#            pp = pp_tmp
#            lowcost = cost
# 
#            if cost < mincost:
#                pp_min = pp
#                mincost = cost
#                print 'step %i in plane fitting, cost %f' %(h, mincost)
#                
#    print 'Finished plane fitting, final cost %f' % mincost

    for elec in electrodes:
        elec.geom_coords = list(plane[tuple(pp[elec.iso_coords])])

def identify_roi_from_atlas( pos, approx=4, atlas=None, subjects_dir=None,
    subject=None ):
    '''
    Find the surface labels contacted by an electrode at this position
    in RAS space.

    Parameters
    ----------

    pos : np.ndarray
        1x3 matrix holding position of the electrode to identify
    approx : int
        Number of millimeters error radius
    atlas : str or None
        The string containing the name of the surface parcellation,
        does not apply to subcortical structures. If None, aparc is used.
    '''
    if subjects_dir is None or subjects_dir=='':
        subjects_dir = os.environ['SUBJECTS_DIR']
    if subject is None or subject=='':
        subject = os.environ['SUBJECT']

    if atlas is None or atlas in ('', 'aparc'):
        return identify_roi_from_aparc(pos, approx=approx, 
            subjects_dir=subjects_dir, subject=subject)

    from scipy.spatial.distance import cdist
    # conceptually, we should grow the closest vertex around this electrode
    # probably following snapping but the code for this function is not
    # altered either way

    # load the surfaces and annotation
    # uses the pial surface, this change is pushed to MNE python

    lh_pia, _ = nib.freesurfer.read_geometry(
        os.path.join(subjects_dir, subject, 'surf', 'lh.pial'))

    rh_pia, _ = nib.freesurfer.read_geometry(
        os.path.join(subjects_dir, subject, 'surf', 'rh.pial'))

    pia = np.vstack((lh_pia, rh_pia))

    # find closest vertex
    #import pdb
    #pdb.set_trace()
    closest_vert = np.argmin(cdist(pia, [pos]))

    # grow the area of surface surrounding the vertex
    import mne

    # we force the label to only contact one hemisphere even if it is
    # beyond the extent of the medial surface
    hemi_str = 'lh' if closest_vert<len(lh_pia) else 'rh'
    hemi_code = 0 if hemi_str=='lh' else 1

    if hemi_str == 'rh':
        closest_vert -= len(lh_pia)

    radius_label, = mne.grow_labels(subject, closest_vert, approx, hemi_code,
        subjects_dir=subjects_dir, surface='pial')

    parcels = mne.read_labels_from_annot(subject, parc=atlas, hemi=hemi_str,
        subjects_dir=subjects_dir, surf_name='pial')

    regions = []
    for parcel in parcels:
        if len(np.intersect1d(parcel.vertices, radius_label.vertices))>0:
            #force convert from unicode
            regions.append(str(parcel.name))
       
    subcortical_regions = identify_roi_from_aparc(pos, approx=approx,
        
        subjects_dir=subjects_dir, subject=subject, subcortical_only=True)

    regions.extend(subcortical_regions)

    return regions

def identify_roi_from_aparc( pos, approx=4, subjects_dir=None, subject=None,
    subcortical_only = False):
    '''
    Find the volumetric labels contacted by an electrode at this position
    in RAS space.

    Parameters
    ----------

    pos : np.ndarray
        1x3 matrix holding position of the electrode to identify
    approx : int
        Number of millimeters error radius
    subcortical_only : bool
        if True, exclude cortical labels
    '''
    if subjects_dir is None or subjects_dir=='':
        subjects_dir = os.environ['SUBJECTS_DIR']
    if subject is None or subject=='':
        subject = os.environ['SUBJECT']

    def find_neighboring_regions(pos, mri_dat, region, approx, excludes):
        spot_sz = int(np.around(approx * 2 + 1))
        x, y, z = np.meshgrid(range(spot_sz), range(spot_sz), range(spot_sz))

        # approx is in units of millimeters as long as we use the RAS space
        # segmentation
        neighb = np.vstack((np.reshape(x, (1, spot_sz ** 3)),
            np.reshape(y, (1, spot_sz ** 3)),
            np.reshape(z, (1, spot_sz ** 3)))).T - approx

        regions = []
    
        #import pdb
        #pdb.set_trace()

        for p in xrange(neighb.shape[0]):
            cx, cy, cz = (pos[0]+neighb[p,0], pos[1]+neighb[p,1],
                pos[2]+neighb[p,2])
            d_type = mri_dat[cx, cy, cz]
            label_index = region['index'].index(d_type)
            regions.append(region['label'][label_index])

        if excludes:
            from re import compile
            excluded = compile('|'.join(excludes))
            regions = [x for x in regions if not excluded.search(x)]

        return np.unique(regions).tolist()

    def import_freesurfer_lut(fs_lut=None):
        """
        Import Look-up Table with colors and labels for anatomical regions.
        It's necessary that Freesurfer is installed and that the environmental
        variable 'FREESURFER_HOME' is present.
        
        Parameters
        ----------
        fs_lut : str
            path to file called FreeSurferColorLUT.txt

        Returns
        -------
        idx : list of int
            indices of regions
        label : list of str
            names of the brain regions
        rgba : numpy.ndarray
            one row is a brain region and the columns are the RGBA colors
        """
        if fs_lut is None:
            try:
                fs_home = os.environ['FREESURFER_HOME']
            except KeyError:
                raise OSError('FREESURFER_HOME not found')
            else:
                fs_lut = os.path.join(fs_home, 'FreeSurferColorLUT.txt')

        idx = []
        label = []
        rgba = np.empty((0, 4))

        with open(fs_lut, 'r') as f:
            for l in f:
                if len(l) <= 1 or l[0] == '#' or l[0] == '\r':
                    continue
                (t0, t1, t2, t3, t4, t5) = [t(s) for t, s in
                        zip((int, str, int, int, int, int), l.split())]

                idx.append(t0)
                label.append(t1)
                rgba = np.vstack((rgba, np.array([t2, t3, t4, t5])))

        return idx, label, rgba

    # get the segmentation file
    asegf = os.path.join( subjects_dir, subject, 'mri', 'aparc+aseg.mgz' )
    aseg = nib.load(asegf)
    asegd = aseg.get_data()

    # get the aseg LUT file
    lut = import_freesurfer_lut()
    lut = {'index':lut[0], 'label':lut[1], 'RGBA':lut[2]}
    
    excludes = ['white', 'WM', 'Unknown', 'White', 'unknown']
    if subcortical_only:
        excludes.append('ctx')


    RAS_AFF = np.array([[-1, 0, 0, 128],
        [0, 0, -1, 128],
        [0, 1, 0, 128],
        [0, 0, 0, 1]])
    ras_pos = np.around(np.dot(RAS_AFF, np.append(pos, 1)))[:3]

    return find_neighboring_regions(ras_pos, asegd, lut, approx, excludes)
