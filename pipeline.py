#
# The code in identify_roi_from_aparc originally comes from Gio Piantoni
#

from __future__ import division
import os
import numpy as np
import nibabel as nib
import geometry as geo
import grid as gl
from electrode import Electrode

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
    use_erosion=True):
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

    Returns
    -------
    electrodes : List(Electrode)
        an list of Electrode objects with only the ct coords indicated.
    '''
    print 'identifying electrode locations from CT image'

    from scipy import ndimage
    import sys

    cti = nib.load(ct)   
    ctd = cti.get_data()

    if mask is not None and type(mask)==str:
        maski = nib.load(mask)
        maskd = maski.get_data()
        maskd = np.around(maskd)    #eliminate noise in registration
        maskd = ndimage.binary_dilation(maskd, iterations=10)
    else:
        maskd = ctd.copy()
        maskd[:] = 1

    mask_test = ctd[np.where(maskd)]
    #print np.mean(mask_test), 'MASK MEAN'
    #print np.std(mask_test), 'MASK STDEV'
    #print np.mean(ctd), 'CT MEAN'
    #print np.std(ctd), 'CT STDEV'

    #threshold = np.mean(mask_test)+3*np.std(mask_test)
    #print threshold, 'COMPROMISE'
    print 'using threshold %.1f to isolate electrodes from CT' % threshold

    supthresh_locs = np.where(np.logical_and(ctd > threshold, maskd))

    ecs = np.zeros(cti.shape)
    ecs[supthresh_locs]=1

    if use_erosion:
        cte = ndimage.binary_erosion(ecs)
    else:
        cte = ecs

    ctpp = np.zeros(cti.shape)
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
                dfs(x,y,z,im,c) 
            
        return clusters

    recursionlimit = sys.getrecursionlimit()
    sys.setrecursionlimit(3000000)

    electrode_clusters = isolate_components(ctpp)

    sys.setrecursionlimit(recursionlimit)

    return [Electrode(ct_coords=cluster.center_of_mass()) 
            for cluster in electrode_clusters]

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

    mask_aff = maski.get_affine()

    orig_ras2vox = geo.get_vox2rasxfm(brain, stem='ras2vox')

    orig_vox2ras = geo.get_vox2rasxfm(brain, stem='vox2ras')

    np.set_printoptions(precision=2)

    #print mask_aff
    #print orig_ras2vox
    #print orig_vox2ras
    #print np.linalg.inv(mask_aff)

    for e in electrodes:
        #find nearest voxel in voxel space
        #voxel, = geo.apply_affine([e.asras()], np.linalg.inv(mask_aff))
        voxel, = geo.apply_affine([e.asras()], orig_ras2vox)
        #nx, ny, nz = map(int, (np.around(voxel)))
        nx, ny, nz = np.around(voxel).astype(np.int)

        try:

            if maskd[nx,ny,nz] == 0:
                removals.append(e)

        except:
            print "BAD %s"%str((nx,ny,nz))
            pass

    return removals

def classify_electrodes(electrodes, known_geometry,
    delta=.35, rho=35, rho_strict=20, rho_loose=50, color_scheme=None,
    epsilon=10, mindist=0, maxdist=36):
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
        def color_scheme():
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

    def name_generator():
        i=0
        while True:
            i+=1
            yield 'grid%i'%i

    colors = color_scheme()
    names = name_generator()

    electrode_arr = map((lambda x:getattr(x, 'ct_coords')), electrodes)

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
            raise ValueError("Could not find any good angles")

        for j,k in enumerate(ba):
            p0,p1,p2 = neighbs[k]
            pog = gl.Grid(p0,p1,p2,new_elecs, delta=delta,
                rho=rho, rho_strict=rho_strict, rho_loose=rho_loose,
                name=names.next())
            pog.extend_grid_arbitrarily()

            try:
                sp = pog.extract_strip(*dims)
            except gl.StripError as e:
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
                #import pdb
                #pdb.set_trace()
                if tuple(p.tolist()) in electrode_arr:
                    ix, = np.where(np.logical_and(np.logical_and( 
                        np.array(electrode_arr)[:,0]==p[0], 
                        np.array(electrode_arr)[:,1]==p[1]),
                        np.array(electrode_arr)[:,2]==p[2]))
                    try:
                        found_grids[pog.name].append(electrodes[ix])
                    except IndexError:
                        raise ValueError("multiple electrodes at same point")
                else:
                    found_grids[pog.name].append(Electrode(ct_coords=tuple(p),
                        is_interpolation=True))
                    
            break

    #return found_grids, grid_colors
    return grid_colors, grid_geom, found_grids, colors

def classify_single_fixed_grid(name, fixed_grids, known_geometry, colors,
    delta=.35, rho=35, rho_strict=20, rho_loose=50, 
    epsilon=10, max_cost=.4, mindist=0, maxdist=36):
    '''
    Sort the electrodes with the given name (in the space of the CT scan) into
    a grid matching the geometry of the grid with the given name.

    If the geometry is user-defined, prompt the user to provide the geometry
    before proceeding.

    Parameters
    ----------
    name : Str
        The name of the grid to fit
    fixed_grids : Dict( Str -> List(Electrode)
        A dictionary describing which electrodes correspond to each grid.
        Grids are represented by string name and electrode by Electrode
        objects. Only the electrodes in the grid at the named key will
        be used, and they will be forced.
    known_geometry : Dict( Str -> List(2x1) )
        A dictionary describing the geometry on each grid. Each grid is
        associated with the same name as in fixed_grids.
    colors : Dict( Str -> Color )
        A dictionary describing colors for each grid.
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
    max_cost : Float
        Does not currently do anything

    Returns
    -------
    returns True if the geometry reconstruction was successful, and False
        if it was unsuccessful
    '''
    cur_grid = fixed_grids[name]
    cur_geom = known_geometry[name]
    elecs = map((lambda x:getattr(x, 'ct_coords')), cur_grid)

    if len(cur_geom) != 2:
        raise ValueError("Specified geometry is not 2-dimensional")
    if (0 in cur_geom):
        raise ValueError("Geometry cannot be Nx0")

    angles, _, neighbs = gl.find_init_angles(np.array(elecs), 
        mindist=mindist, maxdist=maxdist)

    epsilon *= 2.5
    if (1 in cur_geom):
        ba = np.squeeze(sorted(zip(*np.where(np.abs(180-angles)<epsilon)),
                key=lambda v:np.abs(180-angles[v])))
    else:
        ba = np.squeeze(sorted(zip(*np.where(np.abs(90-angles)<epsilon)),
                key=lambda v:np.abs(90-angles[v])))

    if ba.shape==():
        ba=[ba]
    elif len(ba)==0:
        raise ValueError("Could not find any good angles with epsilon %i "
            "mindist %i maxdist %i"%(epsilon, mindist, maxdist))

    for j,k in enumerate(ba):
        p0,p1,p2 = neighbs[k]
        pog = gl.Grid(p0,p1,p2, np.array(elecs), delta=delta,
            rho=rho, rho_strict=rho_strict, rho_loose=rho_loose,
            is_line=(1 in cur_geom))
        pog.extend_grid_arbitrarily()

        try:
            sp = pog.extract_strip(*cur_geom)
            break
        except gl.StripError as e:
            if j==len(ba)-1:
                raise ValueError('No acceptable interpolation could be '
                    'reconstructed from the user specified points')
            continue

    sp = np.reshape(sp, (-1,3))
    #keep track of the electrodes newly added
    elec_dict = {}
    interpolates = []
    for elec in cur_grid:
        elec_dict[elec.ct_coords] = elec

    for point in pog.points:
        #try:
        #    elec_dict[tuple(point)]
        #except KeyError:
        if tuple(point) not in elec_dict.keys():
            new_elec = Electrode(ct_coords=tuple(point), grid_name=name)
            interpolates.append(new_elec)
            elec_dict[tuple(point)] = new_elec

    #this might be dangerous
    #fixed_grids[name] = elec_dict.values()

    return True, interpolates

#    def getgrid_continuation(geom, epsilon=epsilon):
#        if len(geom) != 2:
#            raise ValueError("Specified geometry is not 2-dimensional")
#        if (0 in geom):
#            raise ValueError("Geometry cannot be Nx0")
#
#        angles, _, neighbs = gl.find_init_angles(np.array(elecs), 
#            mindist=mindist, maxdist=maxdist)
#
#        if (1 in geom):
#            epsilon *= 2.5
#
#            ba = np.squeeze(sorted(zip(*np.where(np.abs(180-angles)<epsilon)),
#                    key=lambda v:np.abs(180-angles[v])))
#        else:
#            ba = np.squeeze(sorted(zip(*np.where(np.abs(90-angles)<epsilon)),
#                    key=lambda v:np.abs(90-angles[v])))
#        
#        if ba.shape==():
#            ba=[ba]
#        elif len(ba)==0:
#            raise ValueError("Could not find any good angles with epsilon %i "
#                "mindist %i maxdist %i"%(epsilon, mindist, maxdist))
#
#        newpog = None
#        for j,k in enumerate(ba):
#            p0,p1,p2 = neighbs[k]
#            pog = gl.Grid(p0, p1, p2, np.array(elecs), delta=delta, rho=rho,
#                rho_strict=rho_strict, rho_loose=rho_loose,
#                critical_percentage=1, is_line=(1 in geom))
#                
#            try:
#                pog.recreate_geometry( )
#            except gl.StripError as e:
#                print 'Could not recreate geometry with this initialization'
#                continue
#
#            #if all points were included in the grid already just return
#            #the Grid object for subsequent extraction of geometry
#            #if len(pog.points) == geom[0]*geom[1]:
#            match, _, _ = pog.matches_strip_geometry(*geom)
#            if match:
#                newpog = pog
#                break
#
#            #allow for a second step of interpolation if not all points are
#            #settled
#            try: 
#                # now we want to interpolate
#                pog.critical_percentage = .75
#                import pdb
#                pdb.set_trace()
#                pog.extend_grid_arbitrarily()
#                print pog
#                strip = pog.extract_strip(*geom)
#            except gl.StripError as e:
#                print 'Could not interpolate missing points, rejecting'
#                print pog
#                continue
#
#            try:
#                newpog = gl.Grid(p0, p1, p2, np.array(strip), delta=delta,
#                    rho=rho, rho_strict=rho_strict, rho_loose=rho_loose,
#                    critical_percentage=1)
#                newpog.recreate_geometry()
#            except gl.StripError as e:
#                print "Could not fit geometry with newly interpolated points"
#                continue
#
#            #if len(newpog.points) != geom[0]*geom[1]:
#            match, _, _ = newpog.matches_strip_geometry(*geom)
#            if not match:
#                print "Unknown error in reconstructing interpolated geometry"
#                continue
#            else:
#                break
#
#        #if the loop did not find anything, raise an error
#        if newpog is None:
#            raise ValueError("Could not create a grid matching the specified "
#                "geometry")
#
#        return newpog
#
#    if cur_geom=='user-defined':
#        from utils import GeomGetterWindow, GeometryNameHolder
#        from color_utils import mayavi2traits_color
#        nameholder = GeometryNameHolder(
#            geometry=cur_geom,
#            color=mayavi2traits_color(colors[name]))
#        geomgetterwindow = GeomGetterWindow(holder=nameholder)
#
#        if geomgetterwindow.edit_traits().result:
#            try:
#                pog = getgrid_continuation(geomgetterwindow.geometry)
#            except ValueError as e:
#                print 'Geometry reconstruction failed: specific error follows'
#                print e
#                return False, None
#        else:
#            print "User did not specify any geometry, ignoring geometry"
#            return False, None
#
#    else:
#        try:
#            pog = getgrid_continuation(cur_geom)
#        except ValueError as e:
#            print 'Geometry reconstruction failed: specific error follows'
#            print e
#            return False, None
#
#    print 'Finished reconstructing grid geometry'
#    print pog
#
#    #Add the local geometry position to the electrode object
#    elec_dict = {}
#    interpolates = []
#    for elec in cur_grid:
#        elec_dict[elec.ct_coords] = elec
#
#    xmin = 0
#    ymin = 0
#    for point in pog.points:
#        try:
#            x,y = pog.connectivity[gl.GridPoint(point)]
#            if x < xmin:
#                xmin=x
#            if y < ymin:
#                ymin=y
#        except:
#            pass
#
#    for point in pog.points:
#        try:
#            x,y = pog.connectivity[gl.GridPoint(point)]
#            elec_dict[tuple(point)].geom_coords = list((x-xmin, y-ymin))
#        except KeyError:
#            new_elec = Electrode(ct_coords=tuple(point),
#                                 geom_coords = list((x-xmin, y-ymin)))
#            interpolates.append(new_elec)
#            elec_dict[tuple(point)] = new_elec
#
#    #this might be dangerous
#    fixed_grids[name] = elec_dict.values()
#
#    return True, interpolates

def classify_with_fixed_points(fixed_grids, known_geometry, 
    delta=.35, rho=35, rho_strict=20, rho_loose=50, 
    epsilon=10, max_cost=.4):
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
    fixed_grids : Dict( Str -> List(Electrode)
        A dictionary describing which electrodes correspond to each grid.
        Grids are represented by string name and electrode by Electrode
        objects.
    known_geometry : Dict( Str -> List(2x1) )
        A dictionary describing the geometry on each grid. Each grid is
        associated with the same name as in fixed_grids.
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
    max_cost : Float
        A fitting parameter for classification with fixed points only.
        Represents the maximum value of the cost function for normal
        iteration. Im not sure what this was supposed to be anymore now
        that we are not using this function.
    '''
    electrode_arr = map((lambda x:getattr(x, 'ct_coords')), electrodes)
    all_elecs = np.array(electrode_arr)

    found_grids = {}

    for grid_name in fixed_grids:
        grid = fixed_grids[grid_name]
        elecs = map((lambda x:getattr(x, 'ct_coords')), grid)

        geom = known_geometry[grid_name]

        # if this grid is already fully populated, just return it
        if len(elecs) == geom[0]*geom[1]:
            found_grids[grid_name] = grid
            continue

        angles, _, neighbs = gl.find_init_angles(elecs, mindist=mindist,
            maxdist=maxdist)

        ba = np.squeeze(sorted(zip(*np.where(np.abs(90-angles)<epsilon)),
                key=lambda v:np.abs(90-angles[v])))

        if ba.shape==():
            ba=[ba]
        elif len(ba)==0:
            raise ValueError("Could not find any good angles")

        for j,k in enumerate(ba):
            p0,p1,p2 = neighbs[k]
            pog = gl.Grid(p0, p1, p2, elecs, delta=delta, rho=rho,
                rho_strict=rho_strict, rho_loose=rho_loose, name=grid_name)
                
            try:
                pog.recreate_geometry( )
            except gl.StripError as e:
                print 'Could not recreate geometry with this initialization'
                continue

            # now set the expanded electrode space to re-fit the other grid
            # points
            pog.all_elecs = all_elecs

            pog.extend_grid_arbitrarily()

            try:
                sp = pog.extract_strip(*geom)
            except gl.StripError as e:
                print 'Rejected this choice'
                if j==len(ba)-1:
                    raise ValueError("Could not incorporate fixed points")

            found_grids[grid_name] = []
            for p in sp:
                if tuple(p.tolist()) in electrode_arr:
                    ix, = np.where(np.logical_and(np.logical_and( 
                        np.array(electrode_arr)[:,0]==p[0], 
                        np.array(electrode_arr)[:,1]==p[1]),
                        np.array(electrode_arr)[:,2]==p[2]))
                    try:
                        found_grids[pog.name].append(electrodes[ix])
                    except IndexError:
                        raise ValueError("multiple electrodes at same point")
                else:
                    found_grids[pog.name].append(Electrode(ct_coords=tuple(p),
                        is_interpolation=True))

            break
 
    return found_grids

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
    cm_dist=5, overwrite=False, zf_override=0):
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
    print 'registering CT to MR with manual resampling'

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

    if zf_override == 0:
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

        zoom_factor = (x+n)/x

    else:
        zoom_factor = zf_override

    #resample the ct image and do the final registration

    from scipy.ndimage.interpolation import zoom
    print 'resampling image with zoom_factor %.2f'%zoom_factor
    ct_zoom = zoom( ctd, (1,1,zoom_factor))

    resampled_ct = os.path.join(ct_register_dir, 'ct_resampled_zf.nii.gz')

    resamp_img = nib.Nifti1Image(ct_zoom, affine=cti.get_affine(), 
        header=hdr)
    nib.save(resamp_img, resampled_ct)

    ct_final = os.path.join(ct_register_dir, 'ct_final_resamp_reg.nii.gz')

    mri_robustreg_resampled_cmd = ['mri_robust_register', '--mov', 
        resampled_ct, '--dst', rawavg, '--lta', skewed_lta, '--satit',
        '--vox2vox', '--mapmov', ct_final, '--cost', 'mi']
    s = subprocess.call(mri_robustreg_resampled_cmd)

    skew_aff = geo.get_lta(skewed_lta)
    unskew_aff = np.eye(4)
    unskew_aff[2,2] = zoom_factor
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

    if (os.path.exists(os.path.join(subjects_dir,subject,'surf','lh.dural'))
            and os.path.exists(os.path.join(subjects_dir, subject,'surf',
            'rh.dural'))):
        print 'dural surfaces already exist'
        return

    import subprocess

    curdir = os.getcwd()
    os.chdir(os.path.join(subjects_dir, subject, 'surf'))

    for hemi in ('lh','rh'):
        make_dural_surface_cmd = [os.path.join(curdir, 
            'make_dural_surface.csh'),'-i','%s.pial'%hemi]
        p=subprocess.call(make_dural_surface_cmd)

    os.chdir(curdir)

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

    There is no return value. The 'surf_coords' attribute will be used to
    store the surface locations of the electrodes
    '''
    print 'translating electrodes to surface space'

    if subjects_dir is None or subjects_dir=='':
        subjects_dir = os.environ['SUBJECTS_DIR']
    if subject is None or subject=='':
        subject = os.environ['SUBJECT']

    electrode_arr = map((lambda x:getattr(x, 'ct_coords')), electrodes)
    orig_elecs = geo.apply_affine(electrode_arr, ct2mr)

    rawavg = os.path.join(subjects_dir, subject, 'mri', 'rawavg.mgz')
    orig = os.path.join(subjects_dir, subject, 'mri', 'orig.mgz')
    lta = os.path.join(subjects_dir, subject, 'mri', 'transforms',
        'raw2orig.lta')

    if not os.path.exists(lta):
        import subprocess
        mri_robustreg_cmd = ['mri_robust_register','--mov',rawavg,'--dst',
            orig,'--lta',lta,'--satit','--vox2vox']
        p = subprocess.call(mri_robustreg_cmd)

    nas2ras = geo.get_lta(lta)

    nas_locs = geo.apply_affine(orig_elecs, nas2ras)

    tkr = geo.get_vox2rasxfm(orig, stem='vox2ras-tkr')

    surf_locs = geo.apply_affine(nas_locs, tkr)
    for elec, loc in zip(electrodes, surf_locs):
        elec.surf_coords = loc

def snap_electrodes_to_surface(electrodes, subjects_dir=None, 
    subject=None, max_steps=40000, giveup_steps=10000, 
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

    dura = np.vstack((lh_dura, rh_dura))

    max_deformation = 3
    deformation_choice = 50

    #adjust annealing parameters
    # H determines maximal number of steps
    H = max_steps
    #Texp determines the steepness of temperateure gradient
    Texp=1-1/H
    #T0 sets the initial temperature and scales the energy term
    T0 = 1e-3
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
    pia = np.vstack((lh_pia, rh_pia))

    e_pia = np.argmin(cdist(pia, emin), axis=0)

    for elec, soln in zip(electrodes, e_pia):
        elec.vertno = soln if soln<len(lh_pia) else soln-len(lh_pia)
        elec.hemi = 'lh' if soln<len(lh_pia) else 'rh'
        elec.pial_coords = pia[soln]

def fit_grid_to_line(electrodes, c1, c2, c3, geom=None, mindist=0, maxdist=36,
    epsilon=30, delta=.5, rho=35, rho_strict=20, rho_loose=50):
    '''
    Given a list of electrodes and two endpoints of a line, fit the electrodes
    onto the Nx1 or 1xN line using a greedy fitting procedure. Set the
    geom_coords attribute of the fitted electrodes.

    Parameters
    ----------
    electrodes : List(Electrodes)
        List of electrodes in the specified strip
    c1, c2, c3 : Tuple
        Tuple containing coordinates (in CT space) of the endpoint electrodes
        the user selected. It is assumed that the user selected endpoints
        such that the line looks like (c1,c2,c3,....).
    geom : 2-Tuple
        The known geometry of this grid, which must be either Nx1 or 1xN
        This is not used at all and can be None

    No return value
    '''
    c1 = np.array(c1)
    c2 = np.array(c2)
    c3 = np.array(c3)

    electrode_arr = map((lambda x:getattr(x, 'ct_coords')), electrodes)

    pog = gl.Grid(c2, c3, c1, np.array(electrode_arr), delta=delta,
        rho=rho, rho_strict=rho_strict, rho_loose=rho_loose, is_line=True)
    #import pdb
    #pdb.set_trace()
    pog.extend_grid_arbitrarily()

    if len(pog.points) < len(electrodes):
        raise ValueError('Failed to fit all the electrodes')

    #try:
    #    sp = pog.extract_strip(*geom)
    #except gl.StripError as e:
    #    raise ValueError('No acceptable interpolated line could be found')

    miny=-1
    for elec in electrodes:
        y = pog.connectivity[gl.GridPoint(elec.ct_coords)][1]
        if y<miny:
            miny=y

    for elec in electrodes:
        conn = pog.connectivity[gl.GridPoint(elec.ct_coords)]
        elec.geom_coords = [0, conn[1]-miny]

def fit_grid_to_plane(electrodes, c1, c2, c3, geom):
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

    plane_points = np.array(plane.keys())

    #assign the electrodes to the nearest plane point greedily
    pp = {}
    for elec in electrodes:
        e_greedy = np.argmin(cdist([elec.asct()], plane_points))
        pp[elec.ct_coords] = plane_points[e_greedy]
        plane_points = np.delete(plane_points, e_greedy, axis=0)

    pp_min = pp.copy()

    def globalcost(pp):
        c=0
        for e in pp:
            pe = pp[e]
            c += pdist((pe,e))**2
        return c

    # H determines maximal number of steps
    H = 20000
    #Texp determines the steepness of temperateure gradient
    Texp=1-1/H
    #T0 sets the initial temperature and scales the energy term
    T0 = 1e-5
    #Hbrk sets a break point for the annealing
    Hbrk = 20000
    
    h=0; hcnt=0
    lowcost = mincost = 1e16

    while h<H:
        h+=1; hcnt+=1

        T=T0*(Texp**h)

        e1 = np.random.randint(len(pp))
        cand_pt = np.argmin(cdist([electrodes[e1].asct()], pp.values()))

        if cand_pt==e1:
            continue
        eg = plane[tuple(pp[electrodes[e1].asct()])]
        cg = plane[tuple(pp[electrodes[cand_pt].asct()])]
        if np.abs(eg[0]-cg[0]) > 1 or np.abs(eg[1]-cg[1]) > 1:
            continue

        pp_tmp = pp.copy()
        old_e1 = pp_tmp[electrodes[e1].asct()]
        old_cp = pp_tmp[electrodes[cand_pt].asct()]

        pp_tmp[electrodes[cand_pt].asct()]=old_e1
        pp_tmp[electrodes[e1].asct()]=old_cp

        cost = globalcost(pp_tmp)
        if cost < lowcost or np.random.random()<np.exp(-(cost-lowcost)/T):
            pp = pp_tmp
            lowcost = cost
 
            if cost < mincost:
                pp_min = pp
                mincost = cost
                print 'step %i in plane fitting, cost %f' %(h, mincost)
                
    for elec in electrodes:
        #elec.plane_coords = plane[pp[elec.ct_coords]]
        elec.geom_coords = list(plane[tuple(pp[elec.ct_coords])])

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
