from __future__ import division
import os
import numpy as np
import nibabel as nib
import geometry as geo
import grid_localize as gl

def create_brainmask_in_ctspace(ct, subjects_dir=None, subject=None):
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

    Returns
    -------
    brainmask : str
        The location of the textfile where the brainmask is located,
        which is currently $SUBJECTS_DIR/mri/brain_ct.nii.gz
    '''
    if subjects_dir is None:
        subjects_dir = os.environ['SUBJECTS_DIR']
    if subject is None:
        subject = os.environ['SUBJECT']

    xfms_dir = os.path.join(subjects_dir, subject, 'mri', 'transforms')
    if not os.path.exists(xfms_dir):
        os.mkdir(xfms_dir)
    lta = os.path.join(xfms_dir,'mr2ct.lta')
    #_,lta = tempfile.mkstemp()
    
    orig = os.path.join(subjects_dir, subject, 'mri', 'orig.mgz')
    brain = os.path.join(subjects_dir, subject, 'mri', 'brain.mgz')
    ct_brain = os.path.join(subjects_dir, subject, 'mri', 'brain_ct.nii.gz')

    if os.path.exists(ct_brain):
        return ct_brain

    import tempfile
    import subprocess

    mri_robustreg_cmd1 = ['mri_robust_register','--mov',orig,'--dst',ct,
        '--lta',lta,'--satit','--vox2vox','--cost','nmi']
    p=subprocess.call(mri_robustreg_cmd1)

    _,gbg = tempfile.mkstemp()

    mri_robustreg_cmd2 = ['mri_robust_register','--mov',brain,'--dst',ct,
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
    electrodes : an Nx3 list of electrode locations in CT space.
    '''
    from scipy import ndimage
    import sys

    cti = nib.load(ct)   
    ctd = cti.get_data()

    if mask is not None and type(mask)==str:
        maski = nib.load(mask)
        maskd = mask.get_data()
        maskd = ndimage.binary_dilation(maskd, iterations=15)
    else:
        maskd = ctd.copy()
        maskd[:] = 1

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

    return [cluster.center_of_mass() for cluster in electrode_clusters]

def classify_electrodes(electrodes, known_geometry, fixed_points=None,
    delta=.35, rho=35, rho_strict=20, rho_loose=50, color_scheme=None,
    epsilon=10):
    '''
    Sort the given electrodes (generally in the space of the CT scan) into
    grids and strips matching the specified geometry.

    Parameters
    ----------
    electrodes : list of 3-tuples | Nx3 np.ndarray
        A list of electrode locations, possibly including noise.
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

    Returns
    -------
    found_grids : Dict(Grid : List)
        a dictionary of grid objects mapping to the sorted and interpolated 
        points for these grid objects
    color_scheme : None | Generator
        The color scheme to be used, described above
    '''
    if color_scheme is None:
        def color_scheme():
            predefined_colors = [(.9,.6,1),(.7,1,.6),(.2,.5,.8),(.6,.3,.9),
                (.9,.7,.1),(1,1,1),(.5,.9,.4),(1,.2,.5),(.7,.7,.9),(0,1,0),
                (0,1,0)]
            for color in predefined_colors:
                yield color
            while True:
                yield tuple(np.random.random(3))

    colors = color_scheme()

    found_grids = {}
    used_points = []

    for i,dims in enumerate(known_geometry):
        new_elecs = geo.rm_pts(np.reshape(used_points, (-1,3)), 
            np.array(electrodes))

        angles, dists, neighbs = gl.find_init_angles(new_elecs, mindist=0, 
            maxdist=28)

        ba = np.squeeze(sorted(zip(*np.where(np.abs(90-angles)<epsilon)),
                key=lambda v:np.abs(90-angles[v])))

        if ba.shape==():
            ba=[ba]
        elif len(ba)==0:
            print "Could not find any good angles"
            break

        for j,k in enumerate(ba):
            p0,p1,p2 = neighbs[k]
            pog = gl.Grid(p0,p1,p2,new_elecs, delta=delta,
                rho=rho, rho_strict=rho_strict, rho_loose=rho_loose)
            pog.extend_grid_arbitrarily()

            try:
                sp = pog.extract_strip(*dims)
            except gl.StripError as e:
                print 'Rejected this initialization'
                if j==len(ba)-1:
                    raise ValueError('No suitable strip found')
                continue

            sp = np.reshape(sp, (-1,3))
            found_grids[pog]=sp
            [used_points.append(p) for p in sp]
            break

    return found_grids, colors

def register_ct_to_mr_using_mutual_information(ct, subjects_dir=None,
    subject=None):
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

    Returns
    -------
    affine : 4x4 np.ndarray
        The matrix containing the affine transformation from CT to MR space.
    '''
    #import tempfile
    import subprocess

    #_,lta = tempfile.mkstemp()
    xfms_dir = os.path.join(subjects_dir, subject, 'mri', 'transforms')
    if not os.path.exists(xfms_dir):
        os.mkdir(xfms_dir)
    lta = os.path.join(xfms_dir,'ct2mr.lta')

    if os.path.exists(lta):
        return geo.get_lta(lta)

    if subjects_dir is None:
        subjects_dir = os.environ['SUBJECTS_DIR']
    if subject is None:
        subject = os.environ['SUBJECT']
    orig = os.path.join(subjects_dir, subject, 'mri', 'orig.mgz')

    mri_robustreg_cmd = ['mri_robust_register','--mov',ct,'--dst',orig,
        '--lta',lta,'--satit','--vox2vox','--cost','nmi']
    p=subprocess.call(mri_robustreg_cmd)

    affine=geo.get_lta(lta)

    #os.unlink(lta)
    return affine

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
    if subjects_dir is None:
        subjects_dir = os.environ['SUBJECTS_DIR']
    if subject is None:
        subject = os.environ['SUBJECT']

    if (os.path.exists(os.path.join(subjects_dir,subject,'surf','lh.dural'))
            and os.path.exists(os.path.join(subjects_dir, subject,'surf',
            'rh.dural'))):
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
    subjects_dir=None, subject=None):
    '''
    Translates electrodes from CT space to orig space, and then from
    orig space to surface space.

    Parameters
    ----------
    electrodes : List(3-tuple) | Nx3 ndarray
        List of electrodes in CT space
    ct2mr : 4x4 np.ndarray
        Matrix containing the ct2mr affine transformation
    subjects_dir : Str | None
        The freesurfer subjects_dir. If this is None, it is assumed to be the 
        $SUBJECTS_DIR environment variable.
    subject : Str | None
        The freesurfer subject. If this is None, it is assumed to be the
        $SUBJECT environment variable.
    '''
    if subjects_dir is None:
        subjects_dir = os.environ['SUBJECTS_DIR']
    if subject is None:
        subject = os.environ['SUBJECT']

    orig_elecs = geo.apply_affine(electrodes, ct2mr)

    orig = os.path.join(subjects_dir, subject, 'mri', 'orig.mgz')
    tkr = geo.get_vox2rasxfm(orig, stem='vox2ras-tkr')

    return geo.apply_affine(orig_elecs, tkr)

def snap_electrodes_to_surface(electrodes, hemi, subjects_dir=None, 
    subject=None, max_steps=40000, giveup_steps=10000):
    '''
    Transforms electrodes from surface space to positions on the surface
    using a simulated annealing "snapping" algorithm which minimizes an
    objective energy function as in Dykstra et al. 2012

    Parameters
    ----------
    electrodes : List(3-tuple) | Nx3 ndarray
        List of electrodes in surface space, but not necessarily on the
        surface itself (merely near the surface)
    hemi : Str
        "lh" or "rh", depending on the hemisphere to be snapped to. Maybe
        the user has to set this.
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

    Returns
    -------
    emin : Nx3 np.ndarray
        The electrode positions snapped onto the dural surface.
    '''
    if subjects_dir is None:
        subjects_dir = os.environ['SUBJECTS_DIR']
    if subject is None:
        subject = os.environ['SUBJECT']

    from scipy.spatial.distance import cdist

    n = len(electrodes)
    e_init = np.array(electrodes)

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
            H += float(cdist( [e_new[i]], [e_old[i]] ))

            for j in xrange(i):
                H += alpha[i,j] * (dist_new[i,j] - dist_old[i,j])**2

        return H

    #load the dural surface locations
    dura, _ = nib.freesurfer.read_geometry(
        os.path.join(subjects_dir, subject, 'surf', '%s.dural'%hemi))

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

    return emin

    
