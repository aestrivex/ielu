from __future__ import division
import numpy as np
from numpy.linalg import norm

###########################
# simple geometry functions
###########################

def angle(v1, v2):
    x = np.dot(v1, v2)/(norm(v1)*norm(v2))

    #handle numeric floating point errors
    if x > 1:
        x = 1
    if x < -1:
        x = -1

    theta = 180*np.arccos(x)/np.pi   
    
    return theta
    
def is_perpend(v1, v2, eps=5):
    return np.abs(90-angle(v1,v2)) < eps

def is_parallel(v1, v2, eps=5):
    theta = angle(v1, v2)
    return (theta < eps) or (np.abs(theta - 180) < eps)

def within_distance(c, p1, p2, delta=5):
    return (c*(1-delta) < norm(p1-p2) < c*(1+delta))

def plane_normal(p0, p1, p2):
    v1 = (p1 - p0)/norm(p1 - p0)
    v2 = (p2 - p0)/norm(p2 - p0)
    n = np.cross(v1, v2)
    
    return n
        
def d_to_line(p0, v, p1):
    v1 = p1 - p0
    theta = np.arccos((np.dot(v1, v))/(norm(v1)*norm(v)))
    d = norm(v1)*np.sin(theta)
    return d

def find_plane_from_corners(p0,p1,p2):
    '''
    Given 3 points p0, p1, and p2, find the plane ax+by+cz+d=0
    Returns a,b,c,d
    '''
    v1 = np.array(p0) - np.array(p1)    
    v2 = np.array(p0) - np.array(p2)
    a,b,c = np.cross(v1,v2)
    x0,y0,z0 = p0
    d = - a*x0 - b*y0 - c*z0

    from fractions import gcd
    #g = reduce(gcd, (a,b,c,d)
    #return a//g, b//g, c//g, d//g

    return map(lambda x:x//reduce(gcd, (a,b,c,d)), (a,b,c,d))

def find_best_fit_plane(points):
    '''
    Given a (Nx3) set of points, find the best fit plane ax+by+cz+d=0.
    Returns a,b,c,d
    '''
    points = np.array(points)
    centroid = x0,y0,z0 = np.mean(points, axis=0)
    
    u,_,_ = np.linalg.svd(np.transpose(points))
    norm = a,b,c = u[:,-1]
    d = - a*x0 - b*y0 - c*z0
    
    return a,b,c,d

#############################
# compound geometry functions
#############################

def find_nearest_pt(p0, coords, allow_self=False):
    #if coords is 0 find an implausibly distant point
    #if np.size(coords) == 0:
    #    return np.array((np.inf, np.inf, np.inf)), -1
            #this might cause problems but we will say that any call
            #should check the return value of the point index -- mostly
            #i dont think i care about the index anyway

    d = np.apply_along_axis(np.subtract, 1, coords, p0)
    d = np.sum(d**2, axis=1)

    if d.min() == 0 and not allow_self:
        if not allow_self:
            d[np.argmin(d)] = d[np.argmax(d)]

    which_p = np.argmin(d)
    p1 = coords[which_p, :]
    
    return p1, which_p

def find_neighbors(p0, coords, n):
    n_p = coords.shape[0]
    if n > n_p - 1:
        raise ValueError('number of neighbors exceeds the total number of points')
    else:
        ind = range(n_p)
        ps = []
        for _ in range(n):
            p, which_p = find_nearest_pt(p0, coords[ind, :])
            ps.append(p)
            ind.remove(ind[which_p])
        return ps

def rm_pts(P, coords):
    ''' 
    This function does not mutate its arguments. It returns a numpy view
    of the set of points coords which does not contain the set of points P
    '''
    ind = []
    for v in P:
        x, = np.where(np.sum(np.apply_along_axis(np.equal, 1, coords, v), axis=1) == 3)
        if len(x) > 0:
            ind.append(x[0])
    ind = np.setdiff1d(range(coords.shape[0]), ind)
    
    return coords[ind, :]

#########
# utility
#########

def binarize(W):
    W=W.copy()
    W[W!=0]=1
    return W

##################################################
# functions to load and apply linear registrations
##################################################

def get_lta(lta):
    affine = np.zeros((4,4))
    with open(lta) as fd:
        for i,ln in enumerate(fd):
            #print i,ln
            if i < 8:
                continue
            elif i > 11:
                break
            #else
            #print ln
            affine[i-8,:] = np.array(map(float, ln.strip().split()))
    return affine

def get_vox2rasxfm(volume, stem='vox2ras'):
    import subprocess

    ps = subprocess.Popen('mri_info --%s %s'%(stem, volume), 
        shell=True, stdout=subprocess.PIPE)

    vox2ras = np.zeros((4,4))

    #with open(subprocess.PIPE) as fd:
    i = 0
    for ln in ps.stdout:
        try:
            loc = np.array(map(float, ln.strip('()[]\n').split()))
        except:
            continue

        vox2ras[i,:] = loc
        i+=1

    return vox2ras

def get_xfm(xfm_file):
    xfm = np.zeros((4,4))

    with open(xfm_file) as fd: 
    #with open('talairach.lta') as fd:
        i=0
        for ln in fd:
            try:
                loc = np.array(map(float, ln.strip('()[]\n;').split()))
            except:
                continue
            if len(loc) != 4:
                continue
            if i>=4:
                break

            #print i, ln
            #print loc
            xfm[i,:]=loc
            i+=1

    xfm[-1,:] = (0,0,0,1)
    #print ras2tal

    return xfm

def apply_affine(locs, affine):
    new_locs = []
    for loc in locs:
        new_loc = np.dot( affine, (loc[0], loc[1], loc[2], 1))
        #new_locs.append( [round(new_loc[i]) for i in xrange(3)] )
        new_locs.append( [np.around(new_loc[i], decimals=4) 
            for i in xrange(3)] )
    return new_locs

def concat_affines(aff1, aff2):
    return np.dot( aff2, aff1 )

def save_affine(fname, affine):
    np.savetxt(fname, affine)

def load_affine(fname):
    try:
        return np.loadtxt(fname)
    except:
        pass

    try:
        return np.load(fname)
    except:
        pass

    try:
        return get_lta(fname)
    except:
        pass

    try:
        return get_xfm(fname)
    except:
        pass

    raise ValueError('Unrecognized affine transformation format')
