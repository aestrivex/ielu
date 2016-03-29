from __future__ import division
import numpy as np
from numpy.linalg import norm
from geometry import (angle, is_parallel, is_perpend, within_distance, rm_pts,
    find_nearest_pt, find_neighbors, binarize)
from utils import SortingLabelingError

class GridPoint():
    #this is just a class to make 3D arrays hashable

    #def __init__(self, 3d_loc, 2d_loc):
        #self.2d_loc = 2d_loc
    def __init__(self, loc_3d):
        self.loc_3d = np.array(loc_3d)

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other):
        return np.all(self.loc_3d == other.loc_3d)

    def __str__(self):
        return str(self.loc_3d)

    def __repr__(self):
        return str(self)

class Grid():
    '''
    p0, p1, p2 : 3x 3-tuple
        with coordinates of 3 points forming a right angle
    all_elecs : List(3-tuple)
        list of possible electrode locations
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
        iteration.
    is_line : Bool
        If true, the starting points are colinear and do not form a right
        angle but a 180 degree angle (potentially within a higher epsilon
        tolerance). The connectivity can only be extended in one dimension
        under this setup. The default is false.
    crit_pct : Float
        The critical percentage of electrodes to find. The algorithm won't
        make a lot of sense if this is less than 50/60%
    '''

    def __init__(self,p0,p1,p2, all_elecs, delta=.35, rho=35,
            rho_strict=20, rho_loose=50, max_cost=.4, name='',
            critical_percentage=.75, is_line=False):

        self.name=name 

        #save the unfiltered set of all electrodes
        self.all_elecs = all_elecs

        #maintain a list of points (3D coordinates) that have been used so far
        #to easily calculate set of available points to draw from
        if is_line:
            self.points = [p0, p1]
        else:
            self.points = [p0, p1, p2]

        #maintain an unsorted list of distances that have been created so far
        #to easily calculate the average distance
        self.distances = [norm(p0-p1)]

        #not necessary for line, we can get better estimate but probably will
        #work in 99% of cases

        #maintain a dictionary mapping 3D locations to 2D locations on a grid.
        #this is used mostly as a sparse mapping of 2D grid points so that local connectivity
        #can be easily examined, 

        if is_line:
            self.connectivity = { GridPoint(p0) : (0,0),
                                  GridPoint(p1) : (0,1), 
                                  #GridPoint(p2) : (0,-1), 
                                }

            self.reverse_connectivity = {(0,0) : p0,
                                         (0,1) : p1,
                                         #(0,-1) : p2
                                        }

            self.is_line = True

        else:
            self.connectivity = { GridPoint(p0) : (0,0),
                                  GridPoint(p1) : (0,1), 
                                  GridPoint(p2) : (1,0), }

            self.reverse_connectivity = {(0,0) : p0,
                                         (0,1) : p1,
                                         (1,0) : p2}

            self.is_line = False

        self.marked = {}

        #define some constraint parameters 
        #self.delta = .35

        self.delta = delta

        self.rho = rho
        self.rho_strict = rho_strict
        self.rho_loose = rho_loose

        self.max_cost = max_cost

        #self.delta = .35
        #self.rho = 10
        #self.rho_strict = 10
        #self.rho_loose = 20

        self.critical_percentage = critical_percentage

    def __repr__(self):
        return 'Printing Grid with critdist %.2f ...\n%s' %(self.critdist(),
            repr(self.repr_as_2d_graph()) )

    def repr_as_2d_graph(self, pad_zeros=0):
        min_x = 0
        max_x = 1
        min_y = 0
        max_y = 1
        for x,y in self.connectivity.values():
            if x < min_x:
                min_x = x
            if x > max_x:
                max_x = x
            if y < min_y:
                min_y = y
            if y > max_y:
                max_y = y

        graph = np.zeros((max_x-min_x+1, max_y-min_y+1), dtype=int)
        for x,y in self.connectivity.values():
            graph[x-min_x, y-min_y]=1
        graph[-min_x,-min_y]=2
        graph[-min_x,-min_y+1]=3

        if self.is_line:
            graph[-min_x,-min_y-1]=4
        else:
            graph[-min_x+1,-min_y]=4

        if pad_zeros:
            new_graph = np.zeros((max_x-min_x+1+2*pad_zeros, 
                max_y-min_y+1+2*pad_zeros), dtype=int)
            new_graph[pad_zeros:-pad_zeros, pad_zeros:-pad_zeros] = graph
            graph = new_graph

        return graph

    def remaining_points(self):
        return rm_pts( self.points, self.all_elecs )

    def critdist(self):
        return np.mean(self.distances)

    def nearest(self, p0, allow_self=True):
        try:
            p,_ = find_nearest_pt(p0, self.remaining_points(), 
                allow_self=allow_self)
            return p
        except (IndexError, ValueError):
        #except IndexError:
            return np.array((np.inf, np.inf, np.inf))

    def add_point(self, pJ, coord_2d=None):
        if coord_2d is None:
            raise ValueError("Must specify a 2D coordinate")

        #update the set of all 3D coordinates for easy removal of coordinates
        self.points.append(pJ)

        #update the set of distances for easy calculation of the average distance
        x,y = coord_2d
        i=0
        for coord in ((x,y+1), (x+1,y), (x,y-1), (x-1,y)):
            p = self.get_3d_point(coord)
            if p is not None:
                self.distances.append( norm(pJ - p) )
                i+=1
        if i==0:
            raise ValueError("Internal error: No distances were added")

        #update the connectivity dictionary with the proper 2D coordinate
        p = GridPoint( pJ ) 
        if p in self.connectivity.keys():
            print self
            raise ValueError("Tried to reproduce an existing 2D grid point")
        self.connectivity[p] = coord_2d
        self.reverse_connectivity[coord_2d] = pJ

    def get_3d_point(self, coord_2d):
        try:
            return self.reverse_connectivity[coord_2d]
        except KeyError:
            return None
#        for p,c in zip(self.connectivity, self.connectivity.values()):
#            if np.all(np.array(c) == np.array(coord_2d) ):
#                return p.loc_3d
#        #if no matching point was found
#        return None

    def is_marked(self, coord_2d, connectivity):
        try:
            return self.marked[coord_2d] == connectivity
        except KeyError:
            return False

    def get_local_connectivity_3d(self, p):
        #takes a 3D point and checks the connectivity
        coord_2d = self.connectivity[GridPoint(p)]
        return self.get_local_connectivity_2d(coord_2d)

    def get_local_connectivity_2d(self, coord_2d):
        #takes a 2D point and checks the connectivity
        x,y = coord_2d

        east = self.get_3d_point( (x+1, y) ) is not None
        south = self.get_3d_point( (x, y-1) ) is not None
        west = self.get_3d_point( (x-1, y) ) is not None
        north = self.get_3d_point( (x, y+1) ) is not None

        nr_neighbors = east+south+west+north

        if nr_neighbors == 4:
            connectivity_profile = 'FULL'
            orientation = 'north'
        elif nr_neighbors == 3:
            connectivity_profile = 'TSHAPE'
            if not south:
                orientation = 'north'
            elif not east:
                orientation = 'west'
            elif not north:
                orientation = 'south'
            elif not west:
                orientation = 'east'
        elif nr_neighbors == 2 and ((east and west) or (north and south)):
            connectivity_profile = 'LINE'
            if north:
                orientation = 'north'
            elif east:
                orientation = 'east'
        elif nr_neighbors == 2:
            connectivity_profile = 'MOTIF'
            if north and west:
                orientation = 'north'
            elif west and south:
                orientation = 'west'
            elif south and east:
                orientation = 'south'
            elif east and north:
                orientation = 'east'
        elif nr_neighbors == 1:
            connectivity_profile = 'LEAF'
            if north:
                orientation = 'north'
            elif west:
                orientation = 'west'
            elif south:
                orientation = 'south'
            elif east:
                orientation = 'east'
        else:
            connectivity_profile = 'SINGLETON'
            orientation = 'north'
            #raise ValueError("Singleton point")

        return connectivity_profile, orientation

    def ccw_point(self, orientation, p_orient_3d, nr_rot=1 ):
        #takes the 3d coordinates of the orientation point and its orientation direction,
        #applies nr_rotations counterclockwise rotations
        #and returns the 2D coordinates of the result.
        
        p_orient_2d = self.connectivity[ GridPoint( p_orient_3d )]
        x,y = p_orient_2d

        if orientation == 'north':
            x_long_side = 0    
            direction = -1
        elif orientation == 'west':
            x_long_side = 1
            direction = 1
        elif orientation == 'south':
            x_long_side = 0
            direction = 1
        elif orientation == 'east':
            x_long_side = 1
            direction = -1

        if nr_rot == 1:
            if x_long_side:
                return x + direction, y - direction
            else:
                return x + direction, y + direction
        elif nr_rot == 2: 
            if x_long_side:
                return x + 2*direction, y
            else:
                return x, y + 2*direction
        elif nr_rot == 3:
            if x_long_side:
                return x + direction, y + direction
            else:
                return x - direction, y + direction
        
        else:
            raise ValueError("bad number of rotations")

    def ccw_orientation(self, orientation, nr_rot=1):
        #takes the orientation given and rotates it counterclockwise
        if orientation=='north':
            new_ort = 'west'
        elif orientation=='west':
            new_ort = 'south'
        elif orientation=='south':
            new_ort = 'east'
        elif orientation=='east':
            new_ort = 'north'
        else:
            raise ValueError("Bad orientation")
        if nr_rot==1:
            return new_ort
        elif nr_rot < 1 or nr_rot > 3:
            raise ValueError("nr_rot should be [1,3]")
        else:
            return self.ccw_orientation(new_ort, nr_rot=nr_rot-1)

    def fits_cross_motif(self, pJ, p0, p1, p2 ):
        '''
        compares the angle p1-p0-pJ. p0 is the point being extended and pJ is the point
        being considered.

        Returns true if the following conditions are true:
            the distance d (p0-pJ) is c*(1-delta) < d < c*(1+delta) where c is critdist()
            the angle p1-p0-pJ is within rho degrees of 90
            the angle p1-p0-pJ is within rho_strict degrees of the angle p1-p0-p2
        '''
        if GridPoint(pJ) in self.connectivity:
            return False

        c = self.critdist()
        distance_cond = within_distance(c, p0, pJ, self.delta)
        angle_cond = is_perpend(pJ-p0, p1-p0, self.rho_loose)
        #angle_cond = True
        rel_angle_cond = (np.abs( angle(pJ-p0, p1-p0) - angle(p1-p0, p2-p0) ) < self.rho)
        #rel_angle_cond = True
        return (distance_cond and angle_cond and rel_angle_cond)

    def fits_line(self, pJ, p0, p1): 
        '''
        compares the angle p1-p0-pJ.
        return true if
            the distance d (p0-pJ) is c*(1-delta) < d < c*(1+delta) where c is critdist()
            the angle p1-p0-pJ is within rho degrees of 180
        '''
        if GridPoint(pJ) in self.connectivity:
            return False

        #import pdb
        #pdb.set_trace()

        c = self.critdist()
        distance_cond = within_distance(c, p0, pJ, self.delta)
        angle_cond = is_parallel(pJ-p0, p1-p0, self.rho_loose)
        return (distance_cond and angle_cond)

    def fits_corner(self, pC, pOrig, p1, p2 ) :
        '''
        fits a point onto a corner in a motif.
        returns true if
            the distances d1 (pC - p1) and d2 (pC - p2) are within 1*delta of c 
            the angles pC-p2-pOrig and pC-p2-pOrig are within rho degrees of 90

        note that the actual p0 being evaluated is p1 or p2, and pOrig is a point
        next to p0
        '''
        if GridPoint(pC) in self.connectivity:
            return False
        if p1 is None or p2 is None:
            return False

        c = self.critdist()
        distance_cond_1 = within_distance(c, pC, p1, self.delta) 
        distance_cond_2 = within_distance(c, pC, p2, self.delta) 
        angle_cond_1 = is_perpend(pC-p1, pOrig-p1, self.rho)
        angle_cond_2 = is_perpend(pC-p2, pOrig-p2, self.rho)
        return (distance_cond_1 and distance_cond_2 and angle_cond_1 and angle_cond_2)

    def fits_parallel(self, pJ, p0, p1, pX, pZ):
        '''
        fits a point parallel to a corner
        returns true if
            the distance d (p0-pJ) is within 1*delta of c
            the angle pC-p0-p1 is within rho degrees of 90
            the line pC-p0 is parallel to pX-pZ within rho_strict degrees

        here, p0 is the actual p0 next to pJ unlike in the above method
        '''
        if GridPoint(pJ) in self.connectivity:
            return False
        if p1 is None or pX is None or pZ is None:
            return False

        c = self.critdist()
        distance_cond = within_distance(c, pJ, p0, self.delta)
        angle_cond = is_perpend(pJ-p0, p1-p0, self.rho)
        parallel_cond = is_parallel(pJ-p0, pZ-pX, self.rho_strict)
        return (distance_cond and angle_cond and parallel_cond)

    def extend_grid_arbitrarily(self):
        points = []
        while len(self.points) != len(points):
            for point in map(GridPoint, self.points):

                if point not in points:
                    points.append(point)

            self.extend_grid_systematically()
            print 'started with %i points, now has %i' % (len(points), len(self.points))

    def recreate_geometry(self):
        '''
        Given a reduced set of points, extend the grid only on that
        set of points. If line=True, all three starting points are colinear
        and there is no right angle to start with
        '''
        if len(self.points) != 3:
            raise ValueError("Should only recreate geometry on blank grid")

        #raise ValueError('Noet suppahted')

        #for now do the stupid thing of just building the grid on the
        #confirmed points.

        #ideally this will later include a cost function

        self.extend_grid_arbitrarily() 

    def extend_grid_systematically(self):
        '''
        loops through every point in the grid and tries to extend the grid in all directions
        in the plane
        '''
        for p0 in self.points:
            pts_added = False

            local_connectivity, orient = self.get_local_connectivity_3d(p0)

            if local_connectivity in ('FULL', 'SINGLETON'):
                continue

            x,y = self.connectivity[GridPoint(p0)]
        
            if self.is_marked((x,y), local_connectivity):
                continue

            p1 = self.get_3d_point( (
                x-int(orient=='west')+int(orient=='east'),
                y+int(orient=='north')-int(orient=='south') ))
                
            if local_connectivity == 'MOTIF':
                #check to extend the motif in both directions
                p2 = self.get_3d_point( self.ccw_point(orient, p1, nr_rot=1) )
                pJa = self.nearest( 2*p0-p2 )
                pJa_coord = self.ccw_point(orient, p1, nr_rot=3)
                if self.fits_cross_motif(pJa, p0, p1, p2):
                    self.add_point(pJa, pJa_coord)
                    pts_added = True

                pJb = self.nearest( 2*p0-p1 )
                pJb_coord = self.ccw_point(orient, p1, nr_rot=2)
                if self.fits_cross_motif(pJb, p0, p2, p1):
                    self.add_point(pJb, pJb_coord)
                    pts_added = True

            if local_connectivity == 'TSHAPE':
                # figure out which side of the T is not covered and extend to it using some combination of
                # the two available motif extensions and the line extension

                pA = self.get_3d_point( self.ccw_point(orient, p1, nr_rot=1) )
                pB = self.get_3d_point( self.ccw_point(orient, p1, nr_rot=3) )

                pJ = self.nearest( 2*p0 - p1 )
                pJ_coord = self.ccw_point(orient, p1, nr_rot=2)

                line_cond = self.fits_line( pJ, p0, p1 )
                left_motif_cond = self.fits_cross_motif( pJ, p0, pA, p1 )
                right_motif_cond = self.fits_cross_motif( pJ, p0, pB, p1 )

                if (line_cond + left_motif_cond + right_motif_cond >= 2):
                    self.add_point(pJ, pJ_coord)
                    pts_added = True

            if local_connectivity == 'LEAF':
                # do the line extension 
                pL = self.nearest( 2*p0-p1 )
                pL_coord = self.ccw_point(orient, p1, nr_rot=2)
                if self.fits_line( pL, p0, p1 ):
                    self.add_point(pL, pL_coord) 
                    pts_added = True

                # check for corner extension
                opp_orient = self.ccw_orientation(orient, nr_rot=2)
                pCa_coord = self.ccw_point(orient, p1, nr_rot=1)
                pCb_coord = self.ccw_point(orient, p1, nr_rot=3)

                pSa = self.get_3d_point( self.ccw_point( opp_orient, p0, 
                    nr_rot=3 ))
                if pSa is not None:
                    pCa = self.nearest( p0+pSa-p1 )
                    if self.fits_corner( pCa, p1, p0, pSa):
                        self.add_point(pCa, pCa_coord)
                        pts_added = True
                pSb = self.get_3d_point( self.ccw_point( opp_orient, p0, 
                    nr_rot=1 ))
                if pSb is not None:
                    pCb = self.nearest( p0+pSb-p1 )
                    if self.fits_corner( pCb, p1, p0, pSb):
                        self.add_point(pCb, pCb_coord) 
                        pts_added = True

                # check for parallel extension
                pX = self.get_3d_point( self.ccw_point( opp_orient, p0,  
                    nr_rot=2))
                pZa = self.get_3d_point( self.ccw_point( opp_orient, p1, 
                    nr_rot=3))
                if pZa is not None and pX is not None:
                    pIa = self.nearest( p0+pZa-pX )
                    if self.fits_parallel( pIa, p0, p1, pX, pZa):
                        self.add_point(pIa, pCa_coord) 
                        pts_added = True
                pZb = self.get_3d_point( self.ccw_point( opp_orient, p1, 
                    nr_rot=1))
                if pZb is not None and pX is not None:
                    pIb = self.nearest( p0+pZb-pX )
                    if self.fits_parallel( pIb, p0, p1, pX, pZb):
                        self.add_point(pIb, pCb_coord)
                        pts_added = True

            if local_connectivity == 'LINE':

                p2 = self.get_3d_point( self.ccw_point( orient, p1, nr_rot=2))
                pCa_coord = self.ccw_point(orient, p1, nr_rot=1)
                pCb_coord = self.ccw_point(orient, p1, nr_rot=3)
                opp_orient = self.ccw_orientation(orient, nr_rot=2)

                pSa = self.get_3d_point( self.ccw_point( opp_orient, p0, 
                    nr_rot=3 ))
                pSd = self.get_3d_point( self.ccw_point( orient, p0, 
                    nr_rot=1 ))

                if pSa is not None or pSd is not None: 
                    pCa = (self.nearest(p0+pSa-p1) if pSa is not None else 
                        self.nearest(p0+pSd-p2))
                    corner_1 = self.fits_corner(pCa, p1, pSa, p0)
                    corner_2 = self.fits_corner(pCa, p2, pSd, p0)
                    if corner_1 or corner_2:
                        self.add_point(pCa, pCa_coord)
                        pts_added = True

                pSb = self.get_3d_point( self.ccw_point( opp_orient, p0, 
                    nr_rot=1 ))
                pSc = self.get_3d_point( self.ccw_point( orient, p0, 
                    nr_rot=3 ))

                if pSb is not None or pSc is not None:
                    pCb = (self.nearest(p0+pSb-p1) if pSb is not None else
                        self.nearest(p0+pSc-p2))
                    corner_1 = self.fits_corner(pCb, p1, pSb, p0)
                    corner_2 = self.fits_corner(pCb, p2, pSc, p0)
                    if corner_1 or corner_2:
                        self.add_point(pCb, pCb_coord)
                        pts_added = True

                # check for parallel extension
                pX = self.get_3d_point( self.ccw_point( opp_orient, p0, 
                    nr_rot=2))
                pY = self.get_3d_point( self.ccw_point( orient, p0, nr_rot=2))

                pZa = self.get_3d_point( self.ccw_point( opp_orient, p1, 
                    nr_rot=3))
                pZd = self.get_3d_point( self.ccw_point( orient, p2, 
                    nr_rot=1 ))

                if ((pZa is not None and pX is not None) or 
                        (pZd is not None and pY is not None)):
                    if pX is not None and pZa is not None:
                        pIa = self.nearest( p0+pZa-pX )
                    elif pY is not None and pZd is not None:
                        pIa = self.nearest( p0+pZd-pY )
                    parallel_1 = self.fits_parallel(pIa, p0, p1, pX, pZa)
                    parallel_2 = self.fits_parallel(pIa, p0, p2, pY, pZd)
                    if parallel_1 or parallel_2:
                        self.add_point(pIa, pCa_coord) 
                        pts_added = True
                pZb = self.get_3d_point( self.ccw_point( opp_orient, p1, 
                    nr_rot=1))
                pZc = self.get_3d_point( self.ccw_point( orient, p2, 
                    nr_rot=3 ))
                if ((pZb is not None and pX is not None) or 
                        (pZc is not None and pY is not None)):
                    if pX is not None and pZb is not None:
                        pIb = self.nearest( p0+pZb-pX )
                    elif pY is not None and pZc is not None:
                        pIb = self.nearest( p0+pZc-pY )
                    parallel_1 = self.fits_parallel(pIb, p0, p1, pX, pZb)
                    parallel_2 = self.fits_parallel(pIb, p0, p2, pY, pZc)
                    if parallel_1 or parallel_2:
                        self.add_point(pIb, pCb_coord)
                        pts_added = True

            if not pts_added:
                self.marked[(x,y)] = local_connectivity

    def extract_strip(self, N, M):
        '''
        iterate through the grid and extract the best fit location for a strip 
        with dimensions NxM
        the best fit location is the one that has the most nodes matching the  
        expected geometric configuration

        interpolate any missing nodes into the geometry

        if multiple best fits exist, check the best one based on the best fit 
        plausibility (by distance only)
        of *all* grid nodes that were not selected in this strip. this biases 
        strip points to be in areas of
        possible overlap with other strips, rather than in areas with no 
        electrodes at all.

        if multiple best fits still exist (the distances were very high to any 
        electrodes, pick one arbitrarily)

        returns the set of points in the strip, including possible 
        interpolated points that are not in the
        image at all
        '''
        print 'Extracting an %i by %i strip' % (M,N)

        #from PyQt4.QtCore import pyqtRemoveInputHook
        #pyqtRemoveInputHook()
        #import pdb
        #pdb.set_trace()

        fit_ok, best_locs, best_fit = self.matches_strip_geometry(M,N)

        if not fit_ok:
            raise SortingLabelingError("No strip had a sufficiently good fit, "
                " best fit was %i"%int(best_fit))

        best_loc, points = self.disambiguate_best_fit_strips(best_locs, M, N)

        corners = self.determine_corners( best_loc, M, N, best_locs )

        final_connectivity = self.finalize_connectivity( best_loc, M, N )

        print ('Decided that the %i by %i strip at %s is the best fit' % 
            (M,N,best_loc))

        return points, corners, final_connectivity

    def disambiguate_best_fit_strips(self, potential_strip_locs, M, N):
        '''
        given a list of equally full potential strip locations, assign an 
        objective plausibility score to
        each strip based on the density of electrodes near missing points in 
        the strip.

        to do this, we interpolate the missing points based on the available  
        geometry.

        then we calculate the
        distances of these points to the nearest unrelated point (to account 
        for image artifacts that
        may cause us to lose the electrode to a nearby cluster). each 
        individual distance is capped at
        2*delta to limit the effect of outliers. 

        we return the sum of each of these distances in the grid as an 
        objective penalty function.
        the configuration that returns the lowest penalty function is returned 
        along with a list of its
        points in 3D space, including interpolated points

        if the strip locations all have the same penalty value, one of them is 
        returned arbitrarily
        (but not pseudorandomly)
        '''
        if len(potential_strip_locs) == 1:
            print ('Only one strip location possible, possibly the result of '
                'user intervention, returning it')
        else:
            print '%i potential %ix%i strip locations to check' % (
                len(potential_strip_locs), M, N)
            print potential_strip_locs

        graph = self.repr_as_2d_graph(pad_zeros = max(M,N))

        best_penalty = np.inf
        best_points = []
        #best_loc = None
        best_loc = potential_strip_locs[0]

        origin = v,w = zip(*np.where(graph==2))[0]

        #set the critical distance before we start adding points
        critdist = self.critdist()

        for r,c,orient in potential_strip_locs:

            cur_penalty = 0
            cur_points = []
            interpolated_points = []
            interpolated_gridpoints = []
            #total_points = 0

            strip_graph = (graph[r:r+N,c:c+M] if orient=='horiz' else 
                graph[c:c+M, r:r+N])

            for x,y in zip(*np.where(strip_graph)):
                #print "Added the existing point (%i,%i)"%(x+r-v,y+c-w)
                #print ("Added the existing point "
                #    "(%i,%i) which is %s"%(x+c-v,y+r-w,
                #    None if self.get_3d_point((x+c-v, y+r-w)) is None 
                #    else 'not None')
                cur_points.append(self.get_3d_point( (x+r-v, y+c-w) if 
                    orient=='horiz' else (x+c-v, y+r-w) ))
                #total_points += 1

            print 'starting disambiguation with %i points'%(len(cur_points))

            iter = 0
            while len(interpolated_points) < M*N - len(cur_points):
            #for iter in xrange(M*N):
                iter += 1
                if iter > M*N:
                    raise ValueError("Infinite loop")
                for x,y in zip(*np.where(strip_graph==0)):

                    i = x+r-v if orient=='horiz' else x+c-v
                    j = y+c-w if orient=='horiz' else y+r-w

                    if (i,j) in interpolated_gridpoints:
                        continue

                    connectivity, orientation = (self.
                        get_local_connectivity_2d( (i,j) ))

                    pN = self.get_3d_point( (i, j+1) ) 
                    pE = self.get_3d_point( (i+1, j) )
                    pS = self.get_3d_point( (i, j-1) )
                    pW = self.get_3d_point( (i-1, j) )

                    pNN = self.get_3d_point( (i, j+2) )
                    pEE = self.get_3d_point( (i+2, j) )
                    pSS = self.get_3d_point( (i, j-2) )
                    pWW = self.get_3d_point( (i-2, j) )

                    if connectivity == 'FULL':
                        pInterp = (pN+pE+pS+pW)/4

                    elif connectivity=='TSHAPE':
                        if orientation in ('north','south'):
                            pInterp = (pE+pW)/2
                        else:
                            pInterp = (pN+pS)/2

                    elif connectivity=='LINE':
                        if orientation in ('north','south'):
                            pInterp = (pN+pS)/2
                        else:
                            pInterp = (pE+pW)/2

                    elif connectivity in ('MOTIF', 'LEAF'):
                        if pNN is not None and (orientation=='north' 
                                or (connectivity=='MOTIF' and
                                orientation=='east')):
                            pInterp = 2*pN-pNN
                        elif pWW is not None and (orientation=='west' 
                                or (connectivity=='MOTIF' and 
                                orientation=='north')):
                            pInterp = 2*pW-pWW
                        elif pSS is not None and (orientation=='south' 
                                or (connectivity=='MOTIF' and 
                                orientation=='west')):
                            pInterp = 2*pS-pSS
                        elif pEE is not None and (orientation=='east' 
                                or (connectivity=='MOTIF' and 
                                orientation=='south')):
                            pInterp = 2*pE-pEE
                        else:
                            continue

                    # if we found a singleton it means not enough of the other 
                    # points have been interpolated yet
                    # we pass and wait
                    elif connectivity == 'SINGLETON':
                        pInterp = None
                        continue
                    
                    if pInterp is None:
                        raise ValueError('Could not interpolate point with ' 
                            'current methods')
                    elif self.get_3d_point((i,j)) is not None:
                        # in case the nonexistent point was added in a 
                        # previous iteration dont add it again
                        #if GridPoint(pInterp) in interpolated_gridpoints:
                        #    continue
                        # greedily growing the grid here might cause bias. but 
                        # check for bugs adding the same point multiple times
                        if (i,j) in interpolated_gridpoints:
                            raise ValueError("Internal error: should never be" 
                                "adding a point that already exists")
                        
                        #otherwise, we added this point on another strip 
                        #choice. We should add it to the interpolated
                        #points  as normally, but not add the point to the Grid

                    else:
                        print 'adding the point (%i,%i), %s'%(i,j,str(pInterp))
                        self.add_point(pInterp, (i,j)) 
                        #total_points += 1

                    interpolated_points.append(pInterp) 
                    #interpolated_gridpoints.append(GridPoint(pInterp))
                    interpolated_gridpoints.append((i,j))

                    points_left = rm_pts(cur_points, self.all_elecs)
                    
                    if len(points_left) > 0:
                        pPenalty, _ = find_nearest_pt(pInterp, 
                            rm_pts(cur_points, self.all_elecs))
                        cur_penalty += np.min((norm(pPenalty-pInterp), 
                            2*self.delta*critdist))
                    else:
                        cur_penalty = np.inf

            #update the winner
            if cur_penalty < best_penalty:
                best_penalty = cur_penalty
                best_points = cur_points
                best_points.extend(interpolated_points)
                best_loc = (r,c,orient)
            
        return best_loc, best_points 

    def matches_strip_geometry(self, M, N):
        graph = self.repr_as_2d_graph(pad_zeros = max(M,N))

        best_locs = []
        best_fit = -1

        #If the orientation is 'horiz', then the row dimension corresponds to N.
        #if is 'vert', the row dimension corresponds to M
        for orient in ('horiz', 'vert'):
            for r in xrange(graph.shape[int(orient=='vert')]-N+1):
                for c in xrange(graph.shape[int(orient=='horiz')]-M+1):
                    cur_loc = (r, c, orient)
                    subgraph = (graph[r:r+N, c:c+M] if orient=='horiz' 
                        else graph[c:c+M, r:r+N])

                    #if the control points are not present, reject this choice 
                    #of strip immediately
                    #this causes problems
                    #if (2 not in subgraph or 3 not in subgraph or 
                    #        4 not in subgraph):
                    #    continue

                    #calculate the binary fit of connectivity in this choice
                    cur_fit = np.sum(binarize(subgraph))

                    if cur_fit > best_fit:
                        best_fit = cur_fit
                        best_locs = [cur_loc]
                    elif cur_fit == best_fit:
                        best_locs.append(cur_loc)

        if best_fit < M*N*self.critical_percentage:
            return False, None, best_fit
        return True, best_locs, best_fit

    def determine_corners(self, best_loc, M, N, useless):
        #print best_loc
        #print M,N
        #print self

        #from PyQt4.QtCore import pyqtRemoveInputHook
        #import pdb
        #pyqtRemoveInputHook()
        #pdb.set_trace()

        graph = self.repr_as_2d_graph(pad_zeros = max(M,N))

        r, c, orient = best_loc

        origin = v,w = zip(*np.where(graph==2))[0]

        #corner 1, x=0 y=0
        c1 = self.get_3d_point( (r-v, c-w) if orient=='horiz' else
            (c-v, r-w) )
        
        #corner 2, HORIZ: x=N-1, y=0 VERT: x=0, y=N-1
        c2 = self.get_3d_point( (N-1+r-v, c-w) if orient=='horiz' else
            (c-v, N-1+r-w) )

        #corner 3, HORIZ: x=0, y=M-1 VERT: x=M-1, y=0
        c3 = self.get_3d_point( (r-v, M-1+c-w) if orient=='horiz' else
            (M-1+c-v, r-w) )

        #corner 4, HORIZ: x=N-1, y=M-1 VERT: x=M-1, y=N-1
        c4 = self.get_3d_point( (N-1+r-v, M-1+c-w) if orient=='horiz' else
            (M-1+c-v, N-1+r-w) )
        
        return (c1, c2, c3, c4)

    def finalize_connectivity(self, best_loc, M, N):
        graph = self.repr_as_2d_graph(pad_zeros = max(M,N))

        final_connectivity = {}

        r, c, orient = best_loc

        origin = v,w = zip(*np.where(graph==2))[0]

        if orient=='horiz': 
            for xn in xrange(N):
                for ym in xrange(M):

                    p = self.get_3d_point( (xn+r-v, ym+c-w) )
                    final_connectivity[tuple(p)] = (xn, ym)

        else:
            for xm in xrange(M):
                for yn in xrange(N):
                
                    p = self.get_3d_point( (xm+c-v, yn+r-w) )
                    final_connectivity[tuple(p)] = (xm, yn)

        return final_connectivity

##########################
# initialization point API
##########################

def find_init_angles(all_elecs, mindist=10, maxdist=25):
    ''' Takes the set of all electrodes and some constraint parameters.
        Returns angle for each electrode's best match as Nx1 vector'''
    n = all_elecs.shape[0]
    angles = np.zeros(n)
    dists = np.zeros((n,2))
    actual_points = np.zeros((n,3,3))

    for k in xrange(n):
        p0 = all_elecs[k,:]
        p1, p2 = find_neighbors(p0, all_elecs, 2)

        if ((mindist < norm(p1-p0) < maxdist) and (mindist 
                < norm(p2-p0) < maxdist)):
            angles[k] = angle(p1-p0, p2-p0)
            dists[k] = norm(p1-p0), norm(p2-p0)
            actual_points[k] = (p0,p1,p2)
        else:
            angles[k] = np.inf
            dists[k] = (np.inf, np.inf)
            actual_points[k] = (p0,p1,p2)

    return angles, dists, actual_points

def find_init_pts(init_coords, dist=25, min_angle=10):
    n_p = init_coords.shape[0]    
    As = np.zeros(n_p)
    for k in range(n_p):
        p0 = init_coords[k, :]
        p1, p2 = find_neighbors(p0, init_coords, 2)
        
        if (norm(p1 - p0) < dist) and (norm(p2 - p0) < dist):
            As[k] = angle(p1 - p0, p2 - p0)
        elif (sum(p1 == p0) == 3) or (sum(p2 == p0) == 3):
            As[k] = 400             
        else:
            As[k] = 500

    As[np.where(np.isnan(As))]=np.inf
    
    if np.abs(As - 90).min() < min_angle:
        p0 = init_coords[np.abs(As - 90).argmin(), :]
        p1, p2 = find_neighbors(p0, init_coords, 2)
        return [p0, p1, p2]
    else:
        return -1
