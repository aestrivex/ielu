#the intention of this file is to store operations on groups of electrodes here
#and call them from whatever place is appropriate

#i intend to move working code to this file on an as-tweaked basis

from utils import ask_user_for_savefile
from plotting_utils import coronal_slice

###############
# coronal_slice
###############

def coronal_slice_grid( electrodes, title=None, 
                        subjects_dir=None, subject=None, 
                        dpi=150, size=(200, 200), savefile=None):
    '''
    Helper function which calls the coronal_slice plotting function. Handles
    the case where the user specifies the geometry as any combination of 1 or
    2 dimensional matrices with only 1 meaningful vector.
    '''
    elecs_have_geom = True
    for elec in electrodes:
        if len(elec.geom_coords) != 2:
            elecs_have_geom = False
            break

    # we assume that we are only looking at depth leads with appropriate
    # 1xN geometry
    if elecs_have_geom:
        start = reduce( 
            lambda x,y:x if x.geom_coords[1]<y.geom_coords[1] else y,
            electrodes)
        end = reduce(
            lambda x,y:x if x.geom_coords[1]>y.geom_coords[1] else y,
            electrodes)
    else:
        start, end = (None, None)

    return coronal_slice( electrodes, start=start, end=end, outfile=savefile,
        subjects_dir=subjects_dir, subject=subject, title=title,
        dpi=dpi, size=size )

def coronal_slice_all( grids, grid_types, subjects_dir=None, subject=None,
                       dpi=150, size=(200, 200) ):
    '''
    takes a dictionary with grid assignments and creates a pdf with
    the coronal slices for these grids
    '''
    pdf_file = ask_user_for_savefile('save pdf file with coronal slice images')

    if pdf_file is None:
        return

    from matplotlib.backends.backend_pdf import PdfPages

    with PdfPages(pdf_file) as pdf:
        for grid in grids:

            if grid_types[grid] != 'depth':
                continue

            electrodes = grids[grid] 
            if len(electrodes) == 0:
                continue

            fig = coronal_slice_grid( electrodes, title=grid,
                subjects_dir=subjects_dir, subject=subject,
                dpi=dpi, size=size )

            pdf.savefig(fig)

#################
# get_nearby_rois
#################
def _find_surrounding_rois( elec, parcellation='aparc', error_radius=4,
                            subjects_dir=None, subject=None ):
    if elec.pial_coords is not None:
        pos = elec.pial_coords
    else:
        pos = elec.surf_coords

    import pipeline as pipe
    #TODO incorporate subcortical structures into non-aparc
    roi_hits = pipe.identify_roi_from_atlas( pos,
        atlas = parcellation,
        approx = error_radius,
        subjects_dir = subjects_dir,
        subject = subject)

    if len(roi_hits) == 0:
        roi_hits = ['None']

    elec.roi_list = roi_hits

def get_nearby_rois_elec( *args, **kwargs ):
    return _find_surrounding_rois(*args, **kwargs)

def get_nearby_rois_grid( electrodes, parcellation='aparc',
                          error_radius=4,
                          subjects_dir=None, subject=None ): 
    for elec in electrodes:
        try:
            _find_surrounding_rois( elec, 
                                    parcellation=parcellation,
                                    error_radius=error_radius,
                                    subjects_dir=subjects_dir,
                                    subject=subject )
        except:
            print 'Failed to find ROIs for %s' % str(elec)

def get_nearby_rois_all( grids, subjects_dir=None, subject=None,
                         parcellation='aparc', error_radius=4 ):
    for grid in grids:    
        electrodes = grids[grid]
        get_nearby_rois_grid( electrodes,
                              parcellation=parcellation,
                              error_radius=error_radius,
                              subjects_dir=subjects_dir, subject=subject )

##############
# save_montage
##############

def _save_montage_file( savefile, electrodes, grid_types, 
                        snapping_completed=False ):
    # write the montage file
    with open( savefile, 'w' ) as fd:
        for j, elec in enumerate(electrodes):
            key = elec.grid_name
            if elec.name != '':
                label_name = elec.name
            else:
                elec_id = elec.geom_coords
                elec_2dcoord = ('unsorted%i'%j if len(elec_id)==0 else
                    str(elec_id))
                label_name = '%s_elec_%s'%(key, elec_2dcoord)

            if (snapping_completed and grid_types[key]=='subdural'):
                pos = elec.pial_coords.tolist()
            else:
                pos = tuple(elec.surf_coords)

            x,y,z = ['%.4f'%i for i in pos]

            line = '%s\t%s\t%s\t%s\n' % (label_name, x, y, z)

            fd.write(line)

def _save_csv_file( savefile, electrodes, grid_types,
                    snapping_completed=False ):
    #write the csv file
    import csv
    with open( savefile, 'w' ) as fd:
        writer = csv.writer(fd)

        for j,elec in enumerate(electrodes):
            key = elec.grid_name
            if elec.name != '':
                label_name = elec.name
            else:
                elec_id = elec.geom_coords
                elec_2dcoord = ('unsorted%i'%j if len(elec_id)==0 else
                    str(elec_id))
                label_name = '%s_elec_%s'%(key, elec_2dcoord)

            if (snapping_completed and grid_types[key]=='subdural'):
                pos = elec.pial_coords.tolist() 
            else:
                pos = tuple(elec.surf_coords)

            x,y,z = ['%.4f'%i for i in pos]

            row = [label_name, x, y, z]
            row.extend(elec.roi_list)

            writer.writerow(row)

def save_coordinates( electrodes, grid_types, snapping_completed=False,
                           file_type='csv'):
    savefile = ask_user_for_savefile(title='save %s file'%file_type)

    if savefile is None:
        return

    if file_type == 'csv':
        save_file_continuation = _save_csv_file
    elif file_type == 'montage':
        save_file_continuation = _save_montage_file

    save_file_continuation(savefile, electrodes, grid_types,
        snapping_completed=snapping_completed)

def save_coordinates_all( *args, **kwargs):
    return save_coordinates(*args, **kwargs)
def save_coordinates_grid( *args, **kwargs):
    return save_coordinates(*args, **kwargs)
