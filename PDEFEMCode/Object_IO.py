import os,pickle

def pickle_save(out_path, fname, obj_save, ext='.pkl'):
    '''
    Saves parameter
    :param fname:name of the file to save to. IGNORES EXTENSION.
    :param obj_save: the Python object to save. Prefer that this is a dictionary with objects to save.

    '''
    fname,_ = os.path.splitext(fname) # only keep root part
    with open(os.path.join(out_path,fname + ext),'wb') as f_obj:
        pickle.dump(obj_save, f_obj)

def pickle_load(fname,def_ext = '.pkl'):
    '''

    :param fname: full path of the file to load using pickle.
    :param def_ext: default file extension used to save files.
    :return: the Python object. This should be a dictionary if it has more than one parameter.
    '''
    fname,ext = os.path.splitext(fname)
    if not ext:
        ext = def_ext
    with open(fname + ext,'rb') as f_obj:
        return pickle.load(f_obj)