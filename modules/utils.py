#
# Utility functions which may be common to all modules
#
# Author    : Vikas Chouhan
# Copyright : Nurithm Labs Pvt. Ltd.
import pipes
import subprocess
import sys
import os
import glob
import csv
import math
import uuid
import sys
import re
import json
import shutil
import pickle
import locale
import math
import base64
import requests
import datetime
import random
from   colorama import Fore, Back, Style
import PIL.Image
import numpy as np
from   base64 import b64decode
from   PIL import ImageFile
import PIL.Image
import PIL.ImageFont
import PIL.ImageOps
import PIL.ImageDraw
import io
import itertools
import operator
import logging
import multiprocessing
from   sklearn.decomposition import PCA as sklearnPCA
from   sklearn.manifold import TSNE
import pandas as pd
import xlsxwriter
from   essential_generators import DocumentGenerator

NON_SPECIFIC_CLASS_NAME = 'Non specific'
NO_DIAGNOSIS_CLASS_NAME = 'No Diagnosis'

# Create single manager
mp_manager = multiprocessing.Manager()

#ImageFile.LOAD_TRUNCATED_IMAGES = True

supported_image_ext = ['*.jpg', '*.png', '*.jpeg', '*.bmp', '*.tiff']
supported_image_ext = [x.upper() for x in supported_image_ext] + supported_image_ext
csv_enc_default     = 'iso-8859-1'
csv_enc_utf8        = 'utf-8'

# Initialize logging
def init_logger(filename, level=logging.DEBUG):
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    # endfor
    logging.basicConfig(filename=filename, level=level, format='%(asctime)s - %(levelname)s : %(message)s', filemode='a')
# enddef

# Log message to log file. Also print on console
def log_msg(msg, level=logging.DEBUG, log_only=False):
    logging.log(level, msg)
    if not log_only:
        print(msg)
    # endif
# enddef

# static member decorator
def static(varname, value):
    def decorate(func):
        setattr(func, varname, value)
        return func
    return decorate
# enddef

# Set/Get module global static variables
@static('vars', {})
def static_var(x, y=None):
    if y is not None:
        static_var.vars[x] = y
        return y
    else:
        return None if x not in static_var.vars else static_var.vars[x]
    # endif
# enddef

def islist(obj):
    return isinstance(obj, list)
# enddef

def isdict(obj, key=None):
    d_t = isinstance(obj, dict)
    if d_t and key:
        return key in obj
    # endif
    return d_t
# enddef

def istuple(obj, size=None):
    if size is None:
        return isinstance(obj, tuple)
    else:
        return isinstance(obj, tuple) and len(obj) == size
    # endif
# enddef

def isint(obj):
    return isinstance(obj, int) or isinstance(obj, long)
# enddef

def isstr(obj):
    return isinstance(obj, str)
# enddef

def is_square(x):
    return False if (math.sqrt(x) - int(math.sqrt(x))) else True
# enddef

def sort_dict(x, by_value=True, reverse=False):
    key = 1 if by_value else 0
    return sorted(x.items(), key=operator.itemgetter(key), reverse=reverse)
# enddef

# String trailing & leading spaces
def ssp(x):
    return x.lstrip().rstrip()
# enddef

def bytesio_to_img(img_bytes):
    im = PIL.Image.open(io.BytesIO(img_bytes))
    return np.array(im)
# enddef

def base64_to_img(img_str):
    return bytesio_to_img(b64decode(img_str))
# enddef

def search_for_images(image_dir, full_path=False):
    file_list = []
    for ext_t in supported_image_ext:
        image_list_t = glob.glob1(image_dir, ext_t)
        # Add image_dir if full_path is required
        if full_path:
            image_list_t = ['{}/{}'.format(image_dir, x) for x in image_list_t]
        # endif
        file_list = file_list + image_list_t
    # endfor
    return file_list
# enddef

def search_for(dir_t, file_ext_list=None, full_path=False):
    file_ext_list = ['*'] if file_ext_list is None else \
            [file_ext_list] if not isinstance(file_ext_list, list) else file_ext_list
    file_list = []
    for ext_t in file_ext_list:
        file_l_t  = glob.glob1(dir_t, ext_t)
        # Add dir_t if full_path is required
        if full_path:
            file_l_t = ['{}/{}'.format(dir_t, x) for x in file_l_t]
        # endif
        file_list = file_list + file_l_t
    # endfor
    return file_list
# enddef

def search_for_images_in_subdirs(top_dir, full_path=False):
    dir_dict = {}
    dir_l = [os.path.basename(x.rstrip('/')) for x in glob.glob('{}/*/'.format(top_dir))]
    for dir_t in dir_l:
        dir_dict[dir_t] = search_for_images('{}/{}'.format(top_dir, dir_t), full_path=full_path)
    # endfor
    return dir_dict
# enddef

# @args :-
#           image_dir      : top level directory (can be flat dir of images or may contain sub directories
#                            , each subdirectory containing it's own images).
#           flat_dir_label : label name to assign if top_dir is flat directory.
# @returns :-
#           list of (image_file, label_name)
def search_for_images_in_dirhir(image_dir, flat_dir_label=NON_SPECIFIC_CLASS_NAME):
    # First try to search in subdirs (i.e. if there are any subdirs)
    image_ddict   = search_for_images_in_subdirs(image_dir)
    num_subdirs   = len(image_ddict.keys())
    # If there is at least one subdir, then it's assumed that we have been given a directory of
    # sub-directories, each subdirectory being the label names of the non-specific disease class
    # If no subdirs are there, it's assumed that we have been given flat dir hierarchy.
    if num_subdirs > 0:
        print('***** Picking up {} subdirs in {}'.format(len(image_ddict.keys()), image_dir))
        image_tlist   = [('{}/{}/{}'.format(image_dir, k, z), k) for (k, v) in image_ddict.items() for z in v]
    else:
        print('**** No subdirs found. Assuming a flat hierarchy.')
        image_tlist   = [(x, flat_dir_label) for x in search_for_images(image_dir, full_path=True)]
        
        if len(image_tlist) == 0:
            print('No images found in {}'.format(image_dir))
        else:
            print('**** Found {} images in {}'.format(len(image_tlist), image_dir))
        # endif
    # endif
    return image_tlist
# enddef


def search_for_subdirs(top_dir, full_path=True):
    if full_path:
        dir_l = [x.rstrip('/') for x in glob.glob('{}/*/'.format(top_dir))]
    else:
        dir_l = [x for x in glob.glob1(top_dir, '*') if os.path.isdir('{}/{}'.format(top_dir, x))]
    # endif
    return dir_l
# enddef

def search_for_subdirs_with_pattern(top_dir, pattern, full_path=True):
    dir_l = [x for x in glob.glob1(top_dir, pattern) if os.path.isdir('{}/{}'.format(top_dir, x))]
    dir_l = ['{}/{}'.format(top_dir, x) for x in dir_l] if full_path else dir_l
    return dir_l
# enddef

def parse_label_file(label_file, key_is_name=False):
    label_map  = {}
    row_list   = parse_csv_file(label_file)
    for item_t in row_list:
        label_name = item_t[0]
        label_num  = int(item_t[1])
        if key_is_name:
            label_map[label_name] = label_num
        else:
            label_map[label_num] = label_name
        # endif
    # endfor
    return label_map
# enddef

def parse_csv_file(csv_file, encoding=csv_enc_default, strip_spaces=True,
        delimiter=',', fix_rows=False, fix_index_list=None):
    # Ignore comments and whitelines
    def __ign_line(row_t):
        line_t = delimiter.join(row_t)
        if re.match('^\s*#', line_t) or re.match('^\s*$', line_t):
            return True
        # endif
        return False
    # enddef

    encoding   = locale.getpreferredencoding() if encoding == None else encoding
    csv_reader = csv.reader(open(csv_file, 'r', encoding=encoding), delimiter=delimiter)

    # Ignore comments and whitespaces
    row_list   = [x for x in filter(lambda z: not __ign_line(z), csv_reader)]
    # Strip spaces if enabled. By-default enabled.
    row_list   = [[y.strip() for y in x] if strip_spaces else x for x in row_list]
    # Fix specific indexes for unicode etc
    row_list   = [ x for x in row_list ] if fix_rows == False else \
            [ fix_rowl_ascii(x, index_list=fix_index_list) for x in row_list ]

    return row_list
# enddef

def write_csv(row_list, csv_file, is_header=False, encoding=csv_enc_default):
    # NOTE: is_header is ignore at this time
    with open(csv_file, 'w', encoding=encoding) as out_csv:
        csv_writer = csv.writer(out_csv)
        csv_writer.writerows(row_list)
    # endwith
# enddef

def parse_label_include_file(label_file):
    row_list   = parse_csv_file(label_file)
    mapping_dict = {}
    for item_t in row_list:
        item_0       = item_t[0]
        item_1       = item_t[1] if len(item_t) >= 2 else ''
        item_2       = item_t[2] if len(item_t) >= 3 else ''
        label_name   = ssp(item_0) if item_1 == '' else ssp(item_1)
        train_mode   = 'all' if item_2 == '' else item_2

        mapping_dict[ssp(item_0)] = { 'mapped' : label_name, 'train_mode' : train_mode}
    # endfor
    return mapping_dict
# enddef

def parse_source_label_include_file(label_file):
    mapping_dict = label_file if isinstance(label_file, dict) else \
            parse_label_include_file(label_file) if isinstance(label_file, str) else label_file
    return {k:v['mapped'] for k,v in mapping_dict.items()}
# enddef

def get_train_modes_from_label_include_file(label_file):
    mapping_dict = label_file if isinstance(label_file, dict) else \
            parse_label_include_file(label_file) if isinstance(label_file, str) else label_file
    train_mode_dict = {}
    for label_t in mapping_dict:
        train_mode_t = mapping_dict[label_t]['train_mode']
        if train_mode_t not in train_mode_dict:
            train_mode_dict[train_mode_t] = []
        # endif
        # Append current label
        train_mode_dict[train_mode_t].append(label_t)
    # endfor
    return train_mode_dict
# enddef

def generate_class_distribution_map(metadata_file, label_file=None):
    label_map = {}
    # Parse label file if availabel
    if label_file:
        reader = csv.reader(open(label_file, 'r'))
        for x in reader:
            label_map[int(x[1])] = x[0]
        # endfor
    # endif

    class_distr_map = {}
    # Parse metadata file
    reader = csv.reader(open(metadata_file, 'r'))
    for x in reader:
        key = int(x[1]) if int(x[1]) not in label_map else label_map[int(x[1])]
        if key not in class_distr_map:
            class_distr_map[key] = []
        # endif
        # Append item
        class_distr_map[key].append(x)
    # endfor

    return class_distr_map
# enddef

def concat_files(in_file_list, out_file):
    assert isinstance(in_file_list, list), '{} should be list type !!'.format(in_file_list)
    with open(rp(out_file), 'wt') as fd_out:
        for in_file_t in in_file_list:
            with open(rp(in_file_t), 'rt') as fd_in:
                shutil.copyfileobj(fd_in, fd_out)
            # endwith
        # endfor
    # endwith
# enddef

def assertvd(d, v, msg=None):
    msg = msg if msg else '"{}" not found in {}.'.format(v, d)
    assert v in d, msg
# enddef

def mkdir(dir):
    if dir == None:
        return None
    # endif
    if not os.path.isdir(dir):
        os.makedirs(dir, exist_ok=True)
    # endif
# enddef

def rmdir(dir):
    shutil.rmtree(dir, ignore_errors=True)
# enddef

def rp(dir):
    if dir == None:
        return None
    # endif
    if dir[0] == '.':
        return os.path.normpath(os.getcwd() + '/' + dir)
    else:
        return os.path.normpath(os.path.expanduser(dir))
# enddef

def filename(x, only_name=True):
    n_tok = os.path.splitext(os.path.basename(x))
    return n_tok[0] if only_name else n_tok
# enddef

def chkdir(dir):
    if not os.path.isdir(dir):
        print('{} does not exist !!'.format(dir))
        sys.exit(-1)
    # endif
# enddef

def chkfile(file_path, exit=False):
    if os.path.exists(file_path) and os.path.isfile(file_path):
        return True
    # endif
    if exit:
        print('{} does not exist !!'.format(file_path))
        sys.exit(-1)
    # endif
        
    return False
# enddef

def npath(path):
    return os.path.normpath(path) if path else None
# enddef


def parse_sarg(x, cast_to=str, arg_mod_fn=None, delimiter=','):
    if x:
        if arg_mod_fn:
            return [arg_mod_fn(cast_to(y.rstrip().lstrip())) for y in x.split(',')]
        else:
            return [cast_to(y.rstrip().lstrip()) for y in x.split(delimiter)]
        # endif
    # endif
    return None
# enddef

def date_str():
    return str(datetime.datetime.now()).replace(' ', '_').replace(':', '_').replace('.', '_')
# enddef

def ceil_sqrt(x):
    y = math.log2(x)
    return int(y) if y == int(y) else int(y) + 1
# enddef

def precision(x, y=2):
    return int(x*10**y)/(10**y)
# enddef

def rclone_gdrive_remote(remote_dir, local_dir):
    remote_host = subprocess.check_output(['rclone', 'listremotes']).decode('utf-8').rstrip('\n')
    remote_path = remote_host + remote_dir
    # Clone remote dir
    resp = subprocess.call(['rclone', 'copy', remote_path, local_dir, '--checksum', '--tpslimit', '10'])
    return resp
# enddef

# A markov chain based unique words like string generator.
# exl_list is used to ensure that no collisions happen, when
# we are generating a list of such unique strings
def ustr(excl_list=None, max_tries=50, max_iter=2):
    doc_gen    = DocumentGenerator()
    num_words  = random.randint(1, max_iter)
    removelist = "-"
    excl_list  = [] if excl_list is None else excl_list

    try_cnt    = 1
    while True:
        str_t      = ''
        for i in range(num_words):
            str_t = str_t + '_' + re.sub(r'[^\w'+removelist+']', '', doc_gen.slug()).replace('-', '_')
        # endfor

        if str_t not in excl_list:
            break
        # endif

        if try_cnt > max_tries:
            raise ValueError('max tries {} exceeded in ustr()'.format(max_tries))
        # endif
        try_cnt += 1
    # endwhile

    return str_t.strip('_')
# enddef

def u4(prefix=None):
    u4_str = str(uuid.uuid4()) + '__' + str(uuid.uuid4())
    return '{}__{}'.format(prefix, u4_str) if prefix else u4_str
# enddef

def filename(file_path, remove_ext=True):
    _file_name = os.path.basename(file_path)
    return os.path.splitext(_file_name)[0] if remove_ext else _file_name
# enddef

def sanitize_name(name_str):
    return re.sub(r'\W+', '_', name_str)
# enddef 

def qstr(x):
    return "\"" + str(x) + "\""
# enddef

def copy_file(src, dst, follow_symlinks=False):
    shutil.copy(src, dst, follow_symlinks=follow_symlinks)
# enddef

def soft_link(src, dst):
    if os.path.islink(dst):
        return
    if os.path.isfile(src) and os.path.isdir(dst):
        dst = dst + '/' + os.path.basename(src)
    # endif
    os.symlink(src, dst)
# enddef

def load_image(img_file, resize_to=None):
    pil_image = PIL.Image.open(img_file).convert('RGB')
    img_vec   = np.asarray(pil_image.resize(resize_to[:2])) if resize_to else np.asarray(pil_image)
    return img_vec
# enddef

def chk_image(img_file):
    try:
        PIL.Image.open(img_file).convert('RGB')
        return True
    except OSError as e:
        if re.search('image file is truncated', str(e)):
            return False
        else:
            raise e
        # endif
    # endtry
# enddef

def save_image(img_vec, dst_path):
    PIL.Image.fromarray(img_vec).save(dst_path)
# enddef

# Copy in RGB mode
def copy_image(src, dst):
    PIL.Image.open(src).convert('RGB').save(dst)
# enddef

# Width first then height
def get_image_dims(img_file):
    im_size = PIL.Image.open(img_file).size
    return {'width' : im_size[0], 'height' : im_size[1]}
# enddef

def parse_split_ratio(split_ratio):
    try:
        r_arr = [int(x) for x in split_ratio.split(':')]
    except:
        print('split_ratio {} not in proper format. Eg 50:30:20.'.format(split_ratio))
        sys.exit(-1)
    # endtry
    assert sum(r_arr) == 100, 'Sum of splits in split_ratio {} should add up to 100'.format(split_ratio)
    return r_arr
# enddef

# Create image gallery for easy plotting
def image_gallery(im_array, ncols=3):
    nindex, height, width, channels = im_array.shape
    nrows = nindex//ncols
    assert nindex == nrows*ncols

    result = (im_array.reshape(nrows, ncols, height, width, channels)
              .swapaxes(1,2)
              .reshape(height*nrows, width*ncols, channels))
    return result
# enddef

# color a message
def cmsg(message, fore=None, back=None, style=None):
    all_nms  = [Fore, Back, Style]
    all_eles = [fore, back, style]
    bool_v   = [1 if x is not None else 0 for x in all_eles]
    assert sum(bool_v) == 1, 'ERROR::: Only one of fore, back or style should be defined at a time !!'
    stylep   = all_nms[bool_v.index(1)]
    style    = all_eles[bool_v.index(1)]
    assert style in stylep.__dict__, '{} should be one of {}'.format(stylep.__dict__.keys())

    return stylep.__dict__[style] + message + Style.RESET_ALL
# enddef


def text_to_image(text_path, image_path, font_path=None):
    """Convert text file to a grayscale image with black characters on a white background.

    arguments:
    text_path   - the content of this file will be converted to an image
    image_path  - the target image path
    font_path   - path to a font file (for example impact.ttf)
    """
    grayscale = 'L'
    PIXEL_ON = 0  # PIL color to use for "on"
    PIXEL_OFF = 255  # PIL color to use for "off"
    
    # parse the file into lines
    with open(text_path) as text_file:  # can throw FileNotFoundError
        lines = tuple(l.rstrip() for l in text_file.readlines())
    # endwith

    # choose a font (you can see more detail in my library on github)
    large_font = 20  # get better resolution with larger size
    font_path = font_path or 'cour.ttf'  # Courier New. works in windows. linux may need more explicit path
    try:
        font = PIL.ImageFont.truetype(font_path, size=large_font)
    except IOError:
        font = PIL.ImageFont.load_default()
        print('Could not use chosen font. Using default.')
    # endtry

    # make the background image based on the combination of font and lines
    pt2px = lambda pt: int(round(pt * 96.0 / 72))  # convert points to pixels
    max_width_line = max(lines, key=lambda s: font.getsize(s)[0])
    # max height is adjusted down because it's too large visually for spacing
    test_string = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    max_height = pt2px(font.getsize(test_string)[1])
    max_width = pt2px(font.getsize(max_width_line)[0])
    height = max_height * len(lines)  # perfect or a little oversized
    width = int(round(max_width + 40))  # a little oversized
    image = PIL.Image.new(grayscale, (width, height), color=PIXEL_OFF)
    draw = PIL.ImageDraw.Draw(image)

    # draw each line of text
    vertical_position = 5
    horizontal_position = 5
    line_spacing = int(round(max_height * 0.8))  # reduced spacing seems better
    for line in lines:
        draw.text((horizontal_position, vertical_position),
                  line, fill=PIXEL_ON, font=font)
        vertical_position += line_spacing
    # endfor

    # crop the text
    c_box = PIL.ImageOps.invert(image).getbbox()
    image = image.crop(c_box)

    image.save(image_path)
    return image
# enddef

# Write dictionary of dataframes to one single excel workbook
def df_to_excel(dfs, filename):
    assert isdict(dfs), 'Input (dfs) should be a dictionary of sheets !!'
    writer = pd.ExcelWriter(filename, engine='xlsxwriter')
    for sheetname, df in dfs.items():  # loop through `dict` of dataframes
        df.to_excel(writer, sheet_name=sheetname)  # send df to writer
        worksheet = writer.sheets[sheetname]  # pull worksheet object
        for idx, col in enumerate(df):  # loop through all columns
            series = df[col]
            max_len = max((
                series.astype(str).map(len).max(),  # len of largest item
                len(str(series.name))  # len of column name/header
                )) + 1  # adding a little extra space
            worksheet.set_column(idx+1, idx+1, max_len)  # set column width
        # endfor
        # Fix index
        worksheet.set_column(0, 0, max((df.index.astype(str).map(len).max(), len(str(df.index.name)))))
    # endfor
    writer.save()
# enddef

def printf(statement, file_handle=None, shell=True):
    if file_handle:
        print(statement, file=file_handle)
    # endif
    if shell:
        print(statement)
    # endif
# enddef

def cwargs(kwargs, key, value=sys.maxsize):
    if key not in kwargs:
        if value == sys.maxsize:
            raise ValueError('{} needs to be defined.'.format(key))
        else:
            return value
        # endif
    else:
        return kwargs[key]
    # enddef
# enddef

def load_json(json_file):
    return json.load(open(json_file, 'r'))
# enddef

def save_json(json_obj, json_file, indent=4):
    json.dump(json_obj, open(json_file, 'w'), indent=indent)
# enddef

def load_pickle(pickle_file):
    return pickle.load(open(pickle_file, 'rb')) if os.path.isfile(pickle_file) else None
# enddef

def save_pickle(pickle_obj, pickle_file):
    pickle.dump(pickle_obj, open(pickle_file, 'wb'))
# enddef

def split_chunks(l, n_chunks):
    n = int(len(l)/n_chunks)
    retv = [l[i*n:(i+1)*n] for i in range(int(len(l)/n)+1) if l[i*n:(i+1)*n] != []]
    return retv[0:n_chunks-1] + [list(itertools.chain(*retv[n_chunks-1:]))]
# enddef

def to_categorical(np_array):
    return np.eye(len(np.unique(np_array.reshape(-1))))[np_array.reshape(-1)]
# enddef

# Resolve data files
def re_dataset(dataset_dir):
    return '{}/images'.format(dataset_dir), '{}/annot.csv'.format(dataset_dir), '{}/labels.csv'.format(dataset_dir)
# enddef

# Resolve top level dataset hierarchy
def re_tdataset(dataset_dir):
    return '{}/train'.format(dataset_dir), '{}/val'.format(dataset_dir)
# enddef

def all_of_same_length(l_of_l):
    # Add check
    for x in l_of_l:
        if not isinstance(x, list):
            raise ValueError('{} is not list of lists.'.format(l_of_l))
        # endif
    # endfor
    len_of_l = [len(x) for x in l_of_l]
    if all(np.asarray(len_of_l) == len_of_l[0]):
        return True
    # endif
    return False
# enddef

def chk_any_is_none(arg_l):
    for arg_t in arg_l:
        if arg_t == None:
            raise ValueError('{} is None !!'.format(arg_t))
        # endif
    # endfor
# enddef

# Read argument from config dict
def cf_arg(conf_dict, key, def_val=None, cast_to=None):
    if key in conf_dict:
        if conf_dict[key] == '':
            value = def_val
        else:
            value = conf_dict[key]
        # endif
    elif def_val is not None:
        value = def_val
    else:
        value = None
    # endif

    if value == None or cast_to == None:
        return value
    # endif

    def __cast(x):
        if cast_to is bool:
            return distutils.util.strtobool(x) if isinstance(x, str) else x
        else:
            return cast_to(x)
        # endif
    # enddef

    # if value is of dict type, flag an error
    if isinstance(value, dict):
        raise ValueError('key {} cannot be a dictionary !!'.format(key))
    # endif

    # if list, apply cast_to to every value
    if isinstance(value, list):
        return [__cast(y) for y in value]
    # endif
    return __cast(value)
# enddef

# Print help for arg_map if available
def help_args(arg_map):
    print('Help :')
    for arg_t in arg_map:
        def_val   = arg_map[arg_t]['default'] if 'default' in arg_map[arg_t] else None
        cast_to   = arg_map[arg_t]['type'] if 'type' in arg_map[arg_t] else None
        help_arg  = arg_map[arg_t]['help'] if 'help' in arg_map[arg_t] else ''
        print('    {} : ({}), {}'.format(str.rjust(arg_t, 20), str(cast_to), help_arg))
    # endfor
# enddef

def check_arg_map(arg_map):
    for arg_t in arg_map:
        assert 'default' in arg_map[arg_t], 'key="default" not found in arg_map["{}"]'.format(arg_t)
        assert 'type'    in arg_map[arg_t], 'key="type" not found in arg_map["{}"]'.format(arg_t)
        assert 'lambda'  in arg_map[arg_t], 'key="lambda" not found in arg_map["{}"]'.format(arg_t)
        assert 'help'    in arg_map[arg_t], 'key="help" not found in arg_map["{}"]'.format(arg_t)
    # endfor
# enddef


# Parse config file which is meant for passing arguments
def parse_argcfg_file(config_file, arg_map, override_map=None):
    json_t = json.load(open(config_file, 'r'))
    check_arg_map(arg_map)

    # Additional warning
    li_adi_args = list(set(list(json_t.keys())) - set(list(arg_map.keys())))
    if len(li_adi_args) != 0:
        print('******************************************************************')
        print('''WARNING ::::::: Additional parameters {} found in config file,
                                 which are not defined by args_map !!!!!!!!!!'''.format(li_adi_args))
        print('******************************************************************')
    # endif

    results_map = {}
    # Read arg_map
    for arg_t in arg_map:
        def_val = arg_map[arg_t]['default'] if 'default' in arg_map[arg_t] else None
        cast_to = arg_map[arg_t]['type'] if 'type' in arg_map[arg_t] else None
        opt_fn  = arg_map[arg_t]['lambda'] if 'lambda' in arg_map[arg_t] else None

        # Check override map
        if override_map:
            if arg_t in override_map:
                # Overwrite in config dict
                json_t[arg_t] = override_map[arg_t]
            # endif
        # endif

        # Get value from config dict
        result_this = cf_arg(json_t, arg_t, def_val=def_val, cast_to=cast_to)
        # Check results type
        if opt_fn:
            if isinstance(result_this, list):
                result_this = [opt_fn(x) for x in result_this]
            else:
                result_this = opt_fn(result_this)
            # endif
        # endif
        # Assign
        results_map[arg_t] = result_this
    # endfor

    print('args => {}'.format(results_map))
    return results_map
# enddef

# Parse override args in string format
def parse_str_args(arg_str):
    # Each key=value pair needs to be seperated by ';'
    args_list = [x.split('=') for x in parse_sarg(arg_str, delimiter=';')]
    # Collect key value pair in each
    args_map  = {arg_pair_t[0] : parse_sarg(arg_pair_t[1], delimiter=',') for arg_pair_t in args_list}
    # Sanitize arg_map
    for k,v in args_map.items():
        if len(v) == 1:
            args_map[k] = v[0]
        # endif
    # endfor

    return args_map
# enddef

# Write args str to config json
def write_config(arg_str, json_file):
    args_map = parse_str_args(arg_str)
    json.dump(args_map, open(json_file, 'w'), indent=4)
# enddef

# Encode one or more indexes to ascii
def fix_rowl_ascii(row_l, index_list=None):
    index_list = [x for x in range(len(row_l))] if index_list == None else index_list
    assert max(index_list) < len(row_l) and min(index_list) >= 0, 'Invalid index list.'
    for index_t in index_list:
        row_l[index_t] = row_l[index_t].encode('ascii', 'ignore').decode(locale.getpreferredencoding())
    # endfor

    return row_l
# enddef

# Merge feature vectors from multi class model configuration
# @args :
#           featsv_list -> a list of features generated from each component models
# NOTE  : Each row of the feature vector corresponds to one channel (i.e one crop)
#         Please refer to OpenMAX documentation for what channel/crop means
def merge_fvs_multi(featv_list):
    # Reshape individual feature matrix to a flat array
    featv_list = [x.reshape([x.shape[0], -1]) for x in featv_list]
    return np.hstack(featv_list)
# enddef


# define functions for mapping column map to indexes
class ColumnMap(object):
    KEY_ORIG_LABEL_NAME = 'orig_label_name'
    KEY_LABEL_NAME      = 'label_name'
    KEY_ORIG_FILE_NAME  = 'orig_file_name'
    KEY_FILE_NAME       = 'file_name'
    KEY_LABEL_NUM       = 'label_num'
    KEY_BBOX_XMIN       = 'bbox_xmin'
    KEY_BBOX_XMAX       = 'bbox_xmax'
    KEY_BBOX_YMIN       = 'bbox_ymin'
    KEY_BBOX_YMAX       = 'bbox_ymax'

    column_names = [KEY_ORIG_LABEL_NAME, KEY_LABEL_NAME, KEY_ORIG_FILE_NAME,
            KEY_FILE_NAME, KEY_LABEL_NUM, KEY_BBOX_XMIN, KEY_BBOX_YMIN,
            KEY_BBOX_XMAX, KEY_BBOX_YMAX]
    def __init__(self, map_dict):
        # Add checks
        not_found_list = list(set(self.column_names) - set(map_dict.keys()))
        if len(not_found_list) != 0:
            # Just issue a warning. It's not necessary for column map to have all fields
            print('WARNING!!! column names "{}" not found in map_dict.'.format(not_found_list))
        # endif

        self.map_dict = map_dict
    # enddef

    def __int(self, x):
        return int(x)
    # enddef

    def orig_label_name(self):
        return self.__int(self.map_dict[self.KEY_ORIG_LABEL_NAME])
    # enddef
    def label_name(self):
        return self.__int(self.map_dict[self.KEY_LABEL_NAME])
    # enddef
    def orig_file_name(self):
        return self.__int(self.map_dict[self.KEY_ORIG_FILE_NAME])
    # enddef
    def file_name(self):
        return self.__int(self.map_dict[self.KEY_FILE_NAME])
    # enddef
    def label_num(self):
        return self.__int(self.map_dict[self.KEY_LABEL_NUM])
    # enddef
    def bbox_xmin(self):
        return self.__int(self.map_dict[self.KEY_BBOX_XMIN])
    # enddef
    def bbox_ymin(self):
        return self.__int(self.map_dict[self.KEY_BBOX_YMIN])
    # enddef
    def bbox_xmax(self):
        return self.__int(self.map_dict[self.KEY_BBOX_XMAX])
    # enddef
    def bbox_ymax(self):
        return self.__int(self.map_dict[self.KEY_BBOX_YMAX])
    # enddef
# endclass

# Some column maps
column_map_stage00 = {'file_name' : 0, 'label_name' : 1, 'bbox_xmin' : 2, 'bbox_ymin' : 3,
            'bbox_xmax' : 4, 'bbox_ymax' : 5, 'orig_file_name' : -1}
column_map_stage0 = {'file_name' : 0, 'label_num' : 1, 'bbox_xmin' : 2, 'bbox_ymin' : 3,
            'bbox_xmax' : 4, 'bbox_ymax' : 5, 'orig_label_name' : -3, 'orig_file_name' : -2,
            'label_name' : -1}
column_map_stage2 = {'file_name' : 0, 'label_num' : 1, 'label_name' : 2}

#
# Generate distribution for segments
# @args :
#           image_list  -> list of images for which segments have to be generated.
#           image_map   -> a dict which maps image names to annotation rows
#                          such that row[0] -> image name
#                                    row[1] -> label name (or number)
#                                    row[2] -> xmin
#                                    row[3] -> ymin
#                                    row[4] -> xmax
#                                    row[5] -> ymax
#           in_img_dir  -> input iamge dir
#           out_img_dir -> output image dir
#           label_split -> Label for this split (eg. 'train', 'val' etc.)
#           discard_full_segments -> discard segments which cover full image.
#                          NOTE: This is usually fixed to True for training,
#                                but may be fixed to False for validation.
#                                Reason is there are so many images in our dataset which don't have
#                                bounding boxes, so they are created artificially to box the full image.
#                                If this option is set to
#                                False, then while creating dataset split for validation, all of those images
#                                would be discarded.
def generate_segments(image_list, image_map, in_img_dir, out_img_dir=None, label_split=None, discard_full_segments=True,
        column_map=column_map_stage0):

    # Get indexes
    i_obj    = ColumnMap(column_map)
    i_xmin   = i_obj.bbox_xmin()
    i_xmax   = i_obj.bbox_xmax()
    i_ymin   = i_obj.bbox_ymin()
    i_ymax   = i_obj.bbox_ymax()
    i_lnum   = i_obj.label_num()
    i_lname  = i_obj.label_name()

    seg_images_label_map = {}
    label_split = '' if label_split == None else label_split
    # Output image dir is same as input image dir if not specified
    out_img_dir = in_img_dir if out_img_dir == None else out_img_dir

    print('>> COPY {} SEG IMAGES.'.format(label_split))
    for img_t in image_list:
        img_path_t = '{}/{}'.format(in_img_dir, img_t)
        src_img = load_image(img_path_t)
        im_dims = get_image_dims(img_path_t)

        img_bboxes_t = image_map[img_t]
        patch_ctr = 0
        for bbox_t in img_bboxes_t:
            xmin, ymin, xmax, ymax = int(bbox_t[i_xmin]), int(bbox_t[i_ymin]), int(bbox_t[i_xmax]), int(bbox_t[i_ymax])

            ## Do some basic checks. There may be errors in labelling, so such errors may creep in.
            # We discard zero area bounding boxes here
            if (xmax - xmin) <= 0 or (ymax - ymin) <= 0:
                print('ERROR !! Encountered a zero area bounding box in {}. This may be an error in labelling. Skipping'.format(img_t))
                continue
            # endif
            if (xmax < 0 or xmin < 0 or ymax < 0 or ymin < 0):
                print('ERROR !! Got negative coordinates in  {}. This may be an error in labelling (or labelling tool). Skipping'.format(img_t))
                continue
            # endif
            ##
            if discard_full_segments == True:
                # Check if the bounding box covers the full image. If yes, don't add this segment
                if (xmin == 0 and ymin == 0 and xmax == (im_dims['width'] -1) and ymax == (im_dims['height'] - 1)):
                    print('WARNING !! Encountered a full scale bounding box in {}. Skipping.'.format(img_t), end='\r')
                    continue
                # endif
            # endif

            # Get slice
            #print('{}'.format(img_t))
            #print('xmin = {}, ymin = {}, xmax = {}, ymax = {}\n'.format(xmin, ymin, xmax, ymax))
            img_tt = src_img[ymin:ymax, xmin:xmax, ]
            (fname, fext) = os.path.splitext(img_t)

            # Write patch to file
            dst_file = '{}_{}_{}{}'.format(fname, patch_ctr, bbox_t[i_lnum], fext)
            dst_path = '{}/{}'.format(out_img_dir, dst_file)
            #print('>> COPY TRAIN SEG IMAGES : {}      '.format(dst_file), end='\r')
            save_image(img_tt, dst_path)

            # Add annotation map
            seg_images_label_map[dst_file] = (bbox_t[i_lnum], bbox_t[i_lname])

            # Increment patch counter
            patch_ctr = patch_ctr + 1
        # endfor
    # endfor

    print('INFO:: Complete generate_segments.')

    return seg_images_label_map
# enddef

#
# Overloads of earlier function 
# @args :
#           metadata -> if string, then a metadata file is assumed.
#                       if dict, then a metadata map (file_name -> row) format
#           Other parameters same as "generate_segments"
def generate_segments_using_metadata(metadata, in_image_dir, out_image_dir=None, label_split=None):
    if isinstance(metadata, str):
        row_list = parse_csv_file(metadata)
        metadata_map = {x[0] : x for x in row_list}
        image_list   = [x[0] for x in row_list]
    elif isinstance(metadata, dict):
        metadata_map = metadata
        image_list   = [x[0] for x in metadata_map]
    else:
        print('Wring metadata format.')
        sys.exit(-1)
    # endif

    return generate_segments(image_list, metadata_map, in_image_dir, out_image_dir, label_split)
# enddef

# Call any function which returns a result using multiprocessing
# @args :
#           worker_fn        -> worker function
#           num_threads      -> no of threads or workers to spawn in parallel
#           sanitize_results -> To join results for all workers
#                               if True, then results are joined
#                               if False, they are returned as such
#           kwars            -> arguments to be passed to worker_fn
def spawn_workers(worker_fn, num_threads, sanitize_results=True, **kwargs):
    if 'data_keys' not in kwargs:
        print('**kwargs in spawn_workers need to have named key "data_keys"')
        sys.exit(-1)
    # endif

    # Get list of named parameters which need to be divided
    data_keys = kwargs.pop('data_keys')
    if not isinstance(data_keys, list):
        print('kwargs["data_keys"] should be a list of keys which need to be divided across processes.')
        sys.exit(-1)
    # endif

    # Check if proc_id is to specified for the worker
    if 'proc_id_key' in kwargs:
        proc_id_key = kwargs.pop('proc_id_key')
    else:
        proc_id_key = None
    # endif

    # Divide payloads
    chunk_map   = {}
    count_map   = {}
    for data_key_t in data_keys:
        if data_key_t not in kwargs:
            print('FATAL::: {} not found in kwargs for spawn_workers.'.format(data_key_t))
        # endif
        # Check if keys are list or not
        if not isinstance(kwargs[data_key_t], list):
            print(kwargs[data_key_t], ' is not a list.')
            sys.exit(-1)
        # endif
        data_value_t = kwargs.pop(data_key_t)
        count_map[data_key_t] = len(data_value_t)  # Get length of this list
        chunk_map[data_key_t] = split_chunks(data_value_t, num_threads)
    # endfor

    if len(set(list(count_map.values()))) != 1:
        print('data_keys in spawn_workers have mismatch in length of parameters : {}'.format(count_map))
        sys.exit(-1)
    # endif

    # Define a new worker fn which wraps original worker fn with some customizations
    def __wrap_worker_fn(return_dict, proc_id, **xkwargs):
        ret_result = worker_fn(**xkwargs)
        return_dict[proc_id] = ret_result
    # enddef

    print('INFO::: Dividing {} jobs across {} threads.'.format(list(count_map.values())[0], num_threads))

    # Deploy
    result_list  = []
    process_list = []
    return_dict  = mp_manager.dict()
    for i in range(num_threads):
        # Populate kwargs
        for data_key_t in chunk_map:
            kwargs[data_key_t] = chunk_map[data_key_t][i]
        # endfor
        if proc_id_key:
            kwargs[proc_id_key] = i
        # endif
        # Spawn new process
        proc = multiprocessing.Process(target=__wrap_worker_fn, args=(return_dict,i,), kwargs=kwargs)
        process_list.append(proc)
        proc.start()
    # endfor

    # Wait for all processes to end
    for proc_t in process_list:
        proc_t.join()
    # endfor

    print('INFO:: All processes finished !!')

    # Collate all results
    all_results = []
    for i in range(num_threads):
        chunk_result = return_dict[i]
        all_results.append(chunk_result)
    # endfor

    # Sanitize all_results
    if sanitize_results:
        if isinstance(all_results[0], list):
            all_results = [x for z in all_results for x in z]
        elif isinstance(all_results[0], dict):
            all_results = {x:y for z in all_results for (x,y) in z.items()}
        # endif
    # endif

    return all_results
# enddef

# Fire and forget
# Operations which we don't need to keep track
def fire_and_forget(worker_fn, **kwargs):
    proc = multiprocessing.Process(target=worker_fn, kwargs=kwargs)
    proc.start()
# enddef

# Map prediction scores fro old label domain to new label domain
# This is usually used for post prediction merging of labels.
def map_labels_on_predictions(pred_scores, labels_to_names, mapping_dict=None, new_labels_to_names=None):
    pred_scores_map     = {(labels_to_names[i] if i in labels_to_names else NON_SPECIFIC_CLASS_NAME):pred_scores[i] for i in range(len(pred_scores))}
    pred_scores_map_fix = {}
    mapping_ldict_inv   = {}
    mapping_dict        = {} if mapping_dict is None else mapping_dict

    # Get { mapped_v : list of labels being mapped }
    for key, value in mapping_dict.items():
        if value not in mapping_ldict_inv:
            mapping_ldict_inv[value] = []
        # endif
        mapping_ldict_inv[value].append(key)
    # endfor

    # Populate initial fix map.
    # It only contains mapped keys
    pred_scores_map_fix = {key: sum([pred_scores_map[x] for x in value]) for key,value in mapping_ldict_inv.items()}
    original_keys_list  = list(itertools.chain(*mapping_ldict_inv.values()))
    # Add remaining keys
    for key_t in pred_scores_map:
        if key_t not in original_keys_list:
            original_sum = pred_scores_map_fix[key_t] + 0.0 if key_t in pred_scores_map_fix else 0.0
            pred_scores_map_fix[key_t] = original_sum + pred_scores_map[key_t]
        # endif
    # endfor

    # Get new labels_to_names and pred scores_list (after above merging process)
    label_list  = list(pred_scores_map_fix.keys())
    label_map   = {i:v for i,v in enumerate(label_list)} if new_labels_to_names is None else new_labels_to_names
    scores_list = [pred_scores_map_fix[i] for i in label_list]

    return scores_list, label_map
# enddef

@static("new_labels_to_names", None)
@static("new_names_to_labels", None)
def calibrate_predictions_to_mapped_labels(pred_scores, pred_labeln,
        old_labels_to_names, mapping_dict=None, act_labeln=None):
    # If mapping_dict is invalid or None or {}
    # Don't change the state
    if mapping_dict is None or mapping_dict == {}:
        calibrate_predictions_to_mapped_labels.new_labels_to_names = old_labels_to_names
        calibrate_predictions_to_mapped_labels.new_names_to_labels = {v:k for k,v in old_labels_to_names.items()}

        if act_labeln is not None:
            return pred_scores, pred_labeln, act_labeln
        else:
            return pred_scores, pred_labeln
        # endif
    # endif

    # Get previous values
    new_labels_to_names = calibrate_predictions_to_mapped_labels.new_labels_to_names
    new_names_to_labels = calibrate_predictions_to_mapped_labels.new_names_to_labels

    # Map labels, scores etc
    pred_scores, new_labels_to_names = map_labels_on_predictions(pred_scores, old_labels_to_names, mapping_dict, new_labels_to_names)
    new_names_to_labels = {v:k for k,v in new_labels_to_names.items()} if new_names_to_labels is None else new_names_to_labels

    #print('old_labels_to_names = ', old_labels_to_names)
    #print('new_names_to_labels = ',new_names_to_labels)
    def map_old_lnum_to_new_lnum(_old_lnum, _old_labels_to_names, _new_names_to_labels, _mapping_dict):
        if _old_lnum in _old_labels_to_names:
            old_lname = _old_labels_to_names[_old_lnum]
            old_lname = _mapping_dict[old_lname] if old_lname in _mapping_dict else old_lname
            return _new_names_to_labels[old_lname]
        # endif
        return _old_lnum
    # enddef

    if act_labeln is not None:
        act_labeln   = map_old_lnum_to_new_lnum(act_labeln, old_labels_to_names, new_names_to_labels, mapping_dict)
    # endif
    pred_labeln  = map_old_lnum_to_new_lnum(pred_labeln, old_labels_to_names, new_names_to_labels, mapping_dict)

    # Save intermittent variables
    calibrate_predictions_to_mapped_labels.new_labels_to_names = new_labels_to_names
    calibrate_predictions_to_mapped_labels.new_names_to_labels = new_names_to_labels

    if act_labeln is not None:
        return pred_scores, pred_labeln, act_labeln
    else:
        return pred_scores, pred_labeln
    # endif
# enddef

def map_label_numbers(label_no_list, label_map, ns_lname=None):
    ns_lname = NON_SPECIFIC_CLASS_NAME if ns_lname is None else ns_lname
    mapped_label_list = []

    for indx in range(len(label_no_list)):
        # Predicted labels
        label_no = label_no_list[indx]
        lname    = label_map[label_no] if label_no in label_map else ns_lname
        mapped_label_list.append(lname)
    # endfor
    return mapped_label_list
# enddef

# Add Non specific entry in label_map if not already present
def add_nonspecific_in_label_map(label_map, key_is_name=False):
    lnames_list = label_map.keys() if key_is_name else label_map.values()
    if NON_SPECIFIC_CLASS_NAME not in lnames_list:
        if key_is_name:
            label_map[NON_SPECIFIC_CLASS_NAME] = len(label_map)
        else:
            label_map[len(label_map)] = NON_SPECIFIC_CLASS_NAME
        # endif
    # endif
    return label_map
# enddef

######### Projection functions

def get_pca(featsv, n_components=2):
    pca = sklearnPCA(n_components=n_components)
    transformed = pca.fit_transform(featsv)
    return transformed
# enddef

def get_tsne(featsv, label_list=None, n_components=2, perplexity=12,
        early_exaggeration=12, n_iter=300):
    tsne = TSNE(n_components=n_components, perplexity=perplexity,
            early_exaggeration=early_exaggeration, n_iter=n_iter)
    transformed = tsne.fit_transform(featsv, label_list)
    return transformed
# enddef

############ Server apis

def get_server_response(server_endpoint, image_file):
    image_b64 = base64.b64encode(open(image_file, 'rb').read())
    ret_v     = requests.post(server_endpoint, json={'image' : image_b64.decode('utf-8')})
    ret_v     = ret_v.json()
    if ret_v['status'] == 200:
        return ret_v
    # endif
    raise ValueError('Invalid response {} from server for image {}'.format(ret_v['status'], image_file))

def get_prediction(server_endpoint, image_file):
    return get_server_response(server_endpoint, image_file)['response']['prediction']
# enddef

######################## Terminal functions
def load_envbash(envbash):
    def __read_envbash(envbash):
        # Check if file exists
        try:
            with open(envbash):
                pass
            # endwith
        except FileNotFoundError:
            raise ValueError('{} not found'.format(envbash))
        # endtry

        # construct an inline script which sources env.bash then prints the
        # resulting environment so it can be eval'd back into this process.
        inline = '''
            set -a
            source {} >/dev/null
            {} -c "import os; print(repr(dict(os.environ)))"
        '''.format(pipes.quote(envbash), pipes.quote(sys.executable))

        # run the inline script with bash -c, capturing stdout. if there is any
        # error output from env.bash, it will pass through to stderr.
        # exit status is ignored.
        with open(os.devnull) as null:
            output, _ = subprocess.Popen(
                ['bash', '-c', inline],
                stdin=null, stdout=subprocess.PIPE, stderr=None,
                bufsize=-1, close_fds=True,
            ).communicate()

        # the only stdout from the inline script should be
        # print(repr(dict(os.environ))) so there should be no synt
        # eval'ing this. however there will be no output to eval if the sourced
        # env.bash exited early, and that indicates script failure.
        if not output:
            raise ValueError('{} exited early'.format(envbash))
        # endif

        # the eval'd output should return a dictionary.
        nenv = eval(output)

        return nenv
    # enddef

    # Load only new values
    env_orig      = os.environ
    env_loaded    = __read_envbash(envbash)
    env_new_keys  = set(env_loaded) - set(env_orig)
    env_new       = {}
    for k in env_new_keys:
        env_new[k] = env_loaded[k]
    # endfor
    return env_new
# enddef

#################################################################
# Dummy class (used as a proxy for FLAGS for various functions)
class P(object):
    ''' This is a dummy class used to emulate absl FLAGS system'''
    def __init__(self, *args, **kwargs):
        for item_t in args:
            for key_t in item_t.__dict__:
                self.__dict__[key_t] = item_t.__dict__[key_t]
            # endfor
        # endfor
        for key_t in kwargs:
            self.__dict__[key_t] = kwargs[key_t]
        # endfor
    # enddef
# endclass

