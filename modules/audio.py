import matplotlib
matplotlib.use('Agg')

import pydub
from   pydub import AudioSegment
import librosa
import librosa.display
import matplotlib.pyplot as plt
import modules.tf2_models as tf2_models
from   tensorflow.keras import optimizers
import tensorflow as tf
import numpy as np
import shutil
from   modules.utils import *

###############################################################
# Utils
@static('data_path', None)
def tmp_data_path(data_dir='./data', remove=False):
    if remove:
        data_dir = tmp_data_path.data_path if tmp_data_path.data_path else data_dir
        shutil.rmtree(data_dir, ignore_errors=True)
        return
    else:
        if tmp_data_path.data_path is None:
            tmp_data_path.data_path = data_dir
            mkdir(data_dir)
            return data_dir
        # endif
    # endif

    return tmp_data_path.data_path
# enddef

###############################################################
# Convert audio
def convert_audio(src):
    audio = AudioSegment.from_file(src)
    tmp_f = os.path.join(tmp_data_path(), '{}.wav'.format(u4()))
    audio.export(tmp_f, format='wav')
    return tmp_f
# enddef

def convert_audio_from_byte_stream(byte_stream):
    # Save audio
    audio_tmp = os.path.join(tmp_data_path(), '{}'.format(u4()))
    with open(audio_tmp, 'wb') as f_w:
        f_w.write(byte_stream)
    # endwith

    # Convert audio
    new_file = convert_audio(audio_tmp)
    os.remove(audio_tmp)
    return new_file
# enddef

################################################################
# Preprocessing
def load_and_preprocess_spectrogram(spec_file, target_size=(224, 224)):
    # Load and resize
    im_spec = np.asarray(PIL.Image.open(spec_file).convert('RGB').resize(target_size, PIL.Image.NEAREST))
    # Normalize
    im_spec = (im_spec - np.mean(im_spec, keepdims=True))/(np.std(im_spec, keepdims=True) + 1e-6)

    return im_spec
# enddef

#################################################################
# Feature extractors
def e_mfcc(audio, sr=22050, n_mfcc=40):
    return librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
# enddef
def e_mel_spectrogram(audio, sr=22050, n_fft=2048, hop_length=512, ref=40):
    assert ref in [40, 80], 'ref={} should be 40 or 80 only.'.format(ref)
    return (librosa.power_to_db(librosa.feature.melspectrogram(audio, sr=sr, n_fft=n_fft, hop_length=hop_length), ref=np.max) + ref)/ref
# endif
def e_mel_spectrogram2(audio, sr=22050, n_fft=2048, hop_length=512):
    return librosa.power_to_db(librosa.feature.melspectrogram(audio, sr=sr, n_fft=n_fft, hop_length=hop_length), ref=np.max)
# enddef

##################################################################
# Read files
def read_and_zero_pad(file_name, max_slen=87, sr=22050, verbose=True):
    data, sr = librosa.load(file_name, sr=sr, mono=True, duration=max_slen)
    obuf_len = len(data)
    nbuf_len = max_slen*sr
    try:
        data     = librosa.util.pad_center(data, max_slen*sr)
    except:
        print('WARNING:: Unable to read {}'.format(file_name))
        sys.exit(-1)
    # endtry

    if (obuf_len > nbuf_len) and verbose:
        print('WARNING:: Truncating {} of length {} to {}'.format(file_name, obuf_len, nbuf_len))
    # endif

    return data, sr
# enddef

def extract_vector(audio_file, max_slen=5, sr=22050):
    feat_fn  = e_mel_spectrogram2
    data, sr = read_and_zero_pad(audio_file, max_slen=max_slen, sr=sr, verbose=False)
    feat_t   = feat_fn(data, sr=sr)

    data_dir = tmp_data_path()
    mkdir(data_dir)
    tmp_file = os.path.join(data_dir, u4() + '.png')

    # Save spectrogram
    librosa.display.specshow(feat_t, sr=sr, hop_length=512)
    plt.savefig(tmp_file)
    plt.close()

    # Load spectrogram
    spec_data = load_and_preprocess_spectrogram(tmp_file)
    os.remove(tmp_file)

    return spec_data
# enddef

####################################################
# Load model
def compile_model(model, loss='categorical_crossentropy', metrics=['accuracy'], optimizer=optimizers.Adam(lr=0.00001)):
    model.compile(loss=loss, metrics=metrics, optimizer=optimizer)
    return model
# enddef

def load_osf_sick_nonsick_model(model_ckpt='checkpoints/vgg16_07052020_1/mymodel2_20.h5', model_name='vgg16', num_classes=2):
    model = tf2_models.get_model_fn_map()[model_name](num_classes)
    model = compile_model(model) 
    model.load_weights(model_ckpt)
    return model
# enddef

@static('model', None)
def get_audio_model(model_ckpt='checkpoints/vgg16_07052020_1/mymodel2_20.h5', model_name='vgg16', num_classes=2):
    if get_audio_model.model is None:
        get_audio_model.model = load_osf_sick_nonsick_model(model_ckpt, model_name, num_classes)
    # endif

    return get_audio_model.model
# enddef

#####################################################
# Predict sick vs not sick
def predict_model(model, audio_file, convert_fn=convert_audio):
    taud_file  = convert_fn(audio_file)
    audio_feat = extract_vector(taud_file)
    os.remove(taud_file)
    model_out  = np.squeeze(model.predict(tf.expand_dims(audio_feat, axis=0)), axis=0).tolist()

    return {'not_sick' : model_out[0], 'sick' : model_out[1]}
# enddef

def predict_audio_from_file(audio_file):
    return predict_model(get_audio_model(), audio_file, convert_audio)['sick']
# enddef

def predict_audio_from_byte_stream(byte_stream):
    return predict_model(get_audio_model(), byte_stream, convert_audio_from_byte_stream)['sick']
# enddef

def predict_audio_all_from_byte_stream(byte_stream):
    return predict_model(get_audio_model(), byte_stream, convert_audio_from_byte_stream)
# enddef
