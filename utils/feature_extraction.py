# feature_extracting
import librosa
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
# import time
import torchaudio
from speechbrain.pretrained import EncoderClassifier


def gen_noise_data(data):
    Nsamples = len(data)
    noiseSigma = 0.01
    noiseAmplitude = 0.5
    noise = noiseAmplitude * np.random.normal(0, noiseSigma, Nsamples)

    noised_data = noise + data
    return noised_data, noise




def get_audio_features(audio_path, sampling_rate, classifier, add_noise=False):
    X, sample_rate = librosa.load(audio_path, res_type='kaiser_fast', duration=2.5, sr=sampling_rate * 2, offset=0.5)
    signal, fs = torchaudio.load(audio_path)

    if add_noise:
        X, _ = gen_noise_data(X)
        signal, _ = gen_noise_data(signal)

    sample_rate = np.array(sample_rate)
    y_harmonic, y_percussive = librosa.effects.hpss(X)
    pitches, magnitudes = librosa.core.pitch.piptrack(y=X, sr=sample_rate)

    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13), axis=1)

    pitches = np.trim_zeros(np.mean(pitches, axis=1))[:20]

    magnitudes = np.trim_zeros(np.mean(magnitudes, axis=1))[:20]

    C = np.mean(librosa.feature.chroma_cqt(y=y_harmonic, sr=sampling_rate), axis=1)

    # xvector
    xvector = torch.squeeze(classifier.encode_batch(signal))
    if len(xvector.shape) > 1:
        # there is rare files that generate duplicate tensors (2,512) the code below made to overcome dimension problem
        xvector = xvector[0]
        print(audio_path)
    xvector = np.array(xvector)

    return [mfccs, pitches, magnitudes, C, xvector]


def get_features_dataframe(dataframe, sampling_rate, add_noise=False):
    labels = pd.DataFrame(dataframe['label'])
    # spitchbrain classifier
    classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-xvect-voxceleb",
                                                savedir="pretrained_models/spkrec-xvect-voxceleb")
    #
    features = pd.DataFrame(columns=['mfcc', 'pitches', 'magnitudes', 'C', 'xvector'])
    for index, audio_path in tqdm(enumerate(dataframe['path'])):
        features.loc[index] = get_audio_features(audio_path, sampling_rate, classifier, add_noise)

    mfcc = features.mfcc.apply(pd.Series)
    pit = features.pitches.apply(pd.Series)
    mag = features.magnitudes.apply(pd.Series)
    C = features.C.apply(pd.Series)
    xvector = features.xvector.apply(pd.Series)

    combined_features = pd.concat([mfcc, pit, mag, C, xvector], axis=1, ignore_index=True)

    return combined_features, labels



