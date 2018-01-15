from librosa import logamplitude
from librosa.feature import melspectrogram
from multiprocessing import Pool
import numpy as np
import os
import pandas as pd
from scipy.io import wavfile
from torch.utils.data import Dataset
from tqdm import *

root_dir = ''
train_audio = 'train/audio'
test_audio = 'test/audio'
df = {}
procs = 7

# This can also be set to true to generate new train and validation sets
calc_mels = False
calc_val = False

# create list of classes
classes = os.listdir(os.path.join(root_dir, train_audio))
classes.remove('_background_noise_')
#         self.classes.append('silence')

# create dataframe of training meta data
tr_f_l = []
for c in classes:
    files_cl = os.listdir(os.path.join(root_dir, train_audio, c))
    for f in files_cl:
        speaker, _, utter_no = f[:-4].split('_')
        path_file = os.path.join(root_dir, train_audio, c, f)
        tr_f_l.append((speaker, int(utter_no), 1, c, path_file))
columns = 'speaker utter_no label class_name file'.split()
df['train_all'] = pd.DataFrame(tr_f_l, columns=columns)
cl_map = lambda x: classes.index(x['class_name'])
df['train_all'].label = df['train_all'].apply(cl_map, axis=1)
df['train_all']['idx'] = df['train_all'].index

# split training data into training and validation, done on a speaker basis to
# avoid having samples from the same speaker in different sets
val_pct=0.20
speakers = df['train_all'].speaker.unique()

if calc_val:
    idx = np.random.randint(len(speakers), size = int((len(speakers) * val_pct)))
    np.save('idx.npy', idx)
else:
    idx = np.load('idx.npy')
              
val = speakers[idx]
df['val'] = df['train_all'][df['train_all'].speaker.isin(val)]
df['train'] = df['train_all'][~df['train_all'].speaker.isin(val)]

# create dataframe of test file names
test_files  = os.listdir(os.path.join(root_dir, test_audio))
test_files = list(map(lambda x: os.path.join(test_audio, x), test_files))

df['test'] = pd.DataFrame(test_files)
df['test'].columns = ['file']
df['test']['label'] = -1
df['test'] = df['test'][['label', 'file']]

ms = {}
if not calc_mels:
    for split in ['train_all', 'test']:
        ms[split] = np.load('mels_{}.npy'.format(split))
        if split == 'train_all':
            mean = ms[split].mean((0, 1), dtype=np.float64)
            std = ms[split].std((0, 1), dtype=np.float64)
        ms[split] = np.divide(np.subtract(ms[split], mean, dtype=np.float16), std, dtype=np.float16)

# specify mel spectrogram function
n_fft=500
hop_length=161
n_mels=100
fmax=8000

def mel_spec(samples, rate):
    S = melspectrogram(samples, rate, power=1, n_fft=n_fft,
                       hop_length=hop_length, n_mels=n_mels, fmax=8000)
    log_S = logamplitude(S, ref_power=np.max)
    return log_S, rate

# create mel spectrograms for each set, adding zeros at the end of any
# wav file less than 16k samples

class AudioData(Dataset):
    def __init__(self, split, transform=None):
        self.split = split
        self.transform = transform
        self.labels = df[self.split].label
        self.idxs = df[self.split].index
        self.classes = classes

    def __len__(self):
        return len(df[self.split])

    def __getitem__(self, idx):
        if split == 'test':
            data = ms['test'][idx]
        else:
            data = ms['train_all'][self.idxs[idx]]
            
        label = self.labels[idx]
        
        if self.transform:
            data = self.transform(data)

        return data, label, idx

if __name__ == '__main__':
    def get_mels(i, f):
        samp_arr = np.zeros(16000, dtype=np.float16)
        rate, samples = wavfile.read(f)
        samp_arr[:len(samples)] = samples
        m_samp, _ = mel_spec(samp_arr, rate)

        return i, m_samp

    if calc_mels:
        for split in ['train_all', 'test']:


            N = len(df[split])
            pbar = tqdm(total=N)
            mels = np.zeros((N, 100, 100), dtype=np.float16)

            def wrapper(i, f):
                return get_mels(i, f)

            def update(result):
                i, m_samp = result
                # note: input comes from async `wrapMyFunc`
                mels[i] = m_samp  # put answer into correct index of result list
                pbar.update()

            pool = Pool(procs)
            for i, f in enumerate(df[split].file):
                pool.apply_async(wrapper, args=(i,f), callback=update)
            pool.close()
            pool.join()
            np.save('mels_{}.npy'.format(split), mels)
            pbar.close()