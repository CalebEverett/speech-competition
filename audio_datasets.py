import numpy as np
import os
import pandas as pd
from scipy.io import wavfile
from torch.utils.data import Dataset

root_dir = ''
train_audio = 'train/audio'
test_audio = 'test/audio'
df = {}

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
idx = np.random.randint(len(speakers), size = int((len(speakers) * val_pct)))
val = speakers[idx]
df['val'] = df['train_all'][df['train_all'].speaker.isin(val)]
df['train'] = df['train_all'][~df['train_all'].speaker.isin(val)]

# create dataframe of test file names
df['test'] = pd.DataFrame(os.listdir(os.path.join(root_dir, test_audio)))
df['test'].columns = ['file']
df['test']['label'] = ''
df['test']['idx'] = df['test'].index

class AudioData(Dataset):
    def __init__(self, split, transform=None):
        self.split = split
        self.transform = transform
        self.keys = list(df[split].keys())
        self.data = df[split].to_records(False)

    def __len__(self):
        return len(ds[split])

    def __getitem__(self, idx):
        file = self.data[idx][self.keys.index('file')]
        label = self.data[idx][self.keys.index('label')]
        idx_train_all = self.data[idx][self.keys.index('idx')]
                              
        rate, samples = wavfile.read(file)    

        if self.transform:
            samples = self.transform(samples)

        return (samples, rate), label, idx_train_all
                               