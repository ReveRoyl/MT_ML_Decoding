#!/usr/bin/env python
# coding: utf-8

# In[2]:


# 引入python库
import mne
import os
from mne.datasets import sample
import matplotlib.pyplot as plt

# sample的存放地址
data_path = sample.data_path()
# 该fif文件存放地址
#fname = 'E:\Proj\Previous data\ds003682\sub-001\ses-01\meg\sub-001_ses-01_task-AversiveLearningReplay_run-01_meg.fif' #corrupted
#fname = 'E:\Proj\Previous data\sample\MEG\sample\sample_audvis_raw.fif' # testsample data
fname = 'E:\Proj\Test\sample\MEG\sample\sub-001_localiser_sub-001_ses-01_task-AversiveLearningReplay_run-localiser_proc_ICA-epo.fif.gz'

epochs = mne.read_epochs(fname)
#raw = mne.io.read_raw(fname)
#events = mne.find_events(epochs)
#events = mne.find_events(raw, stim_channel=None)

"""
案例：
获取10-20秒内的良好的MEG数据

# 根据type来选择 那些良好的MEG信号(良好的MEG信号，通过设置exclude="bads") channel,
结果为 channels所对应的的索引
"""
#epochs = mne.Epochs(raw, events, tmin=-0.2, tmax=0.5)
#events = mne.find_events(epochs)
#print(epochs.event_id)

'''picks = mne.pick_types(raw.info, meg=True, ref_meg=False, exclude='bads')
t_idx = raw.time_as_index([10., 20.])
data, times = raw[picks, t_idx[0]:t_idx[1]]
plt.plot(times,data.T)
plt.title("Sample channels")
plt.show()'''

picks = mne.pick_types(epochs.info, meg=True, ref_meg=False, exclude='bads')

#t_idx = epochs.time_as_index([10., 20.])
epochs.plot(block=True)

epochs.plot_drop_log()

#SSP矢量图
#epochs.plot_projs_topomap()
plt.show()

#'''event_id = {'1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '32': 32}
#color = {1: 'green', 2: 'yellow', 3: 'red', 4: 'c', 5: 'black', 32: 'blue'}
#mne.viz.plot_events(events, epochs.info['sfreq'], raw.first_samp, color=color,
#event_id=event_id)
#plt.show()'''


# In[4]:


epochs = mne.read_epochs(fname)

evoked = epochs.average()
evoked.plot_topomap()

plt.show()


# In[2]:


availabe_event = [1, 2, 3, 4, 5, 32]
for i in availabe_event:
    evoked_i = epochs[i].average(picks=picks)
    epochs_i = epochs[i]
    evoked_i.plot(time_unit='s')
    plt.show()







