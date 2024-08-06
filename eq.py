# functions for viewing, sorting, and triming the equivital sensor data files after they have been processed to be aligned to external recording timeline with synch cues. 

# put this in an early cell of any notebook useing these functions, uncommented. With starting %
# %load_ext autoreload
# %autoreload 1
# %aimport eq

import sys
import os
import time
import datetime as dt
import math
import numpy as np 
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.signal import butter,filtfilt
from scipy import interpolate
from scipy.interpolate import interp1d


def data_dets(eq_file_loc,sep): #rec_start = V['DateTime'].iloc[0]
    # adjusted for the information available after time aligning raw measurements
    if not sep:
        sep = '\\'
    # this file pulls recording details from the file name and from inside file to aggregate all metadata
    filings = eq_file_loc.split(sep) 
    eq_data_loc = sep.join(filings[:-1])
    file_name = filings[-1] # concert_segment_partID_filetype.csv
    f = file_name.split('_')
#     if len(f[0])==2:
#         Signal = f[3][2:6]
#         PartID = f[2]#filings[-2]
#         EventName = f[0]
#         SegmentName = f[1]
#     else:
#         Signal = f[3][2:6]
#         PartID = f[0]#filings[-2]
#         EventName = f[1]
#         SegmentName = f[2]

    k = 2
    if f[0].isnumeric():
        fileDate = f[0]
        Concert = f[1]
    else:
        Concert = f[0]
        if f[1].isnumeric():
            fileDate = f[1]
        else:
            fileDate = np.nan
            k = 1
            
    for d in f[k:]:
#         print(d)
        if d[0].isalpha():
            if d.startswith('EQ'):
                Signal =  d[:-4]
            else: # PartID
                if d.startswith('MS'):
                    Signal = d[:-4]
                else:
                    if len(d)>4:
                        if d[4].isdigit():
                            PartID = d
                        else:
                            segtag = d
                    else:
                        segtag = d

    fileSize = os.path.getsize(eq_file_loc)
    
    eq_data_loc = sep.join(filings[:-1])
    M = matched_eqfiles(eq_file_loc,eq_data_loc,sep)
    mfiles = {'DATA':'','RESP':'','CIBI':'','BACC':'','2ECG':''}
    for m in M:
        if 'DATA' in m:
            mfiles['DATA'] = m
        if 'RESP' in m:
            mfiles['RESP'] = m
        if 'IBI' in m:
            mfiles['CIBI'] = m
        if 'ACC' in m:
            mfiles['BACC'] = m       
        if 'ECG' in m:
            mfiles['2ECG'] = m 
            
#     print(eq_file_loc)    
    V = pd.read_csv(eq_file_loc,skipinitialspace=True)
    V['rec_dTime'] = pd.to_datetime(V['rec_dTime'])
    rec_start = V['rec_dTime'].iloc[0]
    rec_end = V['rec_dTime'].iloc[-1]
    if 'c_sTimes' in V.columns:
        ev_start = V['c_sTime'].iloc[0]
        ev_end = V['c_sTime'].iloc[-1]
    else:
        ev_start = 0
        ev_end = 0
    rec_dur=(rec_end-rec_start).total_seconds()
    Batt_start = V['BATTERY(mV)'].iloc[0]
    Batt_end = V['BATTERY(mV)'].iloc[-1]
    Batt_spend=(Batt_end-Batt_start)     

    a = V.loc[:,['SENSOR ID', 'SUBJECT ID', 'SUBJECT AGE', 'HR(BPM)',
       'HRC(%)', 'BELT OFF', 'LEAD OFF', 'MOTION', 'BODY POSITION']].mode().loc[0]
    DevID = V.loc[:,'SENSOR ID'].unique()
    DevNames = V.loc[:,'SUBJECT ID'].unique()

    File_dets={'Signal':Signal, #f[-2].split('_')[-1],
       'PartID':PartID,
       'ID':DevID, 
       'Date':str(rec_start.date()),
       'Session':segtag,
       'FileName':file_name,
       'FileType':'csv',
       'FileSize': fileSize,
       'RecStart':rec_start,
       'RecEnd':rec_end,
       'EventStart':ev_start,
       'EventEnd':ev_end,
       'Duration':rec_dur,
       'BatteryStart':Batt_start,
       'BatteryEnd':Batt_end,
       'BatteryChange(mV)':Batt_spend,
       'FullLoc':eq_file_loc,
        'DATAloc':mfiles['DATA'],
        'BACCloc': mfiles['BACC'],
        'RESPloc':mfiles['RESP'],
        'CIBIloc': mfiles['CIBI'],
        '2ECGloc':mfiles['2ECG'],
       'SubjectNames': DevNames}
    File_dets.update(a) # dic0.update(dic1)
    return File_dets
# mfiles = {'DATA':'','RESP':'','CIBI':'','BACC':'','2ECG':''}
def matched_eqfiles(eq_file_loc,data_path,sep):
    if not sep:
        sep = '\\'
    # from the location of a good file and the location of other files, retrieve the location of all matching files
    dfile = min_dets(eq_file_loc,sep)
    
    # retrieve the files in that path that match 
    file_locs = []
    for root, dirs, files in os.walk(data_path):
        for file in files:
            if(file.lower().endswith(".csv")):
                if(not file.lower().startswith("._")):
                    file_locs.append(os.path.join(root,file))
    k=[]
    for file in file_locs:
        if not file.lower().endswith('recordings.csv'):
#             print(file)
            File_dets=min_dets(file,sep)
            k.append(File_dets)
    df_files=pd.DataFrame(data=k)
#     print(df_files)

    match_fields = ['PartID','Segment','Session']

    matched_files = df_files.loc[df_files['PartID'] == dfile['PartID']]
    for mf in match_fields[1:]:
        matched_files = matched_files.loc[matched_files[mf] == dfile[mf]]

    return list(matched_files['FullLoc'])
    
    
def eq_recordings(projectpath,DATAtag,sep):
    file_locs = []
    for root, dirs, files in os.walk(projectpath):
        for file in files:
            if(file.lower().endswith(DATAtag.lower()+".csv")):
                if(not file.lower().startswith("._")):
                    file_locs.append(os.path.join(root,file))
    if len(file_locs)>0:
        k=[]           
        for f in file_locs:
            File_dets=data_dets(f,sep)
            if File_dets:
                k.append(File_dets)
        df_datafiles=pd.DataFrame(data=k)#
        df_datafiles=df_datafiles.sort_values(by='RecStart').reset_index(drop=True)
#         df_datafiles.to_csv(projectpath + projecttag + '_Qiosk_recordings.csv')
        return df_datafiles
    else:
        print('Path is empty of DATA files.')
        return []
    

def min_dets(eq_file_loc,sep):
    file_name = eq_file_loc.split(sep)[-1]
    f = file_name.split('_')
    if len(f)>2: 
        if len(f[0])==2:
            Signal = f[3][2:6]
            PartID = f[2]#filings[-2]
            EventName = f[0]
            SegmentName = f[1]
        else:
            Signal = f[3][2:6]
            PartID = f[0]#filings[-2]
            EventName = f[1]
            SegmentName = f[2]

        fileSize = os.path.getsize(eq_file_loc)

        File_dets={'Signal':Signal, #f[-2].split('_')[-1],
       'PartID':PartID,
       'Session':EventName,
       'Segment':SegmentName,
       'FileName':file_name,
       'FileType':'csv',
       'FileSize': fileSize,
       'FullLoc':eq_file_loc}
        return File_dets
    else:
        return []
    
# PARTS AND PLAYING    
    
def cascade_fill(Entries):
    # fill a pandas dataframes' NaNs with the previous non-nan value in the colum
    Full_entries = pd.DataFrame(index = Entries.index)
    cols = Entries.columns
    for c in cols:
        if Entries[c].isna().sum()>1:
            D = Entries.loc[Entries[c].notna(),['mm',c]].copy()
            sig_t = D.index
            sig_v = D.loc[:,c].values

            if D.index[0]>Pentries.index[0]:
                sig_t = pd.concat([pd.Series([Pentries.index[0]]), pd.Series(D.index)])
                sig_v = pd.concat([pd.Series([0]), pd.Series(sig_v)])

            f = interpolate.interp1d(sig_t, sig_v,kind='previous')
            Full_entries.loc[:,c] = f(Entries.index)
    return Full_entries

def playing_TS(Entries,time_s):
    # fill a pandas dataframes' NaNs with the previous non-nan value in the colum
    Full_entries = pd.DataFrame(index = time_s)
    cols = Entries.columns
    for c in cols:
        if Entries[c].isna().sum()>1:
            D = Entries.loc[Entries[c].notna(),['mm',c]].copy()
            sig_t = D.index
            sig_v = D.loc[:,c].values

            if time_s[0]<Pentries.index[0]:
                sig_t = pd.concat([pd.Series([time_s[0]]), pd.Series(D.index)])
                sig_v = pd.concat([pd.Series([0]), pd.Series(sig_v)])

            f = interpolate.interp1d(sig_t, sig_v,kind='previous',fill_value='extrapolate')
            Full_entries.loc[:,c] = f(time_s)
    return Full_entries

def Part_maps(Entries):
    parts_on = {}
    cols = Entries.columns
    for c in cols:
        if Entries[c].isna().sum()>1:
            Entries.loc[Entries[c].notna(),['mm',c]]
            V = Entries.loc[Entries[c].notna(),['mm',c]]
            parts_on[c]=V.rename(columns = {c:'Playing'})
            D = pd.DataFrame(index = parts_on[c].index )
            for i in D.index:
                if parts_on[c].loc[i,'Playing'] == 1.0:
                    D[i] = D.index-i
            D[D>=0] = np.nan
            D['Pre'] = D.max(axis=1)
            D.loc[parts_on[c].loc[:,'Playing'] == 1.0,'Pre'] = 0.0
            parts_on[c]['Pre'] = D['Pre']
            parts_on[c]['Post'] = parts_on[c]['Pre'].diff()
            parts_on[c].loc[parts_on[c]['Post']<=0,'Post'] = 0.0
            parts_on[c][parts_on[c].isna()] = 0.0
    return parts_on

def ToEntries_TS(Entries,time_s):
    Entry_Times = pd.DataFrame(index = time_s)
    Entry_Times['ToEntries'] = time_s
    Entry_Times['ToEnd'] = time_s

    Entry_times = list(Entries.loc[Entries['Playing']==1,:].index)
    End_times = list(Entries.loc[Entries['Playing']==0,:].index)
    if End_times[0]<Entry_times[0]:
        End_times = End_times[1:]
    t_0 = time_s[0]

    for i in range(len(Entry_times)-1):
        Entry_Times.loc[t_0:End_times[i],'ToEntries'] = Entry_Times.loc[t_0:End_times[i],'ToEntries'] - Entry_times[i]
        t_0 = End_times[i]
    Entry_Times.loc[t_0:,'ToEntries'] = Entry_Times.loc[t_0:,'ToEntries'] - Entry_times[len(Entry_times)-1]

    t_0 = time_s[0]    
    for i in range(len(End_times)-2):
        Entry_Times.loc[t_0:Entry_times[i+1],'ToEnd'] = Entry_Times.loc[t_0:Entry_times[i+1],'ToEnd'] - End_times[i]
        t_0 = Entry_times[i+1]
    Entry_Times.loc[t_0:,'ToEnd'] = Entry_Times.loc[t_0:,'ToEnd'] - End_times[len(End_times)-1] 
    
    return Entry_Times

def play_shade(ax,Pentries,concert,part):
    # shading plots
    Pentries['Time_s'] = Pentries[concert]
    Pentries = Pentries.set_index('Time_s',drop = True)
    Part_entries = Part_maps(Pentries) 
    if part != 'Not_Playing':
        Entries = Part_entries[part].copy() 
        es = Entries.index[Entries['Playing']==1.0]
        ee = Entries.index[Entries['Playing']==0.0]
        if len(ee)>len(es): ee = ee[1:]  
        for i in range(len(es)):
            ax.axvspan(es[i],ee[i], facecolor='c',alpha = 0.2)
            
    return