# functions for using tapping or clapping cue embedded accelerometer measurements to align sensor recordings to concert time.


# put this in an early cell of any notebook useing these functions, uncommented. With starting %
# %load_ext autoreload
# %autoreload 1
# %aimport al

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
# import heartpy as hp

# https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.astype.html

from scipy.signal import butter, filtfilt, argrelextrema
from scipy import interpolate
from scipy.interpolate import interp1d

def find_nearest_idx(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def cue_template_make(peak_times,sf, t_range):
    # peak times is list of time points for onsets to clapping or tapping sequence, in seconds
    # sf is sample frequency in hz
    # buffer is duration of zeros before and after the peak times in generated template, in seconds
    
    peaks = np.array(peak_times)
    c_start = t_range[0]+peaks[0]
    c_end = t_range[1]+peaks[0]
    
    cue_sTime = np.linspace(t_range[0],t_range[1],sf*(t_range[1]-t_range[0]),endpoint=False)+peaks[0]

    cue = pd.DataFrame()
    cue['sTime'] = cue_sTime
    cue['peaks'] = 0
    cue['taps'] = 0
    cue['claps'] = 0
    cue['slowtable'] = 0
    
    for pk in peak_times:
        cue.loc[find_nearest_idx(cue['sTime'],pk),'peaks'] = 1
        
    roll_par = int(0.2*sf)
    sum_par = int(0.05*sf)
    ewm_par = int(0.1*sf)
    cue['slowtable'] =2*cue['peaks'].ewm(span = ewm_par).mean()+ 0.6*cue['peaks'].rolling(roll_par, win_type='gaussian',center=True).sum(std=2)        

    roll_par = int(0.05*sf)
    sum_par = int(0.02*sf)
    ewm_par = int(0.04*sf)
    cue['claps'] =2*cue['peaks'].ewm(span = ewm_par).mean()+ 0.6*cue['peaks'].rolling(roll_par, win_type='gaussian',center=True).sum(std=2)
    
    roll_par = int(0.02*sf)
    sum_par = int(0.02*sf)
    ewm_par = int(0.02*sf)
    cue['taps'] =1.5*cue['peaks'].ewm(span = ewm_par).mean()+ 0.5*cue['peaks'].rolling(roll_par, win_type='gaussian',center=True).sum(std=2)
    
    cue[cue.isna()] = 0
    return cue

def tap_cue_align(cue,sig_ex,sig_ID):
    # a function to take a segment of signal and the tapping cue to determing
    # the best shift value would allow alignment of signal to cue
    signal = sig_ex.copy()
        # make the signal excerpt corr compatible. Inclusing cutting the extreme peaks
    signal[signal.isna()] = 0
    M = signal.quantile(0.998)
    signal = signal/M
    signal[signal>1] = 1
    
    # cue must be sampled at the same steady rate as the signal exerpt
    sampleshift_s = cue.index.to_series().diff().median()
    length = np.min([len(signal),len(cue)])
    
    if signal.diff().abs().sum()<0.1: # signal is flat
        shifts.append(np.nan)
        print('sig_ex is too flat.')
        return
    else:
        fig = plt.figure(figsize=(15,6))
        ax1 = plt.subplot(311)
        signal.plot(label=sig_ID,ax=ax1,)
        cue.plot.line(y='cue',ax=ax1)
        ax1.set_title(sig_ID + ' synch alignment')
        ax1.legend()
        #plt.xlim(cue_range)
        
        ax2 = plt.subplot(312)
        CCC = ax2.xcorr(cue['cue'].iloc[:length], signal.iloc[:length], usevlines=True, maxlags=length-1, normed=True, lw=3)
        ax2.grid(True)
        ax2.set_xticklabels('')
        signal.index = signal.index + sampleshift_s*CCC[0][np.argmax(CCC[1])]
        
        ax1 = plt.subplot(313)
        signal.plot(label=sig_ID,ax=ax1,)
        cue.plot.line(y='cue',ax=ax1)
        #plt.xlim(cue_range)
        ax1.grid(True)
        ax1.set_title('shift '+ str(sampleshift_s*CCC[0][np.argmax(CCC[1])])+ ' s')
        #plt.saveas('')
        plt.show()

    shift_stats = {"s_corr0": CCC[1][CCC[0]==0][0], # alignment quality without adjustment,
                   "s_corr_offset": np.amax(CCC[1]),
                   "s_offset_samples": CCC[0][np.argmax(CCC[1])], # shifts
                   "s_offset_time": sampleshift_s*CCC[0][np.argmax(CCC[1])],
                   "Length_xcorr_samples": len(CCC[0]),
                   "Length_xcorr_time": len(CCC[0])*sampleshift_s,
                   "devID": sig_ID,
                   "auto_offset_time":sampleshift_s*CCC[0][np.argmax(CCC[1])],
                   "Full_CCC": CCC

    }
    return shift_stats
    
def min_align(ACC,cue,prelim_synch_time,max_offs):
    # sf infered from cue
    sampleshift_s = cue['sTime'].diff().median()
    sf = np.round(1/sampleshift_s) 
    t_range = [cue['sTime'].iloc[0],cue['sTime'].iloc[-1]]
    c_type = cue.columns[1]
    
    xrange = [pd.to_timedelta(t_range[0],unit = 's') + prelim_synch_time,pd.to_timedelta(t_range[1],unit = 's') + prelim_synch_time]
    sig_sTime = cue['sTime'].values #np.linspace(t_range[0],t_range[1],sf*(t_range[1]-t_range[0]),endpoint=False)
    
    # add preliminary time stamples to cue
    cue.loc[:,'dTime'] = pd.to_timedelta(sig_sTime,unit='s')+prelim_synch_time
    
    # ACC signal excerpt at correct sample rate
    X = ACC.loc[ACC['dev_dTime']<xrange[1],:].copy()
    X = X.loc[X['dev_dTime']>=xrange[0],:].copy()
    sig_t = (X['dev_dTime'].dt.tz_localize(None) - prelim_synch_time.tz_localize(None)).dt.total_seconds()
    sig_v = X['signal']
    f = interpolate.interp1d(sig_t,sig_v,fill_value='extrapolate')
    new_sig = f(sig_sTime)
    signal = pd.DataFrame()
    signal.loc[:,'signal'] = new_sig
    signal.loc[signal['signal'].isna(),'signal'] = 0
    # scale signals a little 
    M = signal['signal'].quantile(0.998)
    signal.loc[:,'signal']  = signal['signal']/M
    signal.loc[signal['signal']>1,'signal'] = 1
    signal.loc[signal['signal']<0,'signal'] = 0
    signal.loc[:,'sTime'] = sig_sTime
    signal.loc[:,'dev_dTime'] = pd.to_timedelta(sig_sTime,unit='s')+prelim_synch_time

    length = np.min([len(signal),len(cue)]) # they should match, but just in case
    
    fig = plt.figure(figsize=(15,4))
    ax1 = plt.subplot(311)
    cue.plot.line(x='sTime',y=c_type,ax=ax1)
    signal.plot(x='sTime',y='signal',label='ACC',ax=ax1,)
    ax1.set_title('ACC synch alignment')
    ax1.set_ylabel('Unaligned')
    ax1.legend()
    #plt.xlim(cue_range)

    ax2 = plt.subplot(312)
    CCC = ax2.xcorr(cue[c_type].iloc[:length], signal['signal'].iloc[:length], usevlines=True, maxlags=int(max_offs*sf), normed=True, lw=3)
    ax2.grid(True)
    ax2.set_xticklabels('')
    cue.loc[:,'dev_dTime'] = cue['dTime'] - pd.to_timedelta(sampleshift_s*CCC[0][np.argmax(CCC[1])],unit='s')

    ax1 = plt.subplot(313)
    cue.plot.line(x='dev_dTime',y=c_type,ax=ax1)
    signal.plot(x='dev_dTime',y='signal',label='ACC',ax=ax1,)
    #plt.xlim(cue_range)
    ax1.grid(True)
    ax1.set_title('shift '+ str(np.round(sampleshift_s*CCC[0][np.argmax(CCC[1])],3))+ ' s')
    ax1.set_ylabel('Aligned')
    #plt.saveas('')
    plt.show()

    cue_time = cue.loc[find_nearest_idx(cue['sTime'], 0),'dev_dTime']
    C_results = {'best': cue_time,'CCC':CCC,'cue':cue,'signal':signal}
    return C_results

def min_align_noplot(ACC,cue,prelim_synch_time,max_offs):
    # sf infered from cue
    sampleshift_s = cue['sTime'].diff().median()
    sf = np.round(1/sampleshift_s) 
    t_range = [cue['sTime'].iloc[0],cue['sTime'].iloc[-1]]
    c_type = cue.columns[1]
    
    xrange = [pd.to_timedelta(t_range[0],unit = 's') + prelim_synch_time,pd.to_timedelta(t_range[1],unit = 's') + prelim_synch_time]
    sig_sTime = cue['sTime'].values #np.linspace(t_range[0],t_range[1],sf*(t_range[1]-t_range[0]),endpoint=False)
    
    # add preliminary time stamples to cue
    cue.loc[:,'dTime'] = pd.to_timedelta(sig_sTime,unit='s')+prelim_synch_time
    
    # ACC signal excerpt at correct sample rate
    X = ACC.loc[ACC['dev_dTime']<xrange[1],:].copy()
    X = X.loc[X['dev_dTime']>=xrange[0],:].copy()
    sig_t = (X['dev_dTime'].dt.tz_localize(None) - prelim_synch_time.tz_localize(None)).dt.total_seconds()
    sig_v = X['signal']
    f = interpolate.interp1d(sig_t,sig_v,fill_value='extrapolate')
    new_sig = f(sig_sTime)
    signal = pd.DataFrame()
    signal.loc[:,'signal'] = new_sig
    signal.loc[signal['signal'].isna(),'signal'] = 0
    # scale signals a little 
    M = signal['signal'].quantile(0.998)
    signal.loc[:,'signal']  = signal['signal']/M
    signal.loc[signal['signal']>1,'signal'] = 1
    signal.loc[signal['signal']<0,'signal'] = 0
    signal.loc[:,'sTime'] = sig_sTime
    signal.loc[:,'dev_dTime'] = pd.to_timedelta(sig_sTime,unit='s')+prelim_synch_time

    length = np.min([len(signal),len(cue)]) # they should match, but just in case
#     CCC = ax2.xcorr(cue[c_type].iloc[:length], signal['signal'].iloc[:length], usevlines=True, , normed=True, lw=3)
    x = sp.signal.detrend(np.asarray(cue[c_type].iloc[:length]))
    y = sp.signal.detrend(np.asarray(signal['signal'].iloc[:length]))
    maxlags=int(max_offs*sf)
    c = np.correlate(x, y, mode=2)
    CCC = [[],[]]
    CCC[0]= np.arange(-maxlags, maxlags + 1)
    CCC[1] = c[length - 1 - maxlags:length + maxlags]

    cue.loc[:,'dev_dTime'] = cue['dTime'] - pd.to_timedelta(sampleshift_s*CCC[0][np.argmax(CCC[1])],unit='s')

    cue_time = cue.loc[find_nearest_idx(cue['sTime'], 0),'dev_dTime']
    C_results = {'best': cue_time,'CCC':CCC,'cue':cue,'signal':signal}
    return C_results


def test_shift(Res,shifting):
    alt_cue = Res['cue'].copy()
    if pd.isnull(shifting):
        print('is nan')
        return
    else:
        if isinstance(shifting,float):
            dts = shifting
        else:
            if isinstance(shifting, dt.datetime):
                cue_zero = alt_cue.loc[find_nearest_idx(alt_cue['sTime'], 0),'dTime']
                dts = (cue_zero-shifting).total_seconds()
        c_type = alt_cue.columns[1]
        fig = plt.figure(figsize=(15,3))
        ax1 = plt.subplot(211)
        alt_cue.plot.line(x='sTime',y=c_type,ax=ax1)
        Res['signal'].plot(x='sTime',y='signal',label='ACC',ax=ax1,)
        ax1.set_ylabel('Unaligned')
        ax1.legend()
        alt_cue.loc[:,'dev_dTime'] =  Res['cue']['dTime'] - pd.to_timedelta(dts,unit='s')

        cue_time = alt_cue.loc[find_nearest_idx(alt_cue['sTime'], 0),'dev_dTime']
        dt_sh = pd.to_timedelta(7,unit='s')

        ax1 = plt.subplot(212)
        alt_cue.plot.line(x='dev_dTime',y=c_type,ax=ax1)
        Res['signal'].plot(x='dev_dTime',y='signal',label='ACC',ax=ax1,)
        ax1.set_xlim([cue_time-dt_sh/2,cue_time+dt_sh*1.5])
        ax1.grid(True)
        ax1.set_title('shift '+ str(dts)+ ' s')
        plt.show()

        return cue_time

def alt_xc_peaks(Res,ccthresh):
    CCC = Res['CCC']
    cue = Res['cue']
    mid_off = int((len(CCC[0])-1)/2)
    sf = np.round(1/cue['sTime'].diff().median())

    V =np.clip(CCC[1], ccthresh, 1)
    pks = pd.DataFrame()
    pks['ind'] = argrelextrema(V, np.greater)[0]
    pks['corr']= V[argrelextrema(V, np.greater)]
    pks['shift s'] = (pks['ind']-mid_off)/sf
    return pks

def dt_cut(V,dt_col,t1,t2):
    V[dt_col] = pd.to_datetime(V[dt_col])
    X = V.loc[V[dt_col]>t1,:].copy()
    X = X.loc[X[dt_col]<t2,:].copy()
    if len(X)<1:
        print('Recording does not intersect with that time interval')
        return
    else:
        return X