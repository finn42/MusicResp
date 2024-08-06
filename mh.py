import time
import datetime as dt
import math
import numpy as np 
import pandas as pd
from scipy.signal import butter,filtfilt
from scipy import interpolate
from scipy.interpolate import interp1d

def clean_IBI(beatTimes,beatIntervals,activitytype='default'):
    # function to clean IBI measurements to compensate for common intermittent sensor noise problems and some signal deviations that will complicate upstream calculations. This is not magic, bad signals are bad signals. This is just help.
    # Successive IBI values that exceed trusted behaviour are either:
    #     cut (higher octave errors) or 
    #     replaced with NA (alignment errors, lower octave errors)
    #
    # input: 
    #   beatTimes: seconds timestampe of beats recorded in seconds
    #   beatIntervals: ms time from previous beat , as output by sensor system or heartpy, same length as beatTimes
    #   activitytype: 'default' - non-respiratory musicians
    #               'resp' - respiratory musicians (winds, brass, singers)
    #               'listener' - seated listener, not performer, not singing alone
    # other potential activity types: dancer, singer/flute as distinct from winds with resistant mouth pieces (higher pressure)
    #
    # output:
    #   IBI = pandas dataframe with timestamps in seconds as index and IBI in ms as first column, with cut IBIs replaced with NA
    sig_t = pd.Series(beatTimes)
    if beatIntervals is None:
        sig_v = sig_t.diff()*1000
    else:
        sig_v = pd.Series(beatIntervals)
    if activitytype=='default':
        # make na beats that change more than 10% duration 
        sig_v_pre = np.log2(sig_v/sig_v.shift(1)).abs()
        sig_v_post = np.log2(sig_v.shift(-1)/sig_v).abs()
        sig_v = sig_v.mask((sig_v_post>0.15) & (sig_v_pre>0.15))
         # BPM cut offs, 48 to 167, 
        sig_v = sig_v.mask(sig_v<360)
        sig_v = sig_v.mask(sig_v>1250)
    if activitytype=='resp':
        # make na beats that change more than 25% duration, brass and winds need more forgiving thresholds
        sig_v_pre = np.log2(sig_v/sig_v.shift(1)).abs()
        sig_v_post = np.log2(sig_v.shift(-1)/sig_v).abs()
        sig_v = sig_v.mask((sig_v_post>0.45) & (sig_v_pre>0.45))
         # BPM cut offs, 48 to 182
        sig_v =sig_v.mask(sig_v<330)
        sig_v =sig_v.mask(sig_v>1250)
    if activitytype=='listener':
        # make na beats that change more than 15% duration,  
        sig_v_pre = np.log2(sig_v/sig_v.shift(1)).abs()
        sig_v_post = np.log2(sig_v.shift(-1)/sig_v).abs()
        sig_v = sig_v.mask((sig_v_post>0.2) & (sig_v_pre>0.2))
        # BPM cut offs, 48 to 150
        sig_v =sig_v.mask(sig_v<400)
        sig_v =sig_v.mask(sig_v>1250)
    
    IBI = pd.DataFrame(columns= ['time','IBI'])
    IBI['IBI'] = sig_v 
    IBI['time'] = sig_t
    IBI.set_index('time',drop = True, inplace = True)
    return IBI
#def clean_IBI(beatTimes,beatIntervals,activitytype='default'):

def ibi_beatfeats(sig_t,sig_v):
    # assume cleaned with IBI_clean, then ibi_beatfeats(IBI.index,IBI.IBI)
#     sig_v =sig_v.mask(sig_v.diff().abs()>120)
#     sig_v =sig_v.mask(sig_v<300)
    min_per = 5

    df_card = pd.DataFrame()
    df_card['time'] = sig_t
    df_card['IBI'] = sig_v 
    
    sig_t = pd.Series(sig_t)
    sig_v = pd.Series(sig_v)
    HR = (60000/sig_v)
    cutHR = HR[HR.notna()]
    f = interpolate.interp1d(sig_t[HR.notna()],HR[HR.notna()],kind = 'linear')
    new_t = sig_t[cutHR.index.min():cutHR.index.max()]
    newHR = f(new_t)
    df_card['HR1bt'] = HR
    df_card['normHR1bt'] = (HR - np.min([cutHR.quantile(0.10),80]))/np.max([30,cutHR.quantile(0.98)- cutHR.quantile(0.10)])

    df_card.loc[:,'HR10bt'] = (60000/sig_v).rolling(10,center=True,min_periods=min_per).median() # really lazy smoothing
    df_card.loc[:,'HR30bt'] = (60000/sig_v).rolling(30,center=True,min_periods=min_per).median() # really lazy smoothing
    
#     df_card.loc[:,'HRV10bt_ms'] = sig_v.diff().pow(2).rolling(10,center=True,min_periods=min_per).mean().pow(0.5)
#     df_card.loc[:,'HRV30bt_ms'] = sig_v.diff().pow(2).rolling(30,center=True,min_periods=min_per).mean().pow(0.5)
#     df_card.loc[:,'HRV10bt_qms'] = sig_v.diff().rolling(10,center=True,min_periods=min_per).quantile(0.75)-sig_v.diff().rolling(10,center=True,min_periods=min_per).quantile(0.25)
#     df_card.loc[:,'HRV30bt_qms'] = sig_v.diff().rolling(30,center=True,min_periods=min_per).quantile(0.75)-sig_v.diff().rolling(30,center=True,min_periods=min_per).quantile(0.25)

#     a = sig_v.copy()
#     division = pd.Series(np.divide(a.values[1:],a.values[:-1]))
#     a.loc[1:] = division
#     a.loc[0]=1
#     a.iloc[-1]=1
#     df_card.loc[:,'HRV10bt_r'] = (a-1).abs().rolling(10,center=True,min_periods=min_per).median()+1
#     df_card.loc[:,'HRV30bt_r'] = (a-1).abs().rolling(30,center=True,min_periods=min_per).median()+1
#     a = np.log2(a)
#     df_card.loc[:,'HRV10bt_ar'] = np.exp2(a.rolling(10,center=True,min_periods=min_per).quantile(0.75)-a.rolling(10,center=True,min_periods=min_per).quantile(0.25))
#     df_card.loc[:,'HRV30bt_ar']  = np.exp2(a.rolling(30,center=True,min_periods=min_per).quantile(0.75)-a.rolling(30,center=True,min_periods=min_per).quantile(0.25))
    df_card.set_index('time',drop = True, inplace = True)
 
    return df_card

def ibi_sbeatfeats(sig_t,sig_v):
    # assume cleaned with IBI_clean, then ibi_beatfeats(IBI.index,IBI.IBI)
#     sig_v =sig_v.mask(sig_v.diff().abs()>120)
#     sig_v =sig_v.mask(sig_v<300)
    min_per = 5

    df_card = pd.DataFrame()
    df_card['time'] = sig_t
    df_card['IBI'] = sig_v 
    
    sig_t = pd.Series(sig_t)
    sig_v = pd.Series(sig_v)
    HR = (60000/sig_v)
    cutHR = HR[HR.notna()]
    f = interpolate.interp1d(sig_t[HR.notna()],HR[HR.notna()],kind = 'linear')
    new_t = sig_t[cutHR.index.min():cutHR.index.max()]
    newHR = f(new_t)
    df_card['HR1bt'] = HR
    df_card['normHR1bt'] = (HR - np.min([cutHR.quantile(0.10),80]))/np.max([30,cutHR.quantile(0.98)- cutHR.quantile(0.10)])
    df_card.set_index('time',drop = True, inplace = True)
    
    df_card.loc[:,'HR10s'] = np.nan
    s = 10
    for i in df_card.loc[(df_card.index > df_card.index[0]+s/2) & (df_card.index < df_card.index[-1]-s/2),:].index:
        df_card.loc[i,'HR10s'] = df_card.loc[(df_card.index < i+s/2) & (df_card.index > i-s/2),'HR1bt'].mean()
        
    df_card.loc[:,'HR30s'] = np.nan
    s = 30        
    for i in df_card.loc[(df_card.index > df_card.index[0]+s/2) & (df_card.index < df_card.index[-1]-s/2),:].index:
        df_card.loc[i,'HR30s'] = df_card.loc[(df_card.index < i+s/2) & (df_card.index > i-s/2),'HR1bt'].mean()

    return df_card


def ibi_feats(sig_t,sig_v,time_s):
# assume cleaned with IBI_clean, then ibi_beatfeats(IBI.index,IBI.IBI)
#     sig_v =sig_v.mask(sig_v.diff().abs()>120)
#     sig_v =sig_v.mask(sig_v<300)
    df_card = ibi_beatfeats(sig_t,sig_v)
    df_ts = pd.DataFrame(index = time_s)
    for c in df_card.columns:
            f = interpolate.interp1d(sig_t,df_card.loc[:,c].values,fill_value='extrapolate')
            df_ts.loc[:,c] = f(time_s)
    
    return df_ts

def refeats(df_s,time_s):
#     sf = 1/pd.Series(time_s).diff().mode().values[0]
    r_df = pd.DataFrame(index = time_s, columns = df_s.columns)
    for c in df_s.columns:
        f = interpolate.interp1d(df_s.index, df_s[c],fill_value='extrapolate')
        r_df.loc[:,c] = f(time_s)
    return r_df

def scaledcoh(df_s,frame_sizes,step=1):
    sf = 1/pd.Series(df_s.index).diff().mode().values[0]
    # need to adapt pcoor dimensions to correct for indexing on a step greater than a sample
    pcoor = pd.DataFrame(index = df_s.index, columns = frame_sizes)
    for j in range(len(frame_sizes)):
        fr = int((frame_sizes[j]*sf)/2)
        for i in range(fr,len(df_s.index)-fr,step):
            frame = df_s.iloc[i-fr:i+fr,:]
            pcoor.iloc[i,j] = frame.mean(axis=1).std()/(frame.std(axis=0).mean())
            
    return pcoor.astype(float)

def hrv_beatfeats(sig_t,sig_v):
    # assume cleaned with mh.IBI_clean, then hrv_beatfeats(IBI.index,IBI.IBI)
    min_per = 5

    df_card = pd.DataFrame(index = sig_t)
    df_card['IBI'] = sig_v 

    df_card.loc[:,'HRV10bt_ms'] = sig_v.diff().pow(2).rolling(10,center=True,min_periods=min_per).mean().pow(0.5)
    df_card.loc[:,'HRV30bt_ms'] = sig_v.diff().pow(2).rolling(30,center=True,min_periods=min_per).mean().pow(0.5)
    df_card.loc[:,'HRV10bt_qms'] = sig_v.diff().rolling(10,center=True,min_periods=min_per).quantile(0.75)-sig_v.diff().rolling(10,center=True,min_periods=min_per).quantile(0.25)
    df_card.loc[:,'HRV30bt_qms'] = sig_v.diff().rolling(30,center=True,min_periods=min_per).quantile(0.75)-sig_v.diff().rolling(30,center=True,min_periods=min_per).quantile(0.25)

    a = sig_v.astype(float)
    division = pd.Series(np.divide(a.values[1:],a.values[:-1]))
    a.iloc[1:] = division
    a.iloc[0]=1
    a.iloc[-1]=1
    df_card.loc[:,'HRV10bt_r'] = (a-1).abs().rolling(10,center=True,min_periods=min_per).median()+1
    df_card.loc[:,'HRV30bt_r'] = (a-1).abs().rolling(30,center=True,min_periods=min_per).median()+1
    a = np.log2(a)
    df_card.loc[:,'HRV10bt_ar'] = np.exp2(a.rolling(10,center=True,min_periods=min_per).quantile(0.75)-a.rolling(10,center=True,min_periods=min_per).quantile(0.25))
    df_card.loc[:,'HRV30bt_ar']  = np.exp2(a.rolling(30,center=True,min_periods=min_per).quantile(0.75)-a.rolling(30,center=True,min_periods=min_per).quantile(0.25))
 
    return df_card