from scipy.io import loadmat
import pandas as pd
import numpy as np
from scipy import signal

'''

[12, 8, 1114, 15] = size(eeg)

Number of targets : 12
Number of channels : 8
Number of sampling points : 1114
Number of trials : 15
Sampling rate [Hz] : 256

[9.25, 11.25, 13.25, 9.75, 11.75, 13.75, 10.25, 12.25, 14.25, 10.75, 12.75, 14.75] Hz

'''

def band_filter_process(input):
    
    b_HPF_1,a_HPF_1 = signal.butter(4, 0.046875, btype='highpass') #4阶，6Hz ， 6/(sampling_rate/2) 和iirfilter效果一样
    b_LPF_1,a_LPF_1 = signal.butter(4, 0.125, btype='lowpass') #4阶，16Hz ，16/(sampling_rate/2) 和iirfilter效果一样
    
    b_HPF_2,a_HPF_2 = signal.butter(4, 0.125, btype='highpass') #4阶，16Hz ， 16/(sampling_rate/2) 和iirfilter效果一样
    b_LPF_2,a_LPF_2 = signal.butter(4, 0.25, btype='lowpass') #4阶，32Hz ，32/(sampling_rate/2) 和iirfilter效果一样
    
    b_HPF_3,a_HPF_3 = signal.butter(4, 0.25, btype='highpass') #4阶，32Hz ， 32/(sampling_rate/2) 和iirfilter效果一样
    b_LPF_3,a_LPF_3 = signal.butter(4, 0.5, btype='lowpass') #4阶，64Hz ，64/(sampling_rate/2) 和iirfilter效果一样
    
    data_HF_1 = signal.filtfilt(b_HPF_1, a_HPF_1, input) #零相位滤波，滤两次，反向多滤了一次
    band1 = signal.filtfilt(b_LPF_1, a_LPF_1, data_HF_1) #6-16
    
    data_HF_2 = signal.filtfilt(b_HPF_2, a_HPF_2, input) #零相位滤波，滤两次，反向多滤了一次
    band2 = signal.filtfilt(b_LPF_2, a_LPF_2, data_HF_2) #16-32
    
    data_HF_3 = signal.filtfilt(b_HPF_3, a_HPF_3, input) #零相位滤波，滤两次，反向多滤了一次
    band3 = signal.filtfilt(b_LPF_3, a_LPF_3, data_HF_3) #32-64
    
    return band1,band2,band3

def data_batch_FFT_one_tester_abs_UD(test_trainer,time_length):
    
    simpling_rate = 256
    window_length = int(256*time_length) #窗口长度1s
    data_length = 1024 #4s
    fft_scale = int(data_length/window_length) #5s则scale=5

    # print('Preparing dataset.....')
    all_datas_1 = []
    all_datas_2 = []
    all_datas_3 = []
    all_labels = []
    
    test_datas_1 = []
    test_datas_2 = []
    test_datas_3 = []
    test_labels = []
    
    rawdata = loadmat('./data/' + test_trainer + '.mat') #1号被试
    rawdata = rawdata['eeg']
    
    for j in range(12): #12个targets
        targets_index = j
        for k in range(15): #15个trials
            block_index = k
            
            channel1_data = rawdata[targets_index,0,72:1096,block_index]
            channel2_data = rawdata[targets_index,1,72:1096,block_index]
            channel3_data = rawdata[targets_index,2,72:1096,block_index]
            channel4_data = rawdata[targets_index,3,72:1096,block_index]
            channel5_data = rawdata[targets_index,4,72:1096,block_index]
            channel6_data = rawdata[targets_index,5,72:1096,block_index]
            channel7_data = rawdata[targets_index,6,72:1096,block_index]
            channel8_data = rawdata[targets_index,7,72:1096,block_index]
            
            yd4_1,yd3_1,yd2_1 = band_filter_process(channel1_data)
            yd4_2,yd3_2,yd2_2 = band_filter_process(channel2_data)
            yd4_3,yd3_3,yd2_3 = band_filter_process(channel3_data)
            yd4_4,yd3_4,yd2_4 = band_filter_process(channel4_data)
            yd4_5,yd3_5,yd2_5 = band_filter_process(channel5_data)
            yd4_6,yd3_6,yd2_6 = band_filter_process(channel6_data)
            yd4_7,yd3_7,yd2_7 = band_filter_process(channel7_data)
            yd4_8,yd3_8,yd2_8 = band_filter_process(channel8_data)
            
            for m in range(fft_scale):
                
                all_datas_1.append(np.array([(np.abs(np.fft.rfft(yd4_1[m*window_length:(m+1)*window_length],1024))/window_length)[round(5/0.25):round(55/0.25) + 1],
                                             (np.abs(np.fft.rfft(yd4_2[m*window_length:(m+1)*window_length],1024))/window_length)[round(5/0.25):round(55/0.25) + 1],
                                             (np.abs(np.fft.rfft(yd4_3[m*window_length:(m+1)*window_length],1024))/window_length)[round(5/0.25):round(55/0.25) + 1],
                                             (np.abs(np.fft.rfft(yd4_4[m*window_length:(m+1)*window_length],1024))/window_length)[round(5/0.25):round(55/0.25) + 1],
                                             (np.abs(np.fft.rfft(yd4_5[m*window_length:(m+1)*window_length],1024))/window_length)[round(5/0.25):round(55/0.25) + 1],
                                             (np.abs(np.fft.rfft(yd4_6[m*window_length:(m+1)*window_length],1024))/window_length)[round(5/0.25):round(55/0.25) + 1],
                                             (np.abs(np.fft.rfft(yd4_7[m*window_length:(m+1)*window_length],1024))/window_length)[round(5/0.25):round(55/0.25) + 1],
                                             (np.abs(np.fft.rfft(yd4_8[m*window_length:(m+1)*window_length],1024))/window_length)[round(5/0.25):round(55/0.25) + 1],
                                             (np.abs(np.fft.rfft(yd4_1[m*window_length:(m+1)*window_length],1024))/window_length)[round(5/0.25):round(55/0.25) + 1],
                                             (np.abs(np.fft.rfft(yd4_2[m*window_length:(m+1)*window_length],1024))/window_length)[round(5/0.25):round(55/0.25) + 1]]))
                
                all_datas_2.append(np.array([(np.abs(np.fft.rfft(yd3_1[m*window_length:(m+1)*window_length],1024))/window_length)[round(5/0.25):round(55/0.25) + 1],
                                             (np.abs(np.fft.rfft(yd3_2[m*window_length:(m+1)*window_length],1024))/window_length)[round(5/0.25):round(55/0.25) + 1],
                                             (np.abs(np.fft.rfft(yd3_3[m*window_length:(m+1)*window_length],1024))/window_length)[round(5/0.25):round(55/0.25) + 1],
                                             (np.abs(np.fft.rfft(yd3_4[m*window_length:(m+1)*window_length],1024))/window_length)[round(5/0.25):round(55/0.25) + 1],
                                             (np.abs(np.fft.rfft(yd3_5[m*window_length:(m+1)*window_length],1024))/window_length)[round(5/0.25):round(55/0.25) + 1],
                                             (np.abs(np.fft.rfft(yd3_6[m*window_length:(m+1)*window_length],1024))/window_length)[round(5/0.25):round(55/0.25) + 1],
                                             (np.abs(np.fft.rfft(yd3_7[m*window_length:(m+1)*window_length],1024))/window_length)[round(5/0.25):round(55/0.25) + 1],
                                             (np.abs(np.fft.rfft(yd3_8[m*window_length:(m+1)*window_length],1024))/window_length)[round(5/0.25):round(55/0.25) + 1],
                                             (np.abs(np.fft.rfft(yd3_1[m*window_length:(m+1)*window_length],1024))/window_length)[round(5/0.25):round(55/0.25) + 1],
                                             (np.abs(np.fft.rfft(yd3_2[m*window_length:(m+1)*window_length],1024))/window_length)[round(5/0.25):round(55/0.25) + 1]]))
                
                all_datas_3.append(np.array([(np.abs(np.fft.rfft(yd2_1[m*window_length:(m+1)*window_length],1024))/window_length)[round(5/0.25):round(55/0.25) + 1],
                                             (np.abs(np.fft.rfft(yd2_2[m*window_length:(m+1)*window_length],1024))/window_length)[round(5/0.25):round(55/0.25) + 1],
                                             (np.abs(np.fft.rfft(yd2_3[m*window_length:(m+1)*window_length],1024))/window_length)[round(5/0.25):round(55/0.25) + 1],
                                             (np.abs(np.fft.rfft(yd2_4[m*window_length:(m+1)*window_length],1024))/window_length)[round(5/0.25):round(55/0.25) + 1],
                                             (np.abs(np.fft.rfft(yd2_5[m*window_length:(m+1)*window_length],1024))/window_length)[round(5/0.25):round(55/0.25) + 1],
                                             (np.abs(np.fft.rfft(yd2_6[m*window_length:(m+1)*window_length],1024))/window_length)[round(5/0.25):round(55/0.25) + 1],
                                             (np.abs(np.fft.rfft(yd2_7[m*window_length:(m+1)*window_length],1024))/window_length)[round(5/0.25):round(55/0.25) + 1],
                                             (np.abs(np.fft.rfft(yd2_8[m*window_length:(m+1)*window_length],1024))/window_length)[round(5/0.25):round(55/0.25) + 1],
                                             (np.abs(np.fft.rfft(yd2_1[m*window_length:(m+1)*window_length],1024))/window_length)[round(5/0.25):round(55/0.25) + 1],
                                             (np.abs(np.fft.rfft(yd2_2[m*window_length:(m+1)*window_length],1024))/window_length)[round(5/0.25):round(55/0.25) + 1]]))

                all_labels.append(targets_index)
    
    all_datas_1 = np.expand_dims(np.array(all_datas_1),axis=-1)
    all_datas_2 = np.expand_dims(np.array(all_datas_2),axis=-1)
    all_datas_3 = np.expand_dims(np.array(all_datas_3),axis=-1)
    all_labels = np.array(all_labels)

    return all_datas_1,all_datas_2,all_datas_3,all_labels

def data_batch_FFT_one_tester_real_UD(test_trainer,time_length):
    
    simpling_rate = 256
    window_length = int(256*time_length) #窗口长度1s
    data_length = 1024 #4s
    fft_scale = int(data_length/window_length) #5s则scale=5

    # print('Preparing dataset.....')
    all_datas_1 = []
    all_datas_2 = []
    all_datas_3 = []
    all_labels = []
    
    test_datas_1 = []
    test_datas_2 = []
    test_datas_3 = []
    test_labels = []
    
    rawdata = loadmat('./data/' + test_trainer + '.mat') #1号被试
    rawdata = rawdata['eeg']
    
    for j in range(12): #12个targets
        targets_index = j
        for k in range(15): #15个trials
            block_index = k
            
            channel1_data = rawdata[targets_index,0,72:1096,block_index]
            channel2_data = rawdata[targets_index,1,72:1096,block_index]
            channel3_data = rawdata[targets_index,2,72:1096,block_index]
            channel4_data = rawdata[targets_index,3,72:1096,block_index]
            channel5_data = rawdata[targets_index,4,72:1096,block_index]
            channel6_data = rawdata[targets_index,5,72:1096,block_index]
            channel7_data = rawdata[targets_index,6,72:1096,block_index]
            channel8_data = rawdata[targets_index,7,72:1096,block_index]
            
            yd4_1,yd3_1,yd2_1 = band_filter_process(channel1_data)
            yd4_2,yd3_2,yd2_2 = band_filter_process(channel2_data)
            yd4_3,yd3_3,yd2_3 = band_filter_process(channel3_data)
            yd4_4,yd3_4,yd2_4 = band_filter_process(channel4_data)
            yd4_5,yd3_5,yd2_5 = band_filter_process(channel5_data)
            yd4_6,yd3_6,yd2_6 = band_filter_process(channel6_data)
            yd4_7,yd3_7,yd2_7 = band_filter_process(channel7_data)
            yd4_8,yd3_8,yd2_8 = band_filter_process(channel8_data)
            
            for m in range(fft_scale):
                
                all_datas_1.append(np.array([(np.real(np.fft.rfft(yd4_1[m*window_length:(m+1)*window_length],1024))/window_length)[round(5/0.25):round(55/0.25) + 1],
                                             (np.real(np.fft.rfft(yd4_2[m*window_length:(m+1)*window_length],1024))/window_length)[round(5/0.25):round(55/0.25) + 1],
                                             (np.real(np.fft.rfft(yd4_3[m*window_length:(m+1)*window_length],1024))/window_length)[round(5/0.25):round(55/0.25) + 1],
                                             (np.real(np.fft.rfft(yd4_4[m*window_length:(m+1)*window_length],1024))/window_length)[round(5/0.25):round(55/0.25) + 1],
                                             (np.real(np.fft.rfft(yd4_5[m*window_length:(m+1)*window_length],1024))/window_length)[round(5/0.25):round(55/0.25) + 1],
                                             (np.real(np.fft.rfft(yd4_6[m*window_length:(m+1)*window_length],1024))/window_length)[round(5/0.25):round(55/0.25) + 1],
                                             (np.real(np.fft.rfft(yd4_7[m*window_length:(m+1)*window_length],1024))/window_length)[round(5/0.25):round(55/0.25) + 1],
                                             (np.real(np.fft.rfft(yd4_8[m*window_length:(m+1)*window_length],1024))/window_length)[round(5/0.25):round(55/0.25) + 1],
                                             (np.real(np.fft.rfft(yd4_1[m*window_length:(m+1)*window_length],1024))/window_length)[round(5/0.25):round(55/0.25) + 1],
                                             (np.real(np.fft.rfft(yd4_2[m*window_length:(m+1)*window_length],1024))/window_length)[round(5/0.25):round(55/0.25) + 1]]))
                
                all_datas_2.append(np.array([(np.real(np.fft.rfft(yd3_1[m*window_length:(m+1)*window_length],1024))/window_length)[round(5/0.25):round(55/0.25) + 1],
                                             (np.real(np.fft.rfft(yd3_2[m*window_length:(m+1)*window_length],1024))/window_length)[round(5/0.25):round(55/0.25) + 1],
                                             (np.real(np.fft.rfft(yd3_3[m*window_length:(m+1)*window_length],1024))/window_length)[round(5/0.25):round(55/0.25) + 1],
                                             (np.real(np.fft.rfft(yd3_4[m*window_length:(m+1)*window_length],1024))/window_length)[round(5/0.25):round(55/0.25) + 1],
                                             (np.real(np.fft.rfft(yd3_5[m*window_length:(m+1)*window_length],1024))/window_length)[round(5/0.25):round(55/0.25) + 1],
                                             (np.real(np.fft.rfft(yd3_6[m*window_length:(m+1)*window_length],1024))/window_length)[round(5/0.25):round(55/0.25) + 1],
                                             (np.real(np.fft.rfft(yd3_7[m*window_length:(m+1)*window_length],1024))/window_length)[round(5/0.25):round(55/0.25) + 1],
                                             (np.real(np.fft.rfft(yd3_8[m*window_length:(m+1)*window_length],1024))/window_length)[round(5/0.25):round(55/0.25) + 1],
                                             (np.real(np.fft.rfft(yd3_1[m*window_length:(m+1)*window_length],1024))/window_length)[round(5/0.25):round(55/0.25) + 1],
                                             (np.real(np.fft.rfft(yd3_2[m*window_length:(m+1)*window_length],1024))/window_length)[round(5/0.25):round(55/0.25) + 1]]))
                
                all_datas_3.append(np.array([(np.real(np.fft.rfft(yd2_1[m*window_length:(m+1)*window_length],1024))/window_length)[round(5/0.25):round(55/0.25) + 1],
                                             (np.real(np.fft.rfft(yd2_2[m*window_length:(m+1)*window_length],1024))/window_length)[round(5/0.25):round(55/0.25) + 1],
                                             (np.real(np.fft.rfft(yd2_3[m*window_length:(m+1)*window_length],1024))/window_length)[round(5/0.25):round(55/0.25) + 1],
                                             (np.real(np.fft.rfft(yd2_4[m*window_length:(m+1)*window_length],1024))/window_length)[round(5/0.25):round(55/0.25) + 1],
                                             (np.real(np.fft.rfft(yd2_5[m*window_length:(m+1)*window_length],1024))/window_length)[round(5/0.25):round(55/0.25) + 1],
                                             (np.real(np.fft.rfft(yd2_6[m*window_length:(m+1)*window_length],1024))/window_length)[round(5/0.25):round(55/0.25) + 1],
                                             (np.real(np.fft.rfft(yd2_7[m*window_length:(m+1)*window_length],1024))/window_length)[round(5/0.25):round(55/0.25) + 1],
                                             (np.real(np.fft.rfft(yd2_8[m*window_length:(m+1)*window_length],1024))/window_length)[round(5/0.25):round(55/0.25) + 1],
                                             (np.real(np.fft.rfft(yd2_1[m*window_length:(m+1)*window_length],1024))/window_length)[round(5/0.25):round(55/0.25) + 1],
                                             (np.real(np.fft.rfft(yd2_2[m*window_length:(m+1)*window_length],1024))/window_length)[round(5/0.25):round(55/0.25) + 1]]))

                all_labels.append(targets_index)
    
    all_datas_1 = np.expand_dims(np.array(all_datas_1),axis=-1)
    all_datas_2 = np.expand_dims(np.array(all_datas_2),axis=-1)
    all_datas_3 = np.expand_dims(np.array(all_datas_3),axis=-1)
    all_labels = np.array(all_labels)

    return all_datas_1,all_datas_2,all_datas_3,all_labels

def data_batch_FFT_one_tester_complex_UD(test_trainer,time_length):
    
    simpling_rate = 256
    window_length = int(256*time_length) #窗口长度1s
    data_length = 1024 #4s
    fft_scale = int(data_length/window_length) #5s则scale=5

    # print('Preparing dataset.....')
    all_datas_1 = []
    all_datas_2 = []
    all_datas_3 = []
    all_labels = []
    
    test_datas_1 = []
    test_datas_2 = []
    test_datas_3 = []
    test_labels = []
    
    rawdata = loadmat('./data/' + test_trainer + '.mat') #1号被试
    rawdata = rawdata['eeg']
    
    for j in range(12): #12个targets
        targets_index = j
        for k in range(15): #15个trials
            block_index = k
            
            channel1_data = rawdata[targets_index,0,72:1096,block_index]
            channel2_data = rawdata[targets_index,1,72:1096,block_index]
            channel3_data = rawdata[targets_index,2,72:1096,block_index]
            channel4_data = rawdata[targets_index,3,72:1096,block_index]
            channel5_data = rawdata[targets_index,4,72:1096,block_index]
            channel6_data = rawdata[targets_index,5,72:1096,block_index]
            channel7_data = rawdata[targets_index,6,72:1096,block_index]
            channel8_data = rawdata[targets_index,7,72:1096,block_index]
            
            yd4_1,yd3_1,yd2_1 = band_filter_process(channel1_data)
            yd4_2,yd3_2,yd2_2 = band_filter_process(channel2_data)
            yd4_3,yd3_3,yd2_3 = band_filter_process(channel3_data)
            yd4_4,yd3_4,yd2_4 = band_filter_process(channel4_data)
            yd4_5,yd3_5,yd2_5 = band_filter_process(channel5_data)
            yd4_6,yd3_6,yd2_6 = band_filter_process(channel6_data)
            yd4_7,yd3_7,yd2_7 = band_filter_process(channel7_data)
            yd4_8,yd3_8,yd2_8 = band_filter_process(channel8_data)
            
            for m in range(fft_scale):
                
                yd4_1_fft = np.fft.rfft(yd4_1[m*window_length:(m+1)*window_length],1024)/window_length
                yd4_2_fft = np.fft.rfft(yd4_2[m*window_length:(m+1)*window_length],1024)/window_length
                yd4_3_fft = np.fft.rfft(yd4_3[m*window_length:(m+1)*window_length],1024)/window_length
                yd4_4_fft = np.fft.rfft(yd4_4[m*window_length:(m+1)*window_length],1024)/window_length
                yd4_5_fft = np.fft.rfft(yd4_5[m*window_length:(m+1)*window_length],1024)/window_length
                yd4_6_fft = np.fft.rfft(yd4_6[m*window_length:(m+1)*window_length],1024)/window_length
                yd4_7_fft = np.fft.rfft(yd4_7[m*window_length:(m+1)*window_length],1024)/window_length
                yd4_8_fft = np.fft.rfft(yd4_8[m*window_length:(m+1)*window_length],1024)/window_length
                
                yd3_1_fft = np.fft.rfft(yd3_1[m*window_length:(m+1)*window_length],1024)/window_length
                yd3_2_fft = np.fft.rfft(yd3_2[m*window_length:(m+1)*window_length],1024)/window_length
                yd3_3_fft = np.fft.rfft(yd3_3[m*window_length:(m+1)*window_length],1024)/window_length
                yd3_4_fft = np.fft.rfft(yd3_4[m*window_length:(m+1)*window_length],1024)/window_length
                yd3_5_fft = np.fft.rfft(yd3_5[m*window_length:(m+1)*window_length],1024)/window_length
                yd3_6_fft = np.fft.rfft(yd3_6[m*window_length:(m+1)*window_length],1024)/window_length
                yd3_7_fft = np.fft.rfft(yd3_7[m*window_length:(m+1)*window_length],1024)/window_length
                yd3_8_fft = np.fft.rfft(yd3_8[m*window_length:(m+1)*window_length],1024)/window_length
                
                yd2_1_fft = np.fft.rfft(yd2_1[m*window_length:(m+1)*window_length],1024)/window_length
                yd2_2_fft = np.fft.rfft(yd2_2[m*window_length:(m+1)*window_length],1024)/window_length
                yd2_3_fft = np.fft.rfft(yd2_3[m*window_length:(m+1)*window_length],1024)/window_length
                yd2_4_fft = np.fft.rfft(yd2_4[m*window_length:(m+1)*window_length],1024)/window_length
                yd2_5_fft = np.fft.rfft(yd2_5[m*window_length:(m+1)*window_length],1024)/window_length
                yd2_6_fft = np.fft.rfft(yd2_6[m*window_length:(m+1)*window_length],1024)/window_length
                yd2_7_fft = np.fft.rfft(yd2_7[m*window_length:(m+1)*window_length],1024)/window_length
                yd2_8_fft = np.fft.rfft(yd2_8[m*window_length:(m+1)*window_length],1024)/window_length
                
                
                all_datas_1.append(np.array([np.concatenate((np.real(yd4_1_fft)[round(5/0.25):round(55/0.25) + 1],np.imag(yd4_1_fft)[round(5/0.25):round(55/0.25) + 1]),axis=0),
                                             np.concatenate((np.real(yd4_2_fft)[round(5/0.25):round(55/0.25) + 1],np.imag(yd4_2_fft)[round(5/0.25):round(55/0.25) + 1]),axis=0),
                                             np.concatenate((np.real(yd4_3_fft)[round(5/0.25):round(55/0.25) + 1],np.imag(yd4_3_fft)[round(5/0.25):round(55/0.25) + 1]),axis=0),
                                             np.concatenate((np.real(yd4_4_fft)[round(5/0.25):round(55/0.25) + 1],np.imag(yd4_4_fft)[round(5/0.25):round(55/0.25) + 1]),axis=0),
                                             np.concatenate((np.real(yd4_5_fft)[round(5/0.25):round(55/0.25) + 1],np.imag(yd4_5_fft)[round(5/0.25):round(55/0.25) + 1]),axis=0),
                                             np.concatenate((np.real(yd4_6_fft)[round(5/0.25):round(55/0.25) + 1],np.imag(yd4_6_fft)[round(5/0.25):round(55/0.25) + 1]),axis=0),
                                             np.concatenate((np.real(yd4_7_fft)[round(5/0.25):round(55/0.25) + 1],np.imag(yd4_7_fft)[round(5/0.25):round(55/0.25) + 1]),axis=0),
                                             np.concatenate((np.real(yd4_8_fft)[round(5/0.25):round(55/0.25) + 1],np.imag(yd4_8_fft)[round(5/0.25):round(55/0.25) + 1]),axis=0),
                                             np.concatenate((np.real(yd4_1_fft)[round(5/0.25):round(55/0.25) + 1],np.imag(yd4_1_fft)[round(5/0.25):round(55/0.25) + 1]),axis=0),
                                             np.concatenate((np.real(yd4_2_fft)[round(5/0.25):round(55/0.25) + 1],np.imag(yd4_2_fft)[round(5/0.25):round(55/0.25) + 1]),axis=0)]))
                
                all_datas_2.append(np.array([np.concatenate((np.real(yd3_1_fft)[round(5/0.25):round(55/0.25) + 1],np.imag(yd3_1_fft)[round(5/0.25):round(55/0.25) + 1]),axis=0),
                                             np.concatenate((np.real(yd3_2_fft)[round(5/0.25):round(55/0.25) + 1],np.imag(yd3_2_fft)[round(5/0.25):round(55/0.25) + 1]),axis=0),
                                             np.concatenate((np.real(yd3_3_fft)[round(5/0.25):round(55/0.25) + 1],np.imag(yd3_3_fft)[round(5/0.25):round(55/0.25) + 1]),axis=0),
                                             np.concatenate((np.real(yd3_4_fft)[round(5/0.25):round(55/0.25) + 1],np.imag(yd3_4_fft)[round(5/0.25):round(55/0.25) + 1]),axis=0),
                                             np.concatenate((np.real(yd3_5_fft)[round(5/0.25):round(55/0.25) + 1],np.imag(yd3_5_fft)[round(5/0.25):round(55/0.25) + 1]),axis=0),
                                             np.concatenate((np.real(yd3_6_fft)[round(5/0.25):round(55/0.25) + 1],np.imag(yd3_6_fft)[round(5/0.25):round(55/0.25) + 1]),axis=0),
                                             np.concatenate((np.real(yd3_7_fft)[round(5/0.25):round(55/0.25) + 1],np.imag(yd3_7_fft)[round(5/0.25):round(55/0.25) + 1]),axis=0),
                                             np.concatenate((np.real(yd3_8_fft)[round(5/0.25):round(55/0.25) + 1],np.imag(yd3_8_fft)[round(5/0.25):round(55/0.25) + 1]),axis=0),
                                             np.concatenate((np.real(yd3_1_fft)[round(5/0.25):round(55/0.25) + 1],np.imag(yd3_1_fft)[round(5/0.25):round(55/0.25) + 1]),axis=0),
                                             np.concatenate((np.real(yd3_2_fft)[round(5/0.25):round(55/0.25) + 1],np.imag(yd3_2_fft)[round(5/0.25):round(55/0.25) + 1]),axis=0)]))
                
                all_datas_3.append(np.array([np.concatenate((np.real(yd2_1_fft)[round(5/0.25):round(55/0.25) + 1],np.imag(yd2_1_fft)[round(5/0.25):round(55/0.25) + 1]),axis=0),
                                             np.concatenate((np.real(yd2_2_fft)[round(5/0.25):round(55/0.25) + 1],np.imag(yd2_2_fft)[round(5/0.25):round(55/0.25) + 1]),axis=0),
                                             np.concatenate((np.real(yd2_3_fft)[round(5/0.25):round(55/0.25) + 1],np.imag(yd2_3_fft)[round(5/0.25):round(55/0.25) + 1]),axis=0),
                                             np.concatenate((np.real(yd2_4_fft)[round(5/0.25):round(55/0.25) + 1],np.imag(yd2_4_fft)[round(5/0.25):round(55/0.25) + 1]),axis=0),
                                             np.concatenate((np.real(yd2_5_fft)[round(5/0.25):round(55/0.25) + 1],np.imag(yd2_5_fft)[round(5/0.25):round(55/0.25) + 1]),axis=0),
                                             np.concatenate((np.real(yd2_6_fft)[round(5/0.25):round(55/0.25) + 1],np.imag(yd2_6_fft)[round(5/0.25):round(55/0.25) + 1]),axis=0),
                                             np.concatenate((np.real(yd2_7_fft)[round(5/0.25):round(55/0.25) + 1],np.imag(yd2_7_fft)[round(5/0.25):round(55/0.25) + 1]),axis=0),
                                             np.concatenate((np.real(yd2_8_fft)[round(5/0.25):round(55/0.25) + 1],np.imag(yd2_8_fft)[round(5/0.25):round(55/0.25) + 1]),axis=0),
                                             np.concatenate((np.real(yd2_1_fft)[round(5/0.25):round(55/0.25) + 1],np.imag(yd2_1_fft)[round(5/0.25):round(55/0.25) + 1]),axis=0),
                                             np.concatenate((np.real(yd2_2_fft)[round(5/0.25):round(55/0.25) + 1],np.imag(yd2_2_fft)[round(5/0.25):round(55/0.25) + 1]),axis=0)]))

                all_labels.append(targets_index)
    
    all_datas_1 = np.expand_dims(np.array(all_datas_1),axis=-1)
    all_datas_2 = np.expand_dims(np.array(all_datas_2),axis=-1)
    all_datas_3 = np.expand_dims(np.array(all_datas_3),axis=-1)
    all_labels = np.array(all_labels)

    return all_datas_1,all_datas_2,all_datas_3,all_labels


'''
Main
'''

if __name__ == '__main__':
    
    # data_batch_FFT(True)
    data_batch_FFT_one_tester_complex_UD('s3',time_length = 1)


















