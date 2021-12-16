import json
import textgrid as tg
import numpy as np

def get_parameter_size(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_num, trainable_num

def denormalize(kps, data_mean, data_std):
    #kps: (B, T, C)
    data_std = data_std.reshape(1, 1, -1)
    data_mean = data_mean.reshape(1, 1, -1)
    return (kps * data_std) + data_mean

def parse_audio(textgrid_file):
    #TODO: 暂时用这种简单的逻辑进行实验
    words=['but', 'as', 'to', 'that', 'with', 'of', 'the', 'and', 'or', 'not', 'which', 'what', 'this', 'for', 'because', 'if', 'so', 'just', 'about', 'like', 'by', 'how', 'from', 'whats', 'now', 'very', 'that', 'also', 'actually', 'who', 'then', 'well', 'where', 'even', 'today', 'between', 'than', 'when']
    txt=tg.TextGrid.fromFile(textgrid_file)
    
    total_time=int(np.ceil(txt.maxTime))
    code_seq=np.zeros(total_time)#初始全是0
    
    word_level=txt[0]
    
    for i in range(len(word_level)):
        start_time=word_level[i].minTime
        end_time=word_level[i].maxTime
        mark=word_level[i].mark
        
        if mark in words:
            start=int(np.round(start_time))
            end=int(np.round(end_time))
            
            if start >= len(code_seq) or end >= len(code_seq):
                code_seq[-1] = 1
            else:
                code_seq[start]=1
    
    return code_seq