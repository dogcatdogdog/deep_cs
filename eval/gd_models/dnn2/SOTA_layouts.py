import numpy as np
import networkx as nx
import time
import json
import os
import joblib

# Keep the original internal model imports
from . import models
from . import graph_preprocessing as preprocess

def upSamplePos(pos2d, mask):
    N_max = mask.shape[0]
    upsampled_pos = np.zeros((N_max, 2))
    real_i = 0
    for i in range(N_max):
        if(mask[i] == 1):
            upsampled_pos[i] = pos2d[real_i]
            real_i +=1
    return upsampled_pos

def layoutIsNotConstant(pos, mask):
    N = pos.shape[0]
    first = None
    for i in range(N):
        if(mask[i] == 1):
            if(first is None):
                first = pos[i]
            else:
                if(first[0] != pos[i][0] or first[1] != pos[i][1]):
                    return True
    return False

class ILayout:
    def __init__(self):
        super().__init__()
        self.execTimes = []

    def get_start(self):
        start_timestamp = time.time()
        return start_timestamp
    
    def get_elapsed(self, start):
        end_timestamp = time.time()
        elapsed = end_timestamp - start
        ms_elapsed = elapsed * 1000
        return ms_elapsed

    def layout(self, DM, mask):
        raise NotImplementedError("This interface does not implement this method")

    def predict(self, sequence):
        preds = np.empty((len(sequence), sequence.N_max, 2))
        for i in range(len(sequence)):
            item = sequence[i]
            c_DM = item[0][sequence.model_inputs.index("DM")].squeeze(axis=0)
            c_mask = item[0][sequence.model_inputs.index("nodesMask")].squeeze(axis=0)
            preds[i] = self.layout(c_DM, c_mask)
        return preds     

class DNN2(ILayout):
    def __init__(self, h5path, jsonpath, N_max=-1):
        super().__init__()
        with open(jsonpath, "r") as f:
            self.modelinfos =  json.load(f)
        
        self.name = os.path.basename(h5path).split(".")[0]
        self.N_max = N_max
        self.modelinfos["model"] = models.build_model(
            self.N_max, 
            self.modelinfos["initial_features_vector_size"], 
            self.modelinfos["max_deg"]+1, 
            self.modelinfos["tsnet_loss"]
        )
        self.modelinfos["model"].load_weights(h5path)
        self.modelinfos["inputs"] = [i.name.split(":")[0].split("_")[1] for i in self.modelinfos["model"].inputs]
        self.modelinfos["topology_fn"] = getattr(preprocess, self.modelinfos["topology"])
        
        if("scalersPath" in self.modelinfos.keys()):
            self.modelinfos["scalers"] = joblib.load(self.modelinfos["scalersPath"])
        else:
            scalersPath = os.path.join(h5path.replace(os.path.basename(h5path), ""), "scalers.pkl")
            self.modelinfos["scalers"] = joblib.load(scalersPath)

    def layout(self, g):
        i=0
        max_deg = self.modelinfos["max_deg"]
        model_inputs = self.modelinfos["inputs"]
        nodes_features = self.modelinfos["features"]
        scalers = self.modelinfos["scalers"]
        
        startTime=self.get_start()
        inputs, time_preprocess = preprocess.graph2predictData(
            g, i, self.N_max, max_deg, model_inputs, nodes_features, 
            self.modelinfos["topology_fn"], swap=False, scalers=scalers
        )

        batched_input = [[] for _ in range(len(model_inputs))]
        for i in range(len(inputs)):
            batched_input[i].append(inputs[i])
        for i in range(len(batched_input)):
            batched_input[i] = np.array(batched_input[i])
        
        pred = self.modelinfos["model"](batched_input, training=False) 
        exec_time = self.get_elapsed(startTime)
        self.execTimes.append(exec_time + time_preprocess)
        
        if(type(pred) == np.ndarray):
            pred = pred.squeeze()
        else:
            pred = pred.numpy().squeeze()
        return pred

methods = [
    "DNN2"
]