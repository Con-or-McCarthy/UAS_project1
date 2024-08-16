import torch
import os 
import numpy as np
import pandas as pd

from model_architecture import sensor_based_IMU_fusion
from data_processing import code2label
from utils import downsample_X, reduce_array, separate_X_sensorwise


# Load requested model
def load_model(model_pick:str):
    cwd = os.getcwd()
    model_root = os.path.join(cwd,"saved_models")

    if model_pick == "both_air":
        model = init_model(
            num_IMU=2,
            has_pressure=True
        )
    elif model_pick == "both_noair":
        model = init_model(
            num_IMU=2,
            has_pressure=False
        )
    elif model_pick == "arm_air":
        model = init_model(
            num_IMU=1,
            has_pressure=True
        )
    elif model_pick == "arm_noair":
        model = init_model(
            num_IMU=1,
            has_pressure=False
        )
    elif model_pick == "back_air":
        model = init_model(
            num_IMU=1,
            has_pressure=True
        )
    elif model_pick == "back_noair":
        model = init_model(
            num_IMU=1,
            has_pressure=True
        )
    else:
        print(f"{model_pick} not a valid model name")
        raise ValueError

    path_to_model = os.path.join(model_root,model_pick+".mdl")
    model.load_state_dict(torch.load(path_to_model, weights_only=False))

    return model

def init_model(num_IMU: int,
               has_pressure: bool,
               my_device: str="cpu"
               ):
    model = sensor_based_IMU_fusion(
        num_IMU=num_IMU,
        has_pressure=has_pressure
    )
    model = model.to(my_device, dtype=torch.float)
    return model


# run inference on supplied file
def generate_labels(model, path_to_npy, model_pick):
    # load data
    X = np.load(path_to_npy, allow_pickle=True)
    # Remove columns we're not interested in
    X = reduce_array(X,model_pick)
    # Downsample to  <=> 30 Hz sampling rates
    X = downsample_X(X, window_size=300)
    X = torch.from_numpy(X)
    # separate inputs by sensor for inputting to model
    my_X = separate_X_sensorwise(X, model_pick)
    # run inference
    logits,_ = model(my_X)
    pred_y = torch.argmax(logits, dim=1)
    pred_y += 1 # adding one to match the labels in code2label
    max_confidences, _ = torch.max(logits, dim=1)

    return pred_y, max_confidences


# save outputs
def save_outputs(predictions,confidences,save_location):

    predictions_list = map_values(predictions, map_dict=code2label)
    confidences = confidences.detach().numpy()
    out_arr = np.column_stack((predictions_list,confidences))
    df = pd.DataFrame(out_arr,columns=["predictions","confidence"])
    df.to_csv(save_location, index=False)
    print(f"saved model outputs to {save_location}")

def map_values(in_tensor, map_dict):
    tensor_list = in_tensor.tolist()

    # Map the integers to strings using a list comprehension
    mapped_list = [map_dict[value] for value in tensor_list]

    return mapped_list



def run_inference(model_pick, path_to_npy, path_to_output):
    model = load_model(model_pick)
    pred_y, max_confidences = generate_labels(model, path_to_npy=path_to_npy, model_pick=model_pick)
    save_outputs(pred_y,max_confidences,path_to_output)




if __name__ == "__main__":
    # save_file_name = "_".join([exp_id,model_pick,timenow])
    run_inference(model_pick="both_air",path_to_npy="processed_data/20230628/X.npy",path_to_output="model_outputs/20230628_other_info.csv")