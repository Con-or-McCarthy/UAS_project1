import argparse
import os

from datetime import datetime
from data_processing import process_file
from inference import run_inference

def main(
        filenames: list,
        air_pressure: bool,
        sensor_locations: str
):
    # iterate through list of exp ids
    for exp_id in filenames:
        # preprocess data for that exp id (if not done already)
        cwd = os.getcwd()
        t_npy_loc = os.path.join(cwd,"processed_data",exp_id,"time.npy")
        x_npy_loc = os.path.join(cwd,"processed_data",exp_id,"X.npy")
        if not os.path.isfile(x_npy_loc):
            print(f"Preprocessing exp {exp_id}")
            process_file(exp_id)
        else:
            print(f"exp {exp_id} has already been preprocessed and saved to {x_npy_loc}")

        # perform inference on this exp id and save results to /model_outputs/
        model_pick=sensor_locations
        if air_pressure:
            model_pick += "_air"
        else:
            model_pick += "_noair"
        now = datetime.now()
        formatted_time = now.strftime("%y%m%d%H%M")
        output_filename = "_".join([exp_id,model_pick,formatted_time])
        save_output_loc = os.path.join(cwd,"model_outputs",output_filename+".csv")
        print(f"Running inference on exp {exp_id}")
        run_inference(model_pick=model_pick,
                      path_to_x=x_npy_loc,
                      path_to_t=t_npy_loc,
                      path_to_output=save_output_loc)

    print("Finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-filename", nargs="+", help="Experiment ID", required=True)
    parser.add_argument("-air_pressure", type=bool, help="Include air pressire or not", default=True)
    parser.add_argument("-sensor_locations", type=str, help="Experiment ID", default="both")
    args = parser.parse_args()

    if args.filename is None:
        print("No filename supplied")
        raise ValueError
    
    main(filenames=args.filename,air_pressure=args.air_pressure,sensor_locations=args.sensor_locations)