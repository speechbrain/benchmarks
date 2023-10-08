import json 
import os
from scipy.io import loadmat,savemat
def create_json(json_file, Data_address,json_path):
  json_dict = {}
  for roots, d_names, f_names in os.walk(Data_address):

    for f_name in f_names:
        print(roots,f_name)
        loaded_mat_dict = loadmat(os.path.join(roots,f_name))
        rf_data_path = os.path.join(roots,f_name)
        attenuation = loaded_mat_dict['my_att']

            # Create entry for this utterance
        json_dict[f_name] = {
                "rf_data": rf_data_path,
                "attenuation": float(attenuation[0][0]),
        }
    
    print (json_dict)
    # Writing the dictionary to the json file
    with open(os.path.join(json_path,json_file), mode="w") as json_f:
      json.dump(json_dict,json_f , indent=2)


if __name__ == "__main__":
  create_json('train.json', '../DataSet/train','./json_folder')
  create_json('valid.json', '../DataSet/valid','./json_folder')
  create_json('test.json', '../DataSet/test','./json_folder')
#create_json('train.json', train_files)
#create_json('valid.json', valid_files)
#create_json('test.json', test_files)       