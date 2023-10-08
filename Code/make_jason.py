import json 

def create_json(json_file, audiolist):
  json_dict = {}
  for audiofile in audiolist:
    
    # Getting info
    audioinfo = torchaudio.info(audiofile)# Your code here

    # Compute the duration in seconds.
    # This is the number of samples divided by the sampling frequency
    duration = audioinfo.num_frames / audioinfo.sample_rate # Your code here

    # Get digit Label by manipulating the audio path
    digit = audiofile.split('/')[-1][0] # Your code here (aim for 1 line)_

    # Get a unique utterance id
    uttid = audiofile.split('/')[-1] # Your code here (aim for 1 line)

    # Create entry for this utterance
    json_dict[uttid] = {
            "wav": audiofile,
            "length": duration,
            "digit": digit,
    }

    # Writing the dictionary to the json file
    with open(json_file, mode="w") as json_f:
      json.dump(json_dict, json_f, indent=2)



create_json('train.json', train_files)
create_json('valid.json', valid_files)
create_json('test.json', test_files)       