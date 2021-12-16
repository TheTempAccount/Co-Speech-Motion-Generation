# Free-form Body Motion Generation from Speech (freeMo)

The repo for our work "Free-form Body Motion Generation from Speech" (paper comming soon).

### Video Demo
[![](https://res.cloudinary.com/marcomontalbano/image/upload/v1639639637/video_to_markdown/images/youtube--HWZv0udfkAQ-c05b58ac6eb4c4700831b2b3070cd403.jpg)](https://youtu.be/HWZv0udfkAQ "")

### Directory Structure

    |--src //source code
    |   |--backup //A runnable implementation of our model
    |   |
    |   |--repro_nets //other baseline models will be updated soon
    |   |      |--freeMo_paper.py //same model structure as backup
    |   |      |--audio2body.py //Audio to Body Dynamics
    |   |      |--speech2gesture.py //Speech2Gesture & SpeechDrivenTemplates
    |   |
    |   |--nets //Some modifications to *repro_nets* for further experiments
    |   |   |--freeMo_old.py //Similar macro structure as backup, with some details are different
    |   |   |--freeMo.py //Some different design choices to freeMo_old
    |   |   ...
    |   |
    |   |--visualise 
    |   |--data_utils
    |   |--scripts //train.py & infer.py
    |   |--trainer //args and trainer

- [x] code 
- [ ] data 

### Inference

    python src/backup/generate_on_audio.py --model_name test --model_path pretrained_models/ckpt-48.pt --initial_pose sample_initial_pose/bill_initial.npy --audio_path sample_audio/clip000040_TWeBl1yQ1oI.wav --textgrid_path sample_audio/clip000040_TWeBl1yQ1oI.TextGrid --audio_decoding --normalization --noise_size 512 --sample_index 0,10,20

The result will be different every time you run the script.
The results will be saved in "results/[model_name]", including the json file of 64 randomly generated motion sequences and the visualized videos. 

For explanation of the flags, see [here](src/backup/).

### Training
The *train.sh* will be usable once I upload the data. You can also modify the code to use publicly avaliable gesture dataset.