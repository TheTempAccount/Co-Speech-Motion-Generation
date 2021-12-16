# Free-form Co-Speech Motion Generation

### Inference

    python generate_on_audio.py --model_name test --model_path ../../pretrained_models/ckpt-48.pt 
    --initial_pose ../../sample_initial_pose/bill_initial.npy 
    --audio_path ../../sample_audio/clip000040_TWeBl1yQ1oI.wav 
    --textgrid_path ../../sample_audio/clip000040_TWeBl1yQ1oI.TextGrid 
    --audio_decoding --normalization --noise_size 512 --sample_index 0 10 20

The result will be different every time you run the script.
The results will be saved in "results/[model_name]", including the json file of 64 randomly generated motion sequences and the visualized videos. 

* Explanation of the flags
    - model_name: the name of the folder to save the results.
    - model_path: the pretrained model path.
    - initial_pose: the initial posture of the generated sequence. Can be arbitrarily specified as long as the format is correct. We provide some samples to use in sample_initial_pose.
    - audio_path: the input audio.
    - textgrid_path: the transcript of the input audio. We generate it using the [montreal forced aligner](https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner.git). If you do not have the textgrid file of you audio, you can also manually specify the *code_seq* in the script. The sample audio and textgrid file pairs can be found in sample_audio.
    - sample_index: the indexes of the generated motion sequences to be plotted as videos.
