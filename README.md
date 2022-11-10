# Free-form Co-Speech Gesture Generation

The repo for work "Free-form Co-Speech Gesture Generation"

- source code 
- data preparation (partially)

### Video Demo
[![](https://res.cloudinary.com/marcomontalbano/image/upload/v1639640143/video_to_markdown/images/youtube--Wb5VYqKX_x0-c05b58ac6eb4c4700831b2b3070cd403.jpg)](https://youtu.be/Wb5VYqKX_x0 "")

### Data & Pretrained model
Avaliable through 
- [Google Drive](https://drive.google.com/drive/folders/1v8_4agLNM2jmRuSKnflkdbEcKWZkdtka?usp=sharing)

<!-- - [Baidu Yun](https://pan.baidu.com/s/18aeNlFuUNHbavlJFeSMn-Q) 提取码: 1vji -->

Unzip everything in *pose_dataset*, then change the *Data.data_root* in src/config/*.json. You should be seeing directory structure like this:

    pose_dataset
    |-videos
    |   |-Speaker_A
    |   |-Speaker_B
    |   |-...
    |   |-test_audios
    |-ckpt

The rest of the data will be updated after I finish checking the annotations.

### Inference
Generated gestures for an example audio clip:

    bash demo.sh ../sample_audio/clip000040_ozfGHONpdTA.wav ../sample_audio/clip000040_ozfGHONpdTA.TextGrid

Visualise the generated motions:

    bash visualse.sh

Generate gestures for a speaker in test_audios:

    cd src
    bash infer.sh  \
            pose_dataset/ckpt/ckpt-99.pth \
            pose_dataset/ckpt/freeMo.json \
            <post_fix> \
            <speaker_name>

The results will be saved as "pose_dataset/videos/test_audios/<speaker_name>/*_<post_fix>.json", including the json file of 64 randomly generated gesture sequences for every audio. 

To visualise the results, run

    bash visualise/visualise_all.sh <speaker_name> <post_fix>

Remember to change the file path in all files.

### Training
    
    bash train.sh

For any problem, please let us know.
