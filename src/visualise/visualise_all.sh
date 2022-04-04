speaker=$1
exp_name=$2

for wav_file in `ls pose_dataset/videos/test_audios/${speaker}/*.wav`
do
    echo ${wav_file}
    abs_name=${wav_file##*/}
    res_abs_name=${abs_name%%.*}
    clip_name=${res_abs_name::10}
    len=`expr ${#res_abs_name}-11`
    vid_name=${res_abs_name:11:${len}}
    res_fn=/home/jovyan/pose_dataset/videos/test_audios/${speaker}/${res_abs_name}_${exp_name}.json
    python visualise/visualise_generation_res.py --save_dir ../../experiment/videos/${speaker}/${exp_name}/${res_abs_name} --res_fn ${res_fn} --wav_file ${wav_file} --sample_index 0 \
    --clip_pth /home/jovyan/pose_dataset/videos/${speaker}/clips/${vid_name}/images/half/val/${clip_name}/
done