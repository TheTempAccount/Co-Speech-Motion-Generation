audio_file=$1
textgrid_file=$2
model_path=../pose_dataset/ckpt/ckpt-99.pth
config_file=../pose_dataset/ckpt/freeMo.json


python scripts/demo.py --exp_name demo \
                       --speakers None \
                       --infer \
                       --model_path ${model_path} \
                       --initial_pose_file ../sample_initial_pose/bill_initial.npy \
                       --audio_file ${audio_file} \
                       --textgrid_file ${textgrid_file} \
                       --config_file ${config_file} \
