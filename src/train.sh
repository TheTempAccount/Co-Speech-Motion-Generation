gpu=$1
model_name=$2
r_weight=$3
exp_name=$4
content_dim=$5
embed_dim=$6
noise_dim=$7

python scripts/train.py --gpu ${gpu} \
                        --exp_name ${exp_name} \
                        --data_root ../../pose_dataset/videos/ \
                        --speakers Bill_Gates Dan_Ariely daniel_susskind Keller_Rinaudo \
                        --model_name ${model_name} \
                        --epochs 50 \
                        --save_every 2  \
                        --print_every 100 \
                        --normalization \
                        --save_dir \
                        ../../experiments \
                        --aud_decoding \
                        --r_loss_weight ${r_weight} \
                        --kl_loss_weight 0.001 \
                        --recon_input \
                        --embed_dim ${embed_dim} \
                        --content_dim ${content_dim} \
                        --noise_dim ${noise_dim} \
                        --seq_enc_hidden_size 1024 \
                        --seq_dec_hidden_size 1024 \
                        --latent_enc_fc_size 1024 \
                        --latent_enc_num_layers 3 \
                        --latent_dec_num_layers 3 \
                        --T_layer_norm \
                        --rnn_cell gru \
                        --interaction concat \
                        --shell_cmd train.sh \
                        --residual \
                        --graph_type part \
                        --share_weights \
                        --context_info