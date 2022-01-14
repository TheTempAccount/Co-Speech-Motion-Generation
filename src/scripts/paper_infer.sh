python scripts/infer.py --gpu 1 \
                        --exp_name paper_model \
                        --data_root ../../pose_dataset/videos/ \
                        --speakers Keller_Rinaudo \
                        --model_name freeMo_paper \
                        --epochs 50 \
                        --save_every 2  \
                        --print_every 100 \
                        --normalization \
                        --save_dir \
                        ../../experiments \
                        --aud_decoding \
                        --recon_input \
                        --embed_dim 1024 \
                        --content_dim 512 \
                        --noise_dim 512 \
                        --seq_enc_hidden_size 1024 \
                        --seq_dec_hidden_size 1024 \
                        --latent_enc_fc_size 1024 \
                        --latent_enc_num_layers 1 \
                        --latent_dec_num_layers 1 \
                        --T_layer_norm \
                        --rnn_cell lstm \
                        --interaction add \
                        --infer \
                        --model_path ../pretrained_models/ckpt-48.pt \
                        --aud_feat_win_size 100 \
                        --feat_method mfcc \
                        --aud_feat_dim 13 \

