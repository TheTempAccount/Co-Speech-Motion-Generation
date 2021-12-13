from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--save_dir', default='experiments', type=str)
    parser.add_argument('--exp_name', required=True, type=str)
    parser.add_argument('--data_root', required=True, type=str)
    parser.add_argument('--speakers', required=True, nargs='+')
    parser.add_argument('--model_name', choices=['freeMo'], default='freeMo', type=str)
    parser.add_argument('--shell_cmd', type=str)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--generator_learning_rate', default=1e-4, type=float)
    parser.add_argument('--discriminator_learning_rate', default=1e-4, type=float)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--save_every', default=2, type=int)
    parser.add_argument('--print_every', default=100, type=int)

    parser.add_argument('--normalization', action='store_true')
    parser.add_argument('--augmentation', action='store_true')

    parser.add_argument('--generate_length', default=25, type=int)
    parser.add_argument('--pre_pose_length', default=25, type=int)
    parser.add_argument('--pose_dim', default=108, type=int)
    parser.add_argument('--aud_dim', default=64, type=int)
    parser.add_argument('--max_gradient_norm', default=5, type=float)
    parser.add_argument('--seed', default=1, type=int)

    #用于freeMo的参数
    parser.add_argument('--embed_dim', default=512, type=int)
    parser.add_argument('--content_dim', default=256, type=int)
    parser.add_argument('--noise_dim', default=256, type=int)
    parser.add_argument('--kl_tolerance', default=0.02, type=float)
    parser.add_argument('--velocity_length', default=8, type=int)
    parser.add_argument('--keypoint_loss_weight', default=1, type=float)
    parser.add_argument('--recon_input_weight', default=1, type=float)
    parser.add_argument('--kl_loss_weight', default=0.001, type=float)
    parser.add_argument('--kl_start_weight', default=1e-5, type=float)
    parser.add_argument('--kl_decay_rate', default=0.99995, type=float)
    parser.add_argument('--vel_loss_weight', default=1, type=float)
    parser.add_argument('--r_loss_weight', default=1, type=float)
    parser.add_argument('--zero_loss_weight', default=0, type=float)


    parser.add_argument('--seq_enc_hidden_size', default=512, type=int)
    parser.add_argument('--seq_enc_num_layers', default=1, type=int)
    parser.add_argument('--seq_dec_hidden_size', default=512, type=int)
    parser.add_argument('--seq_dec_num_layers', default=1, type=int)
    parser.add_argument('--latent_enc_fc_size', default=512, type=int)
    parser.add_argument('--latent_enc_num_layers', default=3, type=int)
    parser.add_argument('--latent_dec_num_layers', default=3, type=int)
    parser.add_argument('--aud_kernel_size', default=7, type=int)
    parser.add_argument('--recon_input', action='store_true')
    parser.add_argument('--aud_decoding', action='store_true')
    parser.add_argument('--T_layer_norm', action='store_true')
    parser.add_argument('--interaction', default='add', type=str)
    parser.add_argument('--rnn_cell', default='gru', type=str)
    parser.add_argument('--bidirectional', action='store_true')
    
    #for inference
    parser.add_argument('--infer', action='store_true')
    parser.add_argument('--model_path', type=str)
    
    # args = parser.parse_args()
    return parser