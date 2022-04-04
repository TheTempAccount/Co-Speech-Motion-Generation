from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--save_dir', default='experiments', type=str)
    parser.add_argument('--exp_name', required=True, type=str)
    parser.add_argument('--speakers', required=True, nargs='+')
    parser.add_argument('--seed', default=1, type=int)
    
    parser.add_argument('--use_template', action='store_true')
    parser.add_argument('--template_length', default=0, type=int)

    parser.add_argument('--infer', action='store_true')
    parser.add_argument('--model_path', type=str)    
    parser.add_argument('--same_initial', action='store_true')

    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--pretrained_pth', default=None, type=str)
    parser.add_argument('--style_layer_norm', action='store_true')
    

    parser.add_argument('--config_file', default='./config/style_gestures.json', type=str)
    return parser