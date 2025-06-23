import argparse
import os

from parso import parse

from easydict import EasyDict as edict

def get_args():
    parser = argparse.ArgumentParser(description="Training options")
    # basic parameters
    parser.add_argument('--dataset', choices=['nuscenes', 'argoverse'],
                        default='nuscenes', help='dataset to train on')
    parser.add_argument("--dataroot", type=str, default="/workspace/dataset/nuscenes/mini",
                        choices=[
                            '/workspace/dataset/nuscenes/mini',
                            '/workspace/dataset/nuscenes/trainval'],
                        help="Path to the nuscenes dataset")
    parser.add_argument("--nuscenes_version", type=str, default="v1.0-mini",
                        choices=[
                            'v1.0-mini',
                            'v1.0-trainval'],
                        help="nuscenes version")
    parser.add_argument("--label_root", type=str, default="/workspace/dataset/nuscenes/map-labels-mini",
                        choices=[
                            '/workspace/dataset/nuscenes/map-labels-mini',
                            '/workspace/dataset/nuscenes/map-labels-v1.2'],
                        help="Path to the labels folder")
    # parser.add_argument("--road_layout_root", type=str, default="/workspace/dataset/nuscenes/map-labels-mini-roadLayout",
    #                     choices=[
    #                         '/workspace/dataset/nuscenes/map-labels-mini-roadLayout',
    #                         '/workspace/dataset/nuscenes/map-labels-v1.2-roadLayout'],
    #                     help="Path to the road layout labels folder")
    parser.add_argument("--road_layout_root", type=str, default="/workspace/dataset/nuscenes/map-labels-mini-roadLayout-multicls",
                        choices=[
                            '/workspace/dataset/nuscenes/map-labels-mini-roadLayout-multicls',
                            '/workspace/dataset/nuscenes/map-labels-v1.2-roadLayout-multicls'],
                        help="Path to the road layout labels folder")
    parser.add_argument("--train_csv", type=str, default="/workspace/dataset/nuscenes/seq_bev_dataset/csv_files/train.csv",
                        help="Path to the train csv file")
    parser.add_argument("--val_csv", type=str, default="/workspace/dataset/nuscenes/seq_bev_dataset/csv_files/val.csv",
                        help="Path to the val csv file")
    parser.add_argument("--test_csv", type=str, default="/workspace/dataset/nuscenes/seq_bev_dataset/csv_files/test.csv",
                        help="Path to the test csv file")

    parser.add_argument("--pretrained_backbone", type=bool, default=False,
                        help="if load the pretrained weight for the backbone")
    parser.add_argument("--deeplabv3Plus_model_path", type=str, default="./pretrained_weight/deeplab_mobilenetv2.pth",
                        help="pretrained weight for deeplabv3_plus")
    parser.add_argument("--downsample_factor", type=int, default=8,
                        choices=[8, 16],
                        help="downsample_factor for the backbone network mobilenetv2")


    parser.add_argument("--save_path", type=str, default="./models/",
                        help="path to save models")
    parser.add_argument("--load_weights_folder", type=str, default="",
                        help="path to a pretrained model used for initialization")  # 路径写到.pth为止
    parser.add_argument("--model_name", type=str, default="SeqBEV",
                        help="Model Name with specifications")
    parser.add_argument("--ext", type=str, default='png',
                        help="File extension of the images")
    parser.add_argument('--height', type=int, default=450,
                        help="Image height")
    parser.add_argument('--width', type=int, default=800,
                        help="Image width")
    parser.add_argument("--occ_map_size", type=int, default=200,
                        help="size of topview occupancy map")
    parser.add_argument("--num_class", type=int, default=7,
                        help="Number of classes")
    parser.add_argument("--loss_type", type=str, default="focal",
                        choices=['ce', 'focal'],
                        help="loss type of bev prediction")
    parser.add_argument("--roadLayout_loss_type", type=str, default="focal",
                        choices=['ce', 'focal'],
                        help="loss type of roadLayout prediction")
    parser.add_argument("--if_save_img", type=bool, default=True,
                        help="Whether to save test img")
    parser.add_argument("--save_img_path", type=str, default="./saved_img/",
                        help="path to save test images")
    

    # Temporal network module parameters
    parser.add_argument("--base_model", type=str, default="resnet101",
                        help="the basic model for the encoder")
    parser.add_argument("--num_squence", type=int, default=3,
                        help="Number of image in the sequence")
    parser.add_argument("--shift_div", type=int, default=24,
                        help="Number of the division in temporal dimension, must larger than 5")
    parser.add_argument("--shift_place", type=str, default="block",
                        choices=['block', 'blockres'],
                        help="the basic model for the encoder")
    


    # data argument
    parser.add_argument("--stack", type=bool, default=True,
                        help="Whether to stack the sequence image into one tensor")
    parser.add_argument("--hflip", type=bool, default=True,
                        help="Whether to use horizontal flips for data augmentation")
    parser.add_argument("--hold_out_calibration", type=bool, default=False,
                        help="Hold out portion of train data to calibrate on")
    
    # training parameters
    parser.add_argument("--freeze_train", type=bool, default=False,
                        help="If freeze the backbone")
    parser.add_argument("--Freeze_Epoch", type=int, default=5,
                        help="the epoch to freeze backbone during training")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Mini-Batch size")
    parser.add_argument("--val_batch_size", type=int, default=8,
                        help="Mini-Batch size for val")
    parser.add_argument("--lr", type=float, default=5e-4,
                        help="learning rate") # basic_lr
    parser.add_argument("--TS_lr", type=float, default=1e-5,
                        help="learning rate") # 时空特征提取模块
    parser.add_argument("--trans_lr", type=float, default=1e-4,
                        help="learning rate") # objects trainsformer模块
    parser.add_argument("--lr_steps", default=[5, 10], type=float, nargs="+",
                        metavar='LRSteps', help='epochs to decay learning rate by 10')
    parser.add_argument("--weight_decay", '--wd', default=5e-4, type=float,
                        metavar='W', help='weight decay (defualt: 5e-4)')
    parser.add_argument("--momentum", '--m', default=0.9, type=float,
                        metavar='M', help='momentum (defualt: 0.9)')
    parser.add_argument("--scheduler_step_size", type=int, default=5,
                        help="step size for the both schedulers")
    parser.add_argument("--start_epoch", type=int, default=0,
                        help="the start epoch")
    parser.add_argument("--num_epochs", type=int, default=50,
                        help="Max number of training epochs")
    parser.add_argument("--log_frequency", type=int, default=50,
                        help="Log files every x epochs")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of cpu workers for dataloaders")
    parser.add_argument('--log_root', type=str, default=os.getcwd() + '/log')
    parser.add_argument('--model_save', type=bool, default=True)
    parser.add_argument('--resume', default=None, 
                        help='path to an experiment to resume')

    # transformer parameters
    parser.add_argument("--hidden_dim", type=int, default=256,
                        help='Size of the embeddings (dimension of the transformer)')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout applied in the transformer')
    parser.add_argument('--nheads', type=int, default=8,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=50, type=int,
                        help="Number of query slots")
    parser.add_argument('--dim_feedforward', type=int, default=2048,
                        help='Intermediate size of the feedforward layers in the transformer blocks')
    parser.add_argument('--enc_layers', type=int, default=2,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', type=int, default=2,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--pre_norm', action='store_true', 
                        help='whether to pre norm tgt before calculate attention in decoder')
    parser.add_argument('--position_embedding', type=str, default='sine', 
                        choices=['sine', 'learned'],
                        help='Type of positional embedding to use on top of the image features')

    # loss weight parameters
    parser.add_argument('--dice_weight', type=float, default=1.0,
                        help='use ce loss and dice loss to calculate the bev loss at the same time, determine the weight of dice loss when add those two losses together')
    parser.add_argument('--layout_weight', type=float, default=1,
                        help='use bev loss and layout loss to calculate the total loss, determine the weight of layout loss when add those two losses together')



    
    # val parameters
    parser.add_argument("--pretrained_path", type=str, default="./pretrain_models",
                        help="Path to the pretrained model")
    parser.add_argument("--out_dir", type=str, default="./output")
    # parser.add_argument('--options', nargs='*', default=[],
    #                     help='list of addition config options as key-val pairs')


    configs = edict(vars(parser.parse_args()))
    #print("in opt.py, configs: ", configs)
    config_list = []
    for key in configs.keys():
        config_list.append(key)
        config_list.append(configs[key])

    print("config_list: ", config_list)
    #return configs, config_list
    return configs


