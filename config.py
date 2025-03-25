
config = {
    
    'base_dir': '/kaggle/working/Code', #for colab : '/content/Code'
    # 'last_preprocessing' : 'gait_cycles_iigc',
    'preprocess' : ['transform',
                    'augment',
                    'gait_cycles',
                    #gait_cycles_iigc,
                    'gait_features',
                    'event_features'],

    'drive_checkpoint_path' : '/content/drive/My Drive/trained_gait_model_checkpoints',

    'training' : {
        'nhead':1,
        'num_encoder_layers':1,
        # 'max_len' : 2048,
        # 'd_model : 256,
        'rope' : False,
        'contrastive' : False,
        # 'contrastive_weight' : 0.5,
        'k_fold' : False,
        'epochs' : 60,
    #    'cls_head_hidden_layers': [256, 128]
    },

}

# rope : True/False
# contrastive: True/False 
# k_fold: True/False
# base_dir: '/content/Code' for colab, '/kaggle/working/Code' for kaggle
# preprocess: ['transform','augment', 'gait_cycles', 'gait_cycles_iigc', 'gait_features', 'event_features']
# drive_checkpoint_path: '/content/drive/My Drive/trained_gait_model_checkpoints' for colab, '/kaggle/working/trained_gait_model_checkpoints' for kaggle