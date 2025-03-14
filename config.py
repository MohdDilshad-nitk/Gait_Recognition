
config = {
    
    'base_dir': '/kaggle/working/Gait_Recognition-main', #for colab : '/content/Code'
    'preprocess' : ['transform',
                    'augment',
                    'gait_cycles',
                    'gait_features',
                    'event_features'],

    'drive_checkpoint_path' : '/content/drive/My Drive/trained_gait_model_checkpoints',

    'training' : {
        'nhead':1,
        'num_encoder_layers':1,
        'rope' : False,
        'contrastive' : False,
        # 'contrastive_weight' : 0.5,
        'k_fold' : False,
        'epochs' : 60
    },

}

# rope : True/False, contrastive: True/False * {

# ['transform','gait_cycles','gait_features']
# ['transform','gait_cycles','gait_features','event_features']

# ['transform','augment']
# ['transform','augment', 'gait_cycles']
# ['transform','augment', 'gait_cycles', 'gait_features']
# ['transform','augment', 'gait_cycles', 'gait_features', 'event_features']

# ['transform', 'gait_cycles','augment']
# ['transform', 'gait_cycles','augment', 'gait_features']
# ['transform', 'gait_cycles','augment', 'gait_features', 'event_features']

# }