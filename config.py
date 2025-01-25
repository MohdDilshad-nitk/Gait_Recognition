# config = {
#     'transform' : True,
#     'augment' : False,
#     'gait_cycles' : True,
#     'gait_features' : True,
#     'event_features' : True,
#     'rope' : True,
#     'contrastive' : True,
#     'epochs' : 60
# }


config = {
    
    
    'preprocess' : ['transform',
                    'augment',
                    'gait_cycles',
                    'gait_features',
                    'event_features'],

    'training' : {
        'nhead':1,
        'num_encoder_layers':1,
        'rope' : True,
        'contrastive' : True,
        'epochs' : 60
    }
}