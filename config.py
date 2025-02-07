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
        'k_fold' : False,
        'epochs' : 60
    }
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