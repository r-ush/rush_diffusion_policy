import yaml

with open('hdf5_force_rviz.rviz', 'r') as f:
    config = yaml.safe_load(f)

for disp in config['Visualization Manager']['Displays']:
    if disp['Name'] == 'TF':
        disp['Enabled'] = True
        disp['Value'] = True
        disp['Marker Scale'] = 0.5
        disp['Show Arrows'] = True
        disp['Show Axes'] = True
        disp['Show Names'] = False
        disp['Frames'] = {
            'All Enabled': False,
            'left_thumb_link4': {'Value': True},
            'left_index_link4': {'Value': True},
            'left_middle_link4': {'Value': True},
        }

with open('hdf5_force_rviz.rviz', 'w') as f:
    yaml.dump(config, f, default_flow_style=False)
