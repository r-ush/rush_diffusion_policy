import yaml

with open('hdf5_force_rviz.rviz', 'r') as f:
    config = yaml.safe_load(f)

for disp in config['Visualization Manager']['Displays']:
    if disp['Name'] == 'TF':
        disp['Frames'] = {
            'All Enabled': False,
            'left_thumb_sensor': {'Value': True},
            'left_index_sensor': {'Value': True},
            'left_middle_sensor': {'Value': True},
        }

with open('hdf5_force_rviz.rviz', 'w') as f:
    yaml.dump(config, f, default_flow_style=False)
