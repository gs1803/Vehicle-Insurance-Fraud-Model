import os
import json

def extract_trial_info(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    hyperparameters = data.get('hyperparameters', {}).get('values', {})
    num_dense_layers = hyperparameters.get('num_dense_layers', None)
    
    record = {
        'num_dense_layers': num_dense_layers,
        'learning_rate': hyperparameters.get('learning_rate', None),
    }
    
    for i in range(num_dense_layers):
        record[f'dense_units_{i}'] = hyperparameters.get(f'dense_units_{i}', None)
        record[f'dropout_{i}'] = hyperparameters.get(f'dropout_{i}', None)

    metrics = data.get('metrics', {}).get('metrics', {})
    record['train_accuracy'] = metrics.get('accuracy', {}).get('observations', [{}])[0].get('value', [None])[0]
    record['val_accuracy'] = metrics.get('val_accuracy', {}).get('observations', [{}])[0].get('value', [None])[0]
    record['train_f1_score'] = metrics.get('f1_score', {}).get('observations', [{}])[0].get('value', [None])[0]
    record['val_f1_score'] = metrics.get('val_f1_score', {}).get('observations', [{}])[0].get('value', [None])[0]
    record['train_loss'] = metrics.get('loss', {}).get('observations', [{}])[0].get('value', [None])[0]
    record['val_loss'] = metrics.get('val_loss', {}).get('observations', [{}])[0].get('value', [None])[0]

    return record

def get_all_trials(logs_dir):
    """Parse through all Keras Tuner trial folders and extract data."""
    records = []
    
    for root, dirs, files in os.walk(logs_dir):
        for file in files:
            if file.endswith('trial.json'):
                json_path = os.path.join(root, file)
                trial_record = extract_trial_info(json_path)
                
                records.append(trial_record)
    
    return records