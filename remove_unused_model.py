import os
import shutil
for model_name in os.listdir('models'):
    if not os.path.exists(os.path.join('models', model_name, f'best-{model_name}.pth')):
        if model_name != 'models.json':
            shutil.rmtree(os.path.join('models', model_name),
                          ignore_errors=True)
