import json
import os
from django.conf import settings

def contact_data(request):
    json_path = os.path.join(settings.BASE_DIR, 'contact_info.json')
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        data = {}
    return {'contact_data': data}
