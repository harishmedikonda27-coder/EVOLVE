import urllib.request
import json
import time

try:
    req = urllib.request.Request('http://127.0.0.1:8000/runs', 
        data=b'{"mode":"research", "seed":42, "experiment_type":"evolve", "task_sequence":["env_a"]}', 
        headers={'Content-Type': 'application/json'})
    res = urllib.request.urlopen(req)
    run_data = json.loads(res.read().decode())
    print('Created run:', run_data['run_id'])
    
    req2 = urllib.request.Request('http://127.0.0.1:8000/runs/' + run_data['run_id'] + '/start', method='POST')
    res2 = urllib.request.urlopen(req2)
    print('Start response:', json.loads(res2.read().decode()))
    
    time.sleep(2)
    print('Successfully started and waited 2s. No crash!')
except Exception as e:
    print('Error:', e)
