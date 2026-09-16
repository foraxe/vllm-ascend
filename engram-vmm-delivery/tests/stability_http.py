"""Short correctness/stability gate, not another throughput benchmark."""
import argparse,concurrent.futures,json,time
from pathlib import Path
import requests

p=argparse.ArgumentParser()
p.add_argument('--baseline',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--idle-seconds',type=int,default=20)
a=p.parse_args()
base=json.loads(a.baseline.read_text())
url='http://127.0.0.1:8000'
result={'status':'RUNNING','phases':[]}

def call(prompt,limited=False):
    body={'model':'deepseek-v41','messages':[{'role':'user','content':prompt}],
          'temperature':0,'seed':1234,'max_tokens':32 if limited else 128,
          'return_token_ids':True}
    if limited:body['ignore_eos']=True
    r=requests.post(url+'/v1/chat/completions',json=body,timeout=180)
    r.raise_for_status()
    data=r.json()
    if 'error' in data:raise RuntimeError(data['error'])
    ids=data['choices'][0]['token_ids']
    assert len(ids)==data['usage']['completion_tokens']>0
    if limited:assert len(ids)==32
    return {'prompt':prompt,'response':data}

def save(name,rows):
    result['phases'].append({'name':name,'rows':rows})
    a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print('PHASE_PASS',name,len(rows),flush=True)

def accuracy(name):
    rows=[call(x['prompt']) for x in base['accuracy']]
    for expected,actual in zip(base['accuracy'],rows):
        x,y=expected['response']['choices'][0],actual['response']['choices'][0]
        assert all(x[k]==y[k] for k in ('message','finish_reason','token_ids')),(name,actual['prompt'])
    save(name,rows)

accuracy('initial_accuracy')
# Long variable inputs are serial: concurrent long-input _pool_kernel is a
# known baseline-model failure, not an Engram lifecycle gate.
prompts=[f'Variable {i}. '+('cache table memory data '*n)+'Explain briefly.'
         for i,n in enumerate([1,4,32,256]*4)]
save('variable_serial',[call(p,True) for p in prompts])
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    rows=list(pool.map(lambda i:call(f'Concurrent {i}. '+('cache table memory data '*32)+'Explain.',True),range(16)))
save('short_concurrent',rows)
time.sleep(a.idle_seconds)
accuracy('after_idle_accuracy')
r=requests.get(url+'/health',timeout=10);r.raise_for_status()
result['status']='PASS'
result['completed_requests']=sum(len(p['rows']) for p in result['phases'])
a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print('STABILITY_PASS',result['completed_requests'],flush=True)
