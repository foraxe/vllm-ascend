"""Full HTTP serving correctness and matched streaming latency workload."""
import argparse
import concurrent.futures
import json
import hashlib
import random
import statistics
import time
from pathlib import Path
import requests


def chat(url, prompt, *, stream, output_tokens=128, ignore_eos=False):
    body={"model":"deepseek-v41","messages":[{"role":"user","content":prompt}],
          "temperature":0,"seed":1234,"max_tokens":output_tokens,"stream":stream,
          "return_token_ids":True}
    if ignore_eos: body["ignore_eos"]=True
    if stream: body["stream_options"]={"include_usage":True}
    started=time.perf_counter()
    r=requests.post(url+"/v1/chat/completions",json=body,stream=stream,timeout=300)
    r.raise_for_status()
    if not stream:
        data=r.json()
        return {"prompt":prompt,"elapsed_s":time.perf_counter()-started,"response":data}
    first,last=None,None
    content,reasoning=[],[]
    token_ids=[]
    usage={}
    for line in r.iter_lines(chunk_size=None):
        if not line.startswith(b"data: "): continue
        raw=line[6:]
        if raw==b"[DONE]": break
        value=json.loads(raw)
        if value.get("error"): raise RuntimeError(value["error"])
        if value.get("usage"): usage=value["usage"]
        for choice in value.get("choices",[]):
            delta=choice.get("delta",{})
            text=delta.get("content") or ""
            thought=delta.get("reasoning_content") or delta.get("reasoning") or ""
            delta_ids=choice.get('token_ids') or []
            if delta_ids:
                now=time.perf_counter()
                first=now if first is None else first
                last=now
                token_ids.extend(delta_ids)
            content.append(text);reasoning.append(thought)
    ended=time.perf_counter()
    count=usage.get("completion_tokens",0)
    if first is None or count<1: raise RuntimeError(f"empty streaming result: {usage}")
    if len(token_ids)!=count: raise RuntimeError(f'Streamed token count {len(token_ids)} != usage {count}')
    return {"prompt":prompt,"e2e_s":ended-started,"ttft_s":first-started,
            "tpot_s":((last-first)/(count-1)) if count>1 else None,
            "usage":usage,"token_ids":token_ids,"content":"".join(content),"reasoning":"".join(reasoning)}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--url",default="http://127.0.0.1:8000")
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--baseline",type=Path)
    p.add_argument("--accuracy-only",action="store_true")
    p.add_argument("--nonce",default="A")
    p.add_argument('--pattern',choices=['varied','repeated'],default='varied')
    p.add_argument('--input-sizes',nargs='+',type=int,default=[128,1024])
    p.add_argument('--concurrencies',nargs='+',type=int,default=[1,4,8])
    p.add_argument('--requests',type=int,default=16)
    a=p.parse_args()
    prompts=["Reply with exactly OK.","What is 2+2? Answer with one number.",
             "Name the capital of France. Answer with one word.",
             "Count from 1 to 5 separated by commas.",
             "Translate 'good morning' into Chinese. Only the translation.",
             "Explain why the sky is blue in one short sentence."]
    result={"accuracy":[],"workloads":[],"token_ids_requested":True,
            "prompt_pattern":a.pattern,"nonce":a.nonce,
            "requests_per_case":a.requests,
            "benchmark_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "timing":"HTTP arrival of token-ID deltas; TPOT is average over completion tokens, not per-token intervals"}
    for prompt in prompts:
        row=chat(a.url,prompt,stream=False)
        result["accuracy"].append(row)
        choice=row['response']['choices'][0]
        print('ACCURACY',{'prompt':prompt,'content':choice['message']['content'],
                          'finish_reason':choice['finish_reason'],'tokens':row['response']['usage']['completion_tokens']},flush=True)
    if a.baseline:
        base=json.loads(a.baseline.read_text())
        if [x['prompt'] for x in base['accuracy']] != prompts:
            raise ValueError('Baseline prompt coverage differs from this run')
        result["accuracy_matches"]=[x["response"]["choices"][0]["message"] == y["response"]["choices"][0]["message"]
                                    for x,y in zip(base["accuracy"],result["accuracy"])]
        result["content_matches"]=[x["response"]["choices"][0]["message"]["content"] == y["response"]["choices"][0]["message"]["content"]
                                   for x,y in zip(base["accuracy"],result["accuracy"])]
        print("ACCURACY_MATCHES",result["accuracy_matches"],flush=True)
        result['finish_reason_matches']=[x['response']['choices'][0]['finish_reason']==y['response']['choices'][0]['finish_reason']
                                        for x,y in zip(base['accuracy'],result['accuracy'])]
        if base.get('token_ids_requested'):
            result['token_id_matches']=[x['response']['choices'][0]['token_ids']==y['response']['choices'][0]['token_ids']
                                       for x,y in zip(base['accuracy'],result['accuracy'])]
            print('TOKEN_ID_MATCHES',result['token_id_matches'],flush=True)
    a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    if a.baseline and not all(result['accuracy_matches']+result['finish_reason_matches']+result.get('token_id_matches',[])):
        raise RuntimeError("Generated message or finish reason differs from baseline; results saved, benchmark not started")
    if a.accuracy_only: return
    for input_size in a.input_sizes:
        for concurrency in a.concurrencies:
            text=("A cache keeps frequently used data near the processor. " * (input_size//10))
            # Unique leading IDs prevent reuse of the long benchmark prefix.
            def run(i):
                words='cache core data page disk file block node link rank graph batch task queue stream table row key value scale mask token model head layer gate index shape clock thread load store read write send get put map host chip bank byte bit port wire line buffer slot pool view copy count size state route hash chunk shard group memory time work tree ring mesh code test'.split()
                varied=' '.join(random.Random(input_size*10000+concurrency*100+i).choices(words,k=input_size))
                body=varied if a.pattern=='varied' else text
                prompt=f"Request {a.nonce}-{input_size}-{concurrency}-{i}. "+body+" Discuss how these computing terms relate in detail."
                return chat(a.url,prompt,stream=True,output_tokens=64,ignore_eos=True)
            with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
                list(pool.map(run,range(-max(4,concurrency),0)))
            begin=time.perf_counter()
            with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
                rows=list(pool.map(run,range(a.requests)))
            seconds=time.perf_counter()-begin
            row={"input_text_target":input_size,"concurrency":concurrency,"requests":rows,
                 "duration_s":seconds,"output_tokens_per_second":sum(x["usage"]["completion_tokens"] for x in rows)/seconds,
                 "median_ttft_ms":statistics.median(x["ttft_s"] for x in rows)*1000,
                 "median_tpot_ms":statistics.median(x["tpot_s"] for x in rows if x["tpot_s"] is not None)*1000,
                 "median_e2e_s":statistics.median(x["e2e_s"] for x in rows)}
            result["workloads"].append(row)
            print("BENCH",{k:v for k,v in row.items() if k!="requests"},flush=True)
            a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")


if __name__=="__main__": main()
