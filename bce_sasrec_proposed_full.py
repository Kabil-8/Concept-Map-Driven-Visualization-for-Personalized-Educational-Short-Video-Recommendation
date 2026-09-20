"""BCE-SASRec: complete three-seed training and evaluation script.
Expected structure: processed/{splits,graph} with train/valid/test parquet files.
Run: python bce_sasrec_proposed_full.py --processed /path/to/processed --output /path/to/output
"""
from __future__ import annotations
import argparse, copy, gc, json, math, os, random, time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

@dataclass
class Config:
    seeds: tuple=(42,2026,3407); max_len:int=50; max_concepts_per_video:int=12
    hidden_dim:int=256; transformer_layers:int=2; attention_heads:int=4; feedforward_dim:int=512
    dropout:float=.1429; time_buckets:int=32; negatives:int=100; batch_size:int=128
    eval_batch_size:int=128; max_epochs:int=20; minimum_epochs:int=8
    early_stopping_patience:int=4; early_stopping_min_delta:float=1e-4
    learning_rate:float=5e-4; weight_decay:float=1e-6; concept_loss_weight:float=.1173
    completion_loss_weight:float=.05; label_smoothing:float=.03; gradient_clip:float=.5
    num_workers:int=2; ks:tuple=(5,10,20)

BEHAVIOUR_COLUMNS=['watched_seconds','playback_seconds','duration_seconds','completion_ratio','segment_count','engagement_weight']
LOG_BEHAVIOUR={'watched_seconds','playback_seconds','duration_seconds','segment_count'}
META_COLUMNS=['duration_seconds','subtitle_sentences','subtitle_characters','concept_count']

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False

def locate_processed(explicit=None):
    candidates=[explicit,os.getenv('BCE_PROCESSED'),'/kaggle/working/processed','/kaggle/input/datasets/kabil908/processed/processed','/content/drive/MyDrive/DataCon/processed','./processed']
    for x in candidates:
        if not x: continue
        p=Path(x)
        if all((p/'splits'/f).exists() for f in ['train.parquet','valid.parquet','test.parquet']): return p
    roots=[Path('/kaggle/input'),Path('/content/drive/MyDrive'),Path('.')]
    for root in roots:
        if root.exists():
            for f in root.rglob('train.parquet'):
                p=f.parent.parent
                if (p/'splits'/'valid.parquet').exists() and (p/'graph'/'video_index.parquet').exists(): return p
    raise FileNotFoundError('Could not locate processed/splits and processed/graph. Pass --processed explicitly.')

def left_pad(seq,n,pad):
    seq=list(seq)[-n:]; return seq+[pad]*(n-len(seq))

def raw_behaviour(frame):
    x=frame[BEHAVIOUR_COLUMNS].astype('float32').replace([np.inf,-np.inf],np.nan).fillna(0).copy()
    for c in LOG_BEHAVIOUR: x[c]=np.log1p(x[c].clip(lower=0))
    return x

def build_history(frame,features):
    out={}
    for u,idx in frame.groupby('u',sort=False).groups.items():
        rows=np.asarray(list(idx)); g=frame.loc[rows]
        out[int(u)]={'items':g.i.astype(int).tolist(),'times':g.timestamp.astype('int64').tolist(),
                     'behaviour':features[rows].tolist(),'completion':g.completion_ratio.astype(float).tolist()}
    return out

class PrefixDataset(Dataset):
    def __init__(self,histories,cfg):
        self.h=histories; self.cfg=cfg
        self.examples=[(u,t) for u,h in histories.items() for t in range(1,len(h['items']))]
    def __len__(self): return len(self.examples)
    def __getitem__(self,index):
        u,t=self.examples[index]; h=self.h[u]; c=self.cfg
        return (torch.tensor(u),torch.tensor(left_pad(h['items'][:t],c.max_len,0)),
                torch.tensor(left_pad(h['times'][:t],c.max_len,0)),
                torch.tensor(left_pad(h['behaviour'][:t],c.max_len,[0.]*len(BEHAVIOUR_COLUMNS)),dtype=torch.float32),
                torch.tensor(h['items'][t]),torch.tensor(h['completion'][t],dtype=torch.float32))

class BCESASRec(nn.Module):
    def __init__(self,n_items,n_concepts,n_courses,item_concepts,item_course,item_metadata,cfg):
        super().__init__(); self.cfg=cfg; d=cfg.hidden_dim
        self.item_emb=nn.Embedding(n_items+1,d,padding_idx=0)
        self.concept_emb=nn.Embedding(n_concepts+1,d,padding_idx=0)
        self.course_emb=nn.Embedding(n_courses+1,d,padding_idx=0)
        self.position_emb=nn.Embedding(cfg.max_len,d)
        self.time_emb=nn.Embedding(cfg.time_buckets,d,padding_idx=0)
        self.behaviour_mlp=nn.Sequential(nn.Linear(6,d//2),nn.GELU(),nn.Dropout(cfg.dropout),nn.Linear(d//2,d))
        self.metadata_mlp=nn.Sequential(nn.Linear(4,d//2),nn.GELU(),nn.Linear(d//2,d))
        self.concept_query=nn.Linear(d,d,bias=False); self.concept_key=nn.Linear(d,d,bias=False)
        self.event_norm=nn.LayerNorm(d); self.candidate_norm=nn.LayerNorm(d); self.dropout=nn.Dropout(cfg.dropout)
        layer=nn.TransformerEncoderLayer(d,cfg.attention_heads,cfg.feedforward_dim,cfg.dropout,batch_first=True,norm_first=True,activation='gelu')
        self.transformer=nn.TransformerEncoder(layer,cfg.transformer_layers); self.output_norm=nn.LayerNorm(d)
        self.completion_head=nn.Sequential(nn.Linear(2*d,d),nn.GELU(),nn.Dropout(cfg.dropout),nn.Linear(d,1))
        self.register_buffer('item_concepts',torch.as_tensor(item_concepts,dtype=torch.long))
        self.register_buffer('item_course',torch.as_tensor(item_course,dtype=torch.long))
        self.register_buffer('item_metadata',torch.as_tensor(item_metadata,dtype=torch.float32))
        self.scale=math.sqrt(d)
    def concept_pool(self,item_idx):
        ids=self.item_concepts[item_idx]; c=self.concept_emb(ids)
        q=self.concept_query(self.item_emb(item_idx)).unsqueeze(-2)
        logits=((q*self.concept_key(c)).sum(-1)/self.scale).float(); mask=ids.eq(0)
        weights=torch.softmax(logits.masked_fill(mask,-1e9),-1).masked_fill(mask,0.)
        weights=weights/weights.sum(-1,keepdim=True).clamp_min(1.)
        return (weights.unsqueeze(-1)*c.float()).sum(-2).to(c.dtype)
    def candidate(self,item_idx):
        z=self.item_emb(item_idx)+self.concept_pool(item_idx)+self.course_emb(self.item_course[item_idx])+self.metadata_mlp(self.item_metadata[item_idx])
        return self.candidate_norm(z)
    def time_bucket(self,times):
        gap=torch.zeros_like(times); valid=(times[:,1:]>0)&(times[:,:-1]>0)
        gap[:,1:]=torch.where(valid,(times[:,1:]-times[:,:-1]).clamp_min(0),torch.zeros_like(times[:,1:]))
        return (torch.floor(torch.log2(gap.float()+1)).long()+1).clamp(0,self.cfg.time_buckets-1).masked_fill(times.eq(0),0)
    def encode(self,seq,times,behaviour):
        pos=torch.arange(self.cfg.max_len,device=seq.device).unsqueeze(0); padding=seq.eq(0)
        x=self.candidate(seq)+self.behaviour_mlp(behaviour)+self.time_emb(self.time_bucket(times))+self.position_emb(pos)
        x=self.dropout(self.event_norm(x)).masked_fill(padding.unsqueeze(-1),0.)
        causal=torch.triu(torch.ones(self.cfg.max_len,self.cfg.max_len,device=seq.device,dtype=torch.bool),1)
        x=self.transformer(x,mask=causal,src_key_padding_mask=padding)
        last=seq.ne(0).sum(1).clamp_min(1)-1
        return self.output_norm(x[torch.arange(len(seq),device=seq.device),last])
    def sampled_logits(self,h,candidates): return (h[:,None,:]*self.candidate(candidates)).sum(-1)/self.scale
    def all_candidates(self): return self.candidate(torch.arange(1,self.item_emb.num_embeddings,device=self.item_emb.weight.device))
    def completion(self,h,item): return torch.sigmoid(self.completion_head(torch.cat([h,self.candidate(item)],-1))).squeeze(-1)

def prepare(processed,cfg):
    splits=processed/'splits'; graph=processed/'graph'
    required=[splits/f for f in ['train.parquet','valid.parquet','test.parquet']]+[graph/f for f in ['video_metadata.parquet','concept_video_edges.parquet','video_index.parquet','course_video_edges.parquet']]
    missing=[str(x) for x in required if not x.exists()]
    if missing: raise FileNotFoundError('Missing files:\n'+'\n'.join(missing))
    frames=[pd.read_parquet(splits/f).sort_values(['user_id','timestamp']).reset_index(drop=True) for f in ['train.parquet','valid.parquet','test.parquet']]
    train_df,valid_df,test_df=frames
    need={'user_id','video_id','timestamp',*BEHAVIOUR_COLUMNS}
    for name,f in zip(['train','valid','test'],frames):
        absent=need-set(f.columns)
        if absent: raise ValueError(f'{name} missing columns: {sorted(absent)}')
        f['user_id']=f.user_id.astype(str); f['video_id']=f.video_id.astype(str); f['timestamp']=pd.to_numeric(f.timestamp,errors='coerce').fillna(0).astype('int64')
    items=sorted(train_df.video_id.unique()); users=sorted(set(train_df.user_id)|set(valid_df.user_id)|set(test_df.user_id))
    item2idx={v:i+1 for i,v in enumerate(items)}; user2idx={v:i for i,v in enumerate(users)}
    def mapped(f):
        x=f[f.video_id.isin(item2idx)].copy().reset_index(drop=True); x['u']=x.user_id.map(user2idx).astype('int64'); x['i']=x.video_id.map(item2idx).astype('int64'); return x
    train,valid,test=map(mapped,frames); n_items=len(items)
    train_max=train.groupby('u').timestamp.max(); vb=valid.set_index('u'); tb=test.set_index('u'); common=sorted(set(train_max.index)&set(vb.index)&set(tb.index))
    checks={'chronology_train_valid':bool((train_max.loc[common].values<=vb.loc[common].timestamp.values).all()),'chronology_valid_test':bool((vb.loc[common].timestamp.values<=tb.loc[common].timestamp.values).all()),'valid_catalog':bool(valid.i.isin(set(train.i)).all()),'test_catalog':bool(test.i.isin(set(train.i)).all())}
    if not all(checks.values()): raise RuntimeError(f'Leakage audit failed: {checks}')
    traw=raw_behaviour(train); mean=traw.mean(); std=traw.std().replace(0,1).fillna(1)
    norm=lambda f:((raw_behaviour(f)-mean)/std).astype('float32').to_numpy()
    train_beh,valid_beh,test_beh=map(norm,[train,valid,test])
    vi=pd.read_parquet(graph/'video_index.parquet'); vi[['video_id','ccid']]=vi[['video_id','ccid']].astype(str)
    v2c=dict(zip(vi.video_id,vi.ccid)); c2i={v2c[v]:item2idx[v] for v in item2idx if v in v2c}
    cv=pd.read_parquet(graph/'concept_video_edges.parquet'); cv[['concept_id','ccid']]=cv[['concept_id','ccid']].astype(str); cv=cv[cv.ccid.isin(c2i)].drop_duplicates(['ccid','concept_id'])
    concepts=sorted(cv.concept_id.unique()); con2idx={c:i+1 for i,c in enumerate(concepts)}
    item_concepts=np.zeros((n_items+1,cfg.max_concepts_per_video),np.int64)
    for ccid,g in cv.groupby('ccid'):
        ids=[con2idx[c] for c in g.concept_id.iloc[:cfg.max_concepts_per_video]]; item_concepts[c2i[ccid],:len(ids)]=ids
    course=pd.read_parquet(graph/'course_video_edges.parquet'); course[['video_id','course_id']]=course[['video_id','course_id']].astype(str); course=course[course.video_id.isin(item2idx)].drop_duplicates('video_id')
    courses=sorted(course.course_id.unique()); co2idx={c:i+1 for i,c in enumerate(courses)}; item_course=np.zeros(n_items+1,np.int64)
    for r in course.itertuples(): item_course[item2idx[r.video_id]]=co2idx[r.course_id]
    meta=pd.read_parquet(graph/'video_metadata.parquet'); meta['video_id']=meta.video_id.astype(str)
    for c in META_COLUMNS:
        if c not in meta: meta[c]=0
    meta=meta.drop_duplicates('video_id').set_index('video_id'); mt=pd.DataFrame(index=items)
    for c in META_COLUMNS: mt[c]=np.log1p(pd.to_numeric(meta.reindex(items)[c],errors='coerce').replace([np.inf,-np.inf],np.nan).fillna(0).clip(lower=0))
    mt=((mt-mt.mean())/mt.std().replace(0,1).fillna(1)).astype('float32'); item_meta=np.zeros((n_items+1,4),np.float32); item_meta[1:]=mt.to_numpy()
    train_h=build_history(train,train_beh); ve=build_history(valid,valid_beh); te=build_history(test,test_beh); eval_users=sorted(set(train_h)&set(ve)&set(te))
    valid_hist={u:copy.deepcopy(train_h[u]) for u in eval_users}; test_hist={}; vt={u:ve[u]['items'][0] for u in eval_users}; tt={u:te[u]['items'][0] for u in eval_users}; vc={u:ve[u]['completion'][0] for u in eval_users}; tc={u:te[u]['completion'][0] for u in eval_users}
    for u in eval_users:
        h=copy.deepcopy(train_h[u])
        for k in ['items','times','behaviour','completion']: h[k].append(ve[u][k][0])
        test_hist[u]=h
    all_pos={u:set(train_h[u]['items'])|({vt[u],tt[u]} if u in vt else set()) for u in train_h}
    train_eval_users=sorted(u for u,h in train_h.items() if len(h['items'])>=2); train_eval_hist={u:{k:list(v[:-1]) for k,v in train_h[u].items()} for u in train_eval_users}; train_eval_target={u:train_h[u]['items'][-1] for u in train_eval_users}; train_eval_completion={u:train_h[u]['completion'][-1] for u in train_eval_users}
    return locals()

def metrics(ranks,top,item_pop,n_items,item_concepts,targets,users,ks):
    r=np.asarray(ranks); out={'Accuracy@1':float((r==1).mean()),'MRR':float((1/r).mean()),'MeanRank':float(r.mean()),'MedianRank':float(np.median(r))}
    for k in ks:
        hit=r<=k; rec=float(hit.mean()); pre=rec/k; out.update({f'Precision@{k}':pre,f'Recall@{k}':rec,f'F1@{k}':0 if rec==0 else 2*pre*rec/(pre+rec),f'NDCG@{k}':float(np.where(hit,1/np.log2(r+1),0).mean()),f'MAP@{k}':float(np.where(hit,1/r,0).mean())})
    out['CatalogCoverage@10']=len(np.unique(top))/n_items
    cr=[]
    for u,recs in zip(users,top):
        true=set(item_concepts[targets[u]])-{0}; pred=set(item_concepts[np.asarray(recs)].ravel())-{0}
        if true: cr.append(len(true&pred)/len(true))
    out['ConceptRecall@10']=float(np.mean(cr)) if cr else float('nan'); return out

def main(args):
    cfg=Config(); processed=locate_processed(args.processed); out=Path(args.output or ('/kaggle/working/bce_sasrec_3seeds' if Path('/kaggle/working').exists() else './bce_sasrec_3seeds')); (out/'checkpoints').mkdir(parents=True,exist_ok=True); (out/'reports').mkdir(exist_ok=True)
    D=prepare(processed,cfg); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); print('Processed:',processed,'Device:',device)
    train_dataset=PrefixDataset(D['train_h'],cfg)
    def tensors(hist,us):
        return (torch.tensor([left_pad(hist[u]['items'],cfg.max_len,0) for u in us],device=device),torch.tensor([left_pad(hist[u]['times'],cfg.max_len,0) for u in us],device=device),torch.tensor([left_pad(hist[u]['behaviour'],cfg.max_len,[0.]*6) for u in us],dtype=torch.float32,device=device))
    @torch.no_grad()
    def evaluate(model,hist,targets,completion,users,desc):
        model.eval(); cz=model.all_candidates(); ranks=[]; tops=[]; ce=0.; n=0; errs=[]
        for st in tqdm(range(0,len(users),cfg.eval_batch_size),desc=desc,leave=False):
            us=users[st:st+cfg.eval_batch_size]; seq,times,beh=tensors(hist,us); h=model.encode(seq,times,beh); scores=h@cz.T/model.scale
            tar=torch.tensor([targets[u]-1 for u in us],device=device)
            for row,u in enumerate(us):
                seen=set(hist[u]['items']); seen.discard(targets[u])
                if seen: scores[row,torch.tensor([i-1 for i in seen],device=device)]=torch.finfo(scores.dtype).min
            ce+=F.cross_entropy(scores,tar,reduction='sum').item(); n+=len(us); ts=scores[torch.arange(len(us),device=device),tar]; ranks.extend(((scores>ts[:,None]).sum(1)+1).cpu().tolist()); tops.extend((scores.topk(10,1).indices+1).cpu().tolist()); pred=model.completion(h,tar+1).cpu().numpy(); true=np.asarray([completion[u] for u in us]); errs.extend(pred-true)
        m=metrics(ranks,tops,D['train'].i.value_counts().to_dict(),D['n_items'],D['item_concepts'],targets,users,cfg.ks); e=np.asarray(errs); m.update(Loss=ce/n,CompletionMAE=float(np.abs(e).mean()),CompletionRMSE=float(np.sqrt((e**2).mean()))); return m
    def pair_loss(model,h,pos):
        ids=model.item_concepts[pos]; mask=ids.ne(0); has=mask.any(1)
        if not has.any(): return h.sum()*0
        first=mask.float().argmax(1); pc=ids[torch.arange(len(ids),device=device),first]; nc=torch.randint(1,model.concept_emb.num_embeddings,(len(ids),),device=device); nc=torch.where(nc.eq(pc),(nc%(model.concept_emb.num_embeddings-1))+1,nc); return -F.logsigmoid((h*model.concept_emb(pc)).sum(-1)[has]/model.scale-(h*model.concept_emb(nc)).sum(-1)[has]/model.scale).mean()
    all_rows=[]
    for seed in cfg.seeds:
        seed_all(seed)
        loader_generator=torch.Generator().manual_seed(seed)
        loader=DataLoader(train_dataset,batch_size=cfg.batch_size,shuffle=True,generator=loader_generator,num_workers=cfg.num_workers,pin_memory=device.type=='cuda',persistent_workers=cfg.num_workers>0)
        model=BCESASRec(D['n_items'],len(D['concepts']),len(D['courses']),D['item_concepts'],D['item_course'],D['item_meta'],cfg).to(device); opt=torch.optim.AdamW(model.parameters(),lr=cfg.learning_rate,weight_decay=cfg.weight_decay); best=-1.; bad=0; history=[]; ck=out/'checkpoints'/f'BCE_SASRec_seed_{seed}.pt'
        for ep in range(1,cfg.max_epochs+1):
            model.train(); sums=defaultdict(float); count=0
            for users,seq,times,beh,pos,comp in tqdm(loader,desc=f'Seed {seed} epoch {ep}',leave=False):
                users,seq,times,beh,pos,comp=[x.to(device,non_blocking=True) for x in [users,seq,times,beh,pos,comp]]; neg=[]
                for u in users.cpu().tolist():
                    vals=[]; known=D['all_pos'][u]
                    while len(vals)<cfg.negatives:
                        x=random.randint(1,D['n_items'])
                        if x not in known: vals.append(x)
                    neg.append(vals)
                cand=torch.cat([pos[:,None],torch.tensor(neg,device=device)],1); opt.zero_grad(set_to_none=True); h=model.encode(seq,times,beh); rank=F.cross_entropy(model.sampled_logits(h,cand),torch.zeros(len(seq),dtype=torch.long,device=device),label_smoothing=cfg.label_smoothing); con=pair_loss(model,h,pos); cl=F.mse_loss(model.completion(h,pos),comp.clamp(0,1)); loss=rank+cfg.concept_loss_weight*con+cfg.completion_loss_weight*cl
                if not torch.isfinite(loss): raise FloatingPointError('Non-finite loss')
                loss.backward(); grad=nn.utils.clip_grad_norm_(model.parameters(),cfg.gradient_clip)
                if not torch.isfinite(grad): raise FloatingPointError('Non-finite gradient')
                opt.step(); bs=len(seq); count+=bs
                for k,v in [('TrainLoss',loss),('RankingLoss',rank),('ConceptLoss',con),('CompletionLoss',cl)]: sums[k]+=float(v.detach())*bs
            val=evaluate(model,D['valid_hist'],D['vt'],D['vc'],D['eval_users'],'Validation'); row={'Epoch':ep,**{k:v/count for k,v in sums.items()},**{f'Val_{k}':v for k,v in val.items()}}; history.append(row); score=val['NDCG@10']; print(f'Seed {seed} epoch {ep}: loss={row["TrainLoss"]:.4f}, val NDCG@10={score:.4f}')
            if score>best+cfg.early_stopping_min_delta: best=score; bad=0; torch.save({'state':model.state_dict(),'epoch':ep,'validation':val,'config':asdict(cfg)},ck)
            else: bad+=1
            if ep>=cfg.minimum_epochs and bad>=cfg.early_stopping_patience: break
        pd.DataFrame(history).to_csv(out/'reports'/f'BCE_SASRec_seed_{seed}_epochs.csv',index=False); saved=torch.load(ck,map_location=device,weights_only=False); model.load_state_dict(saved['state'])
        evaluations=[('Train',D['train_eval_hist'],D['train_eval_target'],D['train_eval_completion'],D['train_eval_users']),('Validation',D['valid_hist'],D['vt'],D['vc'],D['eval_users']),('Test',D['test_hist'],D['tt'],D['tc'],D['eval_users'])]
        for split,h,t,c,u in evaluations: all_rows.append({'Seed':seed,'Split':split,'BestEpoch':saved['epoch'],**evaluate(model,h,t,c,u,f'{split} seed {seed}')})
        del model; gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None
    results=pd.DataFrame(all_rows); results.to_csv(out/'reports'/'three_seed_all_metrics.csv',index=False); numeric=[c for c in results.columns if c not in ['Seed','Split']]; summary=results.groupby('Split')[numeric].agg(['mean','std']); summary.to_csv(out/'reports'/'three_seed_mean_std.csv'); json.dump({'processed':str(processed),'config':asdict(cfg)},open(out/'reports'/'manifest.json','w'),indent=2); print(results[['Seed','Split','BestEpoch','NDCG@10','Recall@10','MRR','Accuracy@1','ConceptRecall@10','Recall@20']].to_string(index=False)); print('Outputs:',out)

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--processed'); ap.add_argument('--output'); main(ap.parse_args())
