"""Chẩn đoán bản chất dữ liệu: vì sao các cấu hình trong ablation phẳng.

    python scripts/diagnose_data.py

Kết quả và cách đọc: docs/data-diagnosis.md
"""
import sys, json, numpy as np, pandas as pd, scipy.sparse as sp
sys.stdout.reconfigure(encoding="utf-8")
rng=np.random.default_rng(0)
D="data/processed/"
tr=pd.read_parquet(D+"train.parquet"); te=pd.read_parquet(D+"test.parquet")
z=np.load(D+"facets.npz"); idx=json.load(open(D+"index.json",encoding="utf-8"))
ii={int(k):v for k,v in idx["item_index"].items()}; ui=idx["user_index"]
m=pd.read_parquet(D+"media_core.parquet"); m["pos"]=m.media_id.map(ii)
n_items=len(ii)
tr=tr[tr.media_id.isin(ii)&tr.user.isin(ui)]
X=sp.csr_matrix((np.ones(len(tr)),(tr.user.map(ui),tr.media_id.map(ii))),shape=(len(ui),n_items))
X.data[:]=1
deg=np.asarray(X.sum(0)).ravel()
def norm(v):
    n=np.linalg.norm(v,axis=1,keepdims=True); out=v/np.maximum(n,1e-9); return out,(n.ravel()>1e-8)
F={k:norm(z["facet_"+k]) for k in ("theme","creator","pace","art")}
F["ALS"]=norm(z["item_factors"])
XT=X.T.tocsr()
print("=== C. Láng giềng trong từng mặt có được đọc cùng nhau không ===")
print("lift = P(đọc j | đọc i) / P(đọc j); ngẫu nhiên ≈ 1")
cands=np.where(deg>=20)[0]; sample=rng.choice(cands,800,replace=False)
nu=X.shape[0]
for name,(V,ok) in F.items():
    lifts=[]; same_pop=[]
    for i in sample:
        if not ok[i]: continue
        s=V@V[i]; s[i]=-9; s[~ok]=-9
        nb=np.argpartition(-s,10)[:10]
        ri=XT[i].indices
        co=np.asarray(X[ri][:,nb].sum(0)).ravel()
        p_cond=co/len(ri); p=deg[nb]/nu
        lifts.append(np.mean(p_cond/np.maximum(p,1e-9)))
        same_pop.append(np.median(deg[nb]))
    print(f"  {name:<8} lift={np.median(lifts):6.2f}  (trung vị độ phổ biến láng giềng {np.median(same_pop):.0f}, phủ {ok.mean():.0%})")
# random baseline
lifts=[]
for i in sample:
    nb=rng.choice(n_items,10,replace=False); ri=XT[i].indices
    co=np.asarray(X[ri][:,nb].sum(0)).ravel(); lifts.append(np.mean((co/len(ri))/np.maximum(deg[nb]/nu,1e-9)))
print(f"  ngẫu nhiên lift={np.median(lifts):6.2f}")

print("\n=== D. Gu người đọc có chụm trong từng mặt không (so với tập ngẫu nhiên cùng độ phổ biến) ===")
users=rng.choice(X.shape[0],1500,replace=False)
p=deg/deg.sum()
for name,(V,ok) in F.items():
    real=[];fake=[]
    for u in users:
        it=X[u].indices; it=it[ok[it]]
        if len(it)<5: continue
        real.append(np.linalg.norm(V[it].mean(0)))
        fk=rng.choice(n_items,len(it),replace=False,p=p); fk=fk[ok[fk]]
        if len(fk)>=3: fake.append(np.linalg.norm(V[fk].mean(0)))
    print(f"  {name:<8} độ chụm thật {np.median(real):.3f} | ngẫu nhiên-cùng-độ-phổ-biến {np.median(fake):.3f} | tỉ lệ {np.median(real)/np.median(fake):.2f}")

print("\n=== E. Bên trong top-400 ứng viên ALS: điểm nào phân biệt được truyện trúng? (AUC, 0.5 = vô dụng) ===")
U=z["user_factors"]; I=z["item_factors"]
te=te[te.media_id.isin(ii)&te.user.isin(ui)]
hits=te.groupby("user").media_id.apply(lambda s:set(s.map(ii)))
cut=pd.Timestamp(idx["cutoff"])
recent=tr[tr.event_time>cut-pd.Timedelta(days=60)].media_id.map(ii).value_counts().reindex(range(n_items),fill_value=0).to_numpy()
sy=m.set_index("pos").start_year.reindex(range(n_items)).fillna(0).to_numpy()
rel=(m.set_index("pos").pub_status.reindex(range(n_items))=="RELEASING").to_numpy().astype(float)
from sklearn.metrics import roc_auc_score
feats={k:[] for k in ["ALS","popularity","pop_60_ngày","năm_ra","đang_ra","theme","creator","pace","art"]}
labels=[]; n_hit_in=0; n_hit=0
for u in rng.choice(hits.index.to_numpy(),min(1500,len(hits)),replace=False):
    up=ui[u]; seen=X[up].indices
    s=U[up]@I.T; s[seen]=-np.inf
    c=np.argpartition(-s,400)[:400]
    h=hits[u]; y=np.isin(c,list(h)); n_hit+=len(h); n_hit_in+=y.sum()
    if y.sum()==0: continue
    labels.append(y); feats["ALS"].append(s[c]); feats["popularity"].append(deg[c]); feats["pop_60_ngày"].append(recent[c])
    feats["năm_ra"].append(sy[c]); feats["đang_ra"].append(rel[c])
    for name in ("theme","creator","pace","art"):
        V,ok=F[name]; it=seen[ok[seen]]
        prof=V[it].mean(0) if len(it) else np.zeros(V.shape[1])
        v=V[c]@prof; v[~ok[c]]=np.nan; feats[name].append(v)
y=np.concatenate(labels)
print(f"  truyện test lọt vào top-400 ứng viên: {n_hit_in/n_hit:.1%}")
for k,v in feats.items():
    v=np.concatenate(v); good=np.isfinite(v)
    print(f"  {k:<12} AUC={roc_auc_score(y[good],v[good]):.3f}")

print("\n=== F. Tín hiệu có cộng được không: hồi quy logistic trên ứng viên, chia theo người đọc ===")
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
groups=np.concatenate([np.full(len(l),i) for i,l in enumerate(labels)])
cols={k:np.nan_to_num(np.concatenate(v)) for k,v in feats.items()}
cols["popularity"]=np.log1p(cols["popularity"]); cols["pop_60_ngày"]=np.log1p(cols["pop_60_ngày"])
cols["năm_ra"]=np.clip(cols["năm_ra"],1990,2027)
def cv_auc(keys):
    Xf=np.column_stack([cols[k] for k in keys]); Xf=(Xf-Xf.mean(0))/(Xf.std(0)+1e-9)
    aucs=[]
    for a,b in GroupKFold(4).split(Xf,y,groups):
        clf=LogisticRegression(max_iter=500,class_weight="balanced").fit(Xf[a],y[a])
        aucs.append(roc_auc_score(y[b],clf.decision_function(Xf[b])))
    return np.mean(aucs)
base=["ALS"]
for keys in (["ALS"],["ALS","theme","pace","art","creator"],
             ["ALS","pop_60_ngày","năm_ra","đang_ra"],
             ["ALS","pop_60_ngày","năm_ra","đang_ra","theme","pace","art","creator"]):
    print(f"  {' + '.join(keys):<70} AUC={cv_auc(keys):.3f}")
print("  tương quan theme vs ALS trong ứng viên:", round(np.corrcoef(cols['theme'],cols['ALS'])[0,1],3))
print("  tương quan theme vs pop_60_ngày      :", round(np.corrcoef(cols['theme'],cols['pop_60_ngày'])[0,1],3))

print("\n=== G. Gu mờ vì người đọc 'ăn tạp' hay vì đặc trưng mờ? ===")
cty=m.set_index("pos").country.reindex(range(n_items)).to_numpy()
V,ok=F["theme"]
whole=[];part=[]
for u in users:
    it=X[u].indices; it=it[ok[it]]
    if len(it)<15: continue
    whole.append(np.linalg.norm(V[it].mean(0)))
    # chia lịch sử thành cụm bằng kmeans 3 cụm trong không gian theme
    from sklearn.cluster import KMeans
    km=KMeans(3,n_init=2,random_state=0).fit(V[it])
    part.append(np.mean([np.linalg.norm(V[it[km.labels_==c]].mean(0)) for c in range(3) if (km.labels_==c).sum()>=3]))
print(f"  độ chụm theme: một tâm {np.median(whole):.3f} | ba tâm {np.median(part):.3f}")
# test items closer to nearest history item than to centroid?
near=[];cent=[]
for u in rng.choice(hits.index.to_numpy(),800,replace=False):
    up=ui[u]; it=X[up].indices; it=it[ok[it]]
    if len(it)<5: continue
    h=[j for j in hits[u] if ok[j]]
    if not h: continue
    prof=V[it].mean(0); prof/=np.linalg.norm(prof)+1e-9
    rnd=rng.choice(n_items,len(h)); 
    s_max=lambda js: np.max(V[js]@V[it].T,axis=1)
    near.append(np.mean(s_max(h))-np.mean(s_max(rnd)))
    cent.append(np.mean(V[h]@prof)-np.mean(V[rnd]@prof))
print(f"  truyện test hơn ngẫu nhiên — theo tâm: +{np.median(cent):.3f} | theo truyện gần nhất trong lịch sử: +{np.median(near):.3f}")
