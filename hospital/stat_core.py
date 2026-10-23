import runtime
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit,logit
from scipy.stats import rankdata,norm
from sklearn.metrics import roc_curve


def bh(p):
 p=np.asarray(p);order=np.argsort(p);q=np.empty(len(p));q[order]=np.minimum(1,np.minimum.accumulate((p[order]*len(p)/np.arange(1,len(p)+1))[::-1])[::-1]);return q

def weighted_auc(y,p,w):
 order=np.argsort(p,kind='stable');yy=y[order];pp=p[order];starts=np.r_[0,np.flatnonzero(np.diff(pp))+1];ww=w[:,order]
 pos=np.add.reduceat(ww*yy,starts,axis=1);neg=np.add.reduceat(ww*(1-yy),starts,axis=1)
 return np.sum(pos*(np.cumsum(neg,axis=1)-.5*neg),axis=1)/(pos.sum(axis=1)*neg.sum(axis=1))

def bootstrap(y,p,groups,thresholds,B,rs):
 _,gi=np.unique(groups,return_inverse=True);G=gi.max()+1;rng=np.random.default_rng(rs)
 result={k:np.empty((B,p.shape[1])) for k in ['AUROC','Brier','F1']}
 err=(p-y[:,None])**2;pred=(p>=np.asarray(thresholds)[None,:]).astype(float)
 tp=pred*y[:,None];fp=pred*(1-y[:,None]);fn=(1-pred)*y[:,None]
 for start in range(0,B,32):
  end=min(B,start+32);counts=rng.multinomial(G,np.full(G,1/G),size=end-start);w=counts[:,gi].astype(float)
  assert ((w@y)>0).all() and ((w@(1-y))>0).all()
  result['Brier'][start:end]=(w@err)/w.sum(axis=1)[:,None]
  t=w@tp;den=2*t+w@fp+w@fn;result['F1'][start:end]=np.divide(2*t,den,out=np.zeros_like(t),where=den!=0)
  for j in range(p.shape[1]):result['AUROC'][start:end,j]=weighted_auc(y,p[:,j],w)
 return result

def calibration(y,p):
 z=logit(np.clip(p,1e-6,1-1e-6));x=np.c_[np.ones(len(z)),z]
 def f(b):
  eta=x@b;return np.mean(np.logaddexp(0,eta)-y*eta),x.T@(expit(eta)-y)/len(y)
 fit=minimize(f,[0.,1.],jac=True,method='BFGS',options={'gtol':1e-7,'maxiter':1000})
 assert fit.success or np.max(abs(f(fit.x)[1]))<1e-5
 return fit.x

def delong(y,p):
 # p is n x number_of_models; midranks give the paired DeLong covariance.
 pp=p[y==1].T;nn=p[y==0].T;m=pp.shape[1];n=nn.shape[1]
 tx=np.array([rankdata(v) for v in pp]);ty=np.array([rankdata(v) for v in nn]);tz=np.array([rankdata(v) for v in np.c_[pp,nn]])
 auc=tz[:,:m].sum(axis=1)/(m*n)-(m+1)/(2*n)
 v01=(tz[:,:m]-tx)/n;v10=1-(tz[:,m:]-ty)/m
 cov=np.cov(v01)/m+np.cov(v10)/n
 return auc,cov

def youden(y,p):
 fpr,tpr,t=roc_curve(y,p);return float(np.clip(t[np.argmax(tpr-fpr)],0,1))

def boot_p(x):
 x=np.asarray(x);return float(min(1.,2*min(np.mean(x<=0),np.mean(x>=0))))
