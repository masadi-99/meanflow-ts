"""Residual flow with SEPARATE optimizers and pretrained base."""
import os, sys, time, logging, tempfile
import numpy as np
import torch
from torch.optim import AdamW
from copy import deepcopy
from tqdm.auto import tqdm
from gluonts.dataset.repository.datasets import get_dataset
from gluonts.dataset.loader import TrainDataLoader
from gluonts.evaluation import Evaluator, make_evaluation_predictions
from gluonts.itertools import Cached
from gluonts.time_feature import time_features_from_frequency_str
from gluonts.torch.batchify import batchify
from gluonts.torch.model.predictor import PyTorchPredictor
from gluonts.transform import *
try:
    import pykeops; tmp = tempfile.mkdtemp(prefix="pykeops_"); pykeops.set_build_folder(tmp); pykeops.clean_pykeops()
except: pass
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from meanflow_ts.model_v2 import ConditionalMeanFlowNetV2, conditional_meanflow_loss_v2, MeanFlowForecasterV2, extract_lag_features
from oneflow_ts.improvements import BaseForecaster, AdaptiveForecaster, _sample_t_r
import torch.nn.functional as F

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)
FREQ="H"; CTX=24; PRED=24; MAX_LAG=672; N_LAGS=7; SAMPLES=16; TSFLOW=0.045

def get_loader():
    ds = get_dataset("electricity_nips")
    tr = Chain([AsNumpyArray(field="target",expected_ndim=1),AddObservedValuesIndicator(target_field="target",output_field="observed_values"),AddTimeFeatures(start_field="start",target_field="target",output_field="time_feat",time_features=time_features_from_frequency_str(FREQ),pred_length=PRED)])
    sp = InstanceSplitter(target_field="target",is_pad_field="is_pad",start_field="start",forecast_start_field="forecast_start",instance_sampler=ExpectedNumInstanceSampler(num_instances=1,min_future=PRED),past_length=CTX+MAX_LAG,future_length=PRED,time_series_fields=["time_feat","observed_values"])
    td = tr.apply(ds.train, is_train=True)
    loader = TrainDataLoader(Cached(td),batch_size=64,stack_fn=batchify,transform=sp,num_batches_per_epoch=128,shuffle_buffer_length=10000)
    return ds, tr, loader

def main():
    device = torch.device("cuda")
    torch.manual_seed(6432); np.random.seed(6432)
    ds, tr, loader = get_loader()

    flow = ConditionalMeanFlowNetV2(pred_len=PRED,ctx_len=CTX,n_lags=N_LAGS,model_channels=128,num_res_blocks=4,time_emb_dim=64,dropout=0.1).to(device)
    base = BaseForecaster(CTX,PRED,N_LAGS).to(device)
    flow_ema = deepcopy(flow).eval()
    base_ema = deepcopy(base).eval()

    # SEPARATE optimizers with different LRs
    opt_flow = AdamW(flow.parameters(), lr=6e-4)
    opt_base = AdamW(base.parameters(), lr=1e-3)  # Higher LR for simple MLP

    logger.info(f"[residual-fixed] Flow: {sum(p.numel() for p in flow.parameters()):,}, Base: {sum(p.numel() for p in base.parameters()):,}")

    # Phase 1: Pretrain base for 50 epochs (MSE only)
    logger.info("Phase 1: Pretraining base forecaster (50 epochs)")
    for epoch in range(50):
        base.train()
        eloss, nb = 0, 0
        for batch in loader:
            past = batch["past_target"].to(device); future = batch["future_target"].to(device)
            ctx = past[:,-CTX:]; loc = ctx.abs().mean(dim=1,keepdim=True).clamp(min=0.01)
            ctx_lags = extract_lag_features(past,CTX,FREQ,N_LAGS)/loc.unsqueeze(1)
            scaled = future/loc
            pred = base(ctx_lags)
            loss = F.mse_loss(pred, scaled)
            opt_base.zero_grad(); loss.backward(); opt_base.step()
            with torch.no_grad():
                for p,pe in zip(base.parameters(),base_ema.parameters()): pe.data.lerp_(p.data,0.01)
            eloss += loss.item(); nb += 1
        if (epoch+1)%10==0:
            logger.info(f"  Base pretrain epoch {epoch+1}/50 | MSE: {eloss/nb:.4f}")

    # Phase 2: Joint training (flow on residuals, base continues learning)
    logger.info("Phase 2: Joint training (600 epochs)")
    best_crps = float('inf')
    for epoch in range(600):
        flow.train(); base.train()
        eloss_f, eloss_b, nb = 0, 0, 0
        t0 = time.time()
        for batch in loader:
            past = batch["past_target"].to(device); future = batch["future_target"].to(device)
            ctx = past[:,-CTX:]; loc = ctx.abs().mean(dim=1,keepdim=True).clamp(min=0.01)
            ctx_lags = extract_lag_features(past,CTX,FREQ,N_LAGS)/loc.unsqueeze(1)
            scaled = future/loc

            # Base prediction (detached for flow)
            base_pred = base(ctx_lags)
            residual = scaled - base_pred.detach()

            # Flow loss on residual
            B = scaled.shape[0]; e = torch.randn_like(residual)
            t,r = _sample_t_r(B,device); t_bc=t.unsqueeze(-1); r_bc=r.unsqueeze(-1)
            z = (1-t_bc)*residual + t_bc*e; v = e-residual
            def u_func(z,t_bc,r_bc):
                h_bc=t_bc-r_bc; return flow(z,(t_bc.squeeze(-1),h_bc.squeeze(-1)),ctx_lags)
            with torch.amp.autocast("cuda",enabled=False):
                u_pred,dudt = torch.func.jvp(u_func,(z,t_bc,r_bc),(v,torch.ones_like(t_bc),torch.zeros_like(r_bc)))
                u_tgt = (v-(t_bc-r_bc)*dudt).detach()
                fl = (u_pred-u_tgt)**2; fl=fl.sum(dim=1)
                adp=(fl.detach()+1e-3)**0.75; fl=(fl/adp).mean()

            opt_flow.zero_grad(); fl.backward()
            torch.nn.utils.clip_grad_norm_(flow.parameters(),1.0); opt_flow.step()

            # Base loss (separate backward)
            bl = F.mse_loss(base_pred, scaled)
            opt_base.zero_grad(); bl.backward()
            torch.nn.utils.clip_grad_norm_(base.parameters(),1.0); opt_base.step()

            with torch.no_grad():
                for p,pe in zip(flow.parameters(),flow_ema.parameters()): pe.data.lerp_(p.data,1e-4)
                for p,pe in zip(base.parameters(),base_ema.parameters()): pe.data.lerp_(p.data,1e-4)
            eloss_f+=fl.item(); eloss_b+=bl.item(); nb+=1

        elapsed=time.time()-t0
        if (epoch+1)%20==0:
            logger.info(f"[residual-fixed] Epoch {epoch+1}/600 | FlowLoss: {eloss_f/nb:.4f} | BaseMSE: {eloss_b/nb:.4f} | {elapsed:.1f}s")

        if (epoch+1)%100==0 or (epoch+1)==600:
            flow_ema.eval(); base_ema.eval()
            tt = tr.apply(ds.test,is_train=False)
            ts = InstanceSplitter(target_field="target",is_pad_field="is_pad",start_field="start",forecast_start_field="forecast_start",instance_sampler=TestSplitSampler(),past_length=CTX+MAX_LAG,future_length=PRED,time_series_fields=["time_feat","observed_values"])
            fc = AdaptiveForecaster(flow_ema,base_ema,CTX,PRED,num_samples=SAMPLES,freq=FREQ,n_lags=N_LAGS,max_steps=1).to(device)
            pr = PyTorchPredictor(prediction_length=PRED,input_names=["past_target","past_observed_values"],prediction_net=fc,batch_size=512,input_transform=ts,device=device)
            fi,ti = make_evaluation_predictions(dataset=tt,predictor=pr,num_samples=SAMPLES)
            forecasts=list(tqdm(fi,total=len(tt),desc="Eval",leave=False)); tss=list(ti)
            metrics,_ = Evaluator(num_workers=0)(tss,forecasts)
            crps=metrics["mean_wQuantileLoss"]; nd=metrics["ND"]
            if crps<best_crps: best_crps=crps
            logger.info(f"[residual-fixed] Epoch {epoch+1} -> CRPS={crps:.6f} | ND={nd:.6f} | Best={best_crps:.6f} | TSFlow={TSFLOW}")

    logger.info(f"[residual-fixed] FINAL Best CRPS: {best_crps:.6f}")

if __name__=="__main__": main()
