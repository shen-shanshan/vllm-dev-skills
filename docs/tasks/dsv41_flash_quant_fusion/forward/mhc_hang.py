"""Isolate: does AITER mega_mhc hang on a CUDA-graph replay at a small T after
large-T calls, in mxfp8 vs bf16 mode?
Usage: python mhc_hang.py <bf16|mxfp8> <T_big> <T_small>"""
import sys, time, torch
from aiter.ops.mega_mhc import mega_mhc, mega_mhc_shuffle_fn
mode, tb, ts = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
dev, H = "cuda", 5120
torch.manual_seed(0)
fnp = mega_mhc_shuffle_fn(torch.randn(24, 4 * H, device=dev) * 0.01)
hs, hb = torch.rand(3, device=dev), torch.randn(24, device=dev) * 0.1
w = (torch.rand(H, device=dev) + 0.5).to(torch.bfloat16)
def args(T):
    return dict(residual=torch.randn(T, 4, H, dtype=torch.bfloat16, device=dev),
                pre_mix=torch.softmax(torch.randn(T, 4, device=dev), -1),
                sublayer_out=torch.randn(T, H, dtype=torch.bfloat16, device=dev),
                post_layer_mix=torch.rand(T, 4, device=dev), comb_res_mix=torch.rand(T, 4, 4, device=dev))
def call(a):
    return mega_mhc(a["residual"], fnp, hs, hb, 1e-6, 1e-6, 1e-6, 2.0, 20, a["pre_mix"],
                    sublayer_out=a["sublayer_out"], post_layer_mix=a["post_layer_mix"],
                    comb_res_mix=a["comb_res_mix"], norm_weight=w, norm_eps=1e-6,
                    residual_out=torch.empty_like(a["residual"]), w_preshuffle_bf16=True,
                    out_dtype=mode, scale_row_major=(mode == "mxfp8"))
big, small = args(tb), args(ts)
for _ in range(3): call(big)
torch.cuda.synchronize(); print("big eager ok", flush=True)
for _ in range(3): call(small)
torch.cuda.synchronize(); print("small eager ok", flush=True)
s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for _ in range(2): call(small)
torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g, stream=s):
    call(small)
print("captured", flush=True)
t = time.time()
for i in range(5):
    g.replay()
torch.cuda.synchronize()
print(f"replay ok {time.time() - t:.3f}s", flush=True)
