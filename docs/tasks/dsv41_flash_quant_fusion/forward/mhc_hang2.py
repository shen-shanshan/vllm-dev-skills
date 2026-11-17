"""Like mhc_hang.py, but follows `vllm bench forward`'s order on one stream:
big eager -> big graph capture + replays -> small eager -> small graph capture
+ replays. Usage: python mhc_hang2.py <bf16|mxfp8> <T_big> <T_small>"""
import sys, time, torch
sys.argv += []
exec(open(__file__.replace("mhc_hang2.py", "mhc_hang.py")).read().split("big, small = args(tb), args(ts)")[0])
big, small = args(tb), args(ts)
s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())
def eager_then_graph(a, name):
    with torch.cuda.stream(s):
        for _ in range(2): call(a)
    torch.cuda.synchronize()
    print(f"{name} eager ok", flush=True)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s):
        call(a)
    for _ in range(5): g.replay()
    torch.cuda.synchronize()
    print(f"{name} graph replay ok", flush=True)
    return g
gb = eager_then_graph(big, "big")
gs = eager_then_graph(small, "small")
for _ in range(3):
    gb.replay(); gs.replay()
torch.cuda.synchronize()
print("interleaved replay ok", flush=True)
