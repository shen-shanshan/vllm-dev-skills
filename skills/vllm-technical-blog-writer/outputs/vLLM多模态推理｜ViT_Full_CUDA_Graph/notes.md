补充 CG：
torch Custom Op 与 Dynamo 的交互：
torch.library 注册的 custom op 对 Dynamo 不透明——trace 时不内联其 Python 实现。因此在 op 内做基于运行时 shape 的分支，可以在同一个编译图里实现「decode 走 CK、prefill 走 Triton」的动态调度；分支若放在被 trace 的调用方，会被固化成 trace 时 shape 对应的固定路径。

挑战：
newline token 与 view-separator 的拼装在 postprocess_encoder_output 中于图外完成，从而保持捕获计算形状静态。
