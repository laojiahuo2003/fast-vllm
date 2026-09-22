import importlib.util
import pathlib

import torch
import torch.nn.functional as F

# 直接加载 fused_ops.py 模块，绕开 fastvllm/__init__.py 的完整引擎链
# （kernel 是独立单元，不需要 model/engine 就能测——这就是单独放 kernels/ 的收益）
_fused_ops_path = pathlib.Path(__file__).resolve().parent.parent / "fastvllm" / "kernels" / "fused_ops.py"
_spec = importlib.util.spec_from_file_location("fused_ops", _fused_ops_path)
fused_ops = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fused_ops)

silu_and_mul = fused_ops.silu_and_mul


def _ref_silu_and_mul(x: torch.Tensor) -> torch.Tensor:
    """参考实现：FP32 计算，作为正确性基准"""
    gate, up = x.chunk(2, -1)
    return F.silu(gate.float()) * up.float()


def test_silu_and_mul():
    # 用多种 shape 验证：行数变化、列数变化（都取 BLOCK_SIZE 的整数倍）
    shapes = [(64, 8192), (1, 4096), (128, 1024), (7, 2048)]
    for shape in shapes:
        x = torch.randn(shape, dtype=torch.float16, device="cuda")
        ref = _ref_silu_and_mul(x)
        out = silu_and_mul(x)
        err = (out.float() - ref).abs().max().item()
        # FP16 存储精度是相对的：值越大舍入误差越大，用相对误差判据
        assert torch.allclose(out.float(), ref, rtol=1e-2, atol=1e-3), f"shape={shape} 误差超限: {err:.2e}"
        print(f"shape={str(shape):<12} max abs err={err:.2e}  OK")


if __name__ == "__main__":
    test_silu_and_mul()
    print("\n全部通过")
