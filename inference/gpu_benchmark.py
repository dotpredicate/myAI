import time
from typing import Dict, Any
from tinygrad import Tensor, TinyJit, dtypes, Context, device, GlobalCounters

@Context(DEV="HIP")
def benchmark_tflops(M: int = 4096, K: int = 4096, N: int = 4096, warmup: int = 10, iterations: int = 50) -> Dict[str, Any]:
    return {
        'fp32': _run_tflops_test(M, K, N, warmup, iterations, dtypes.float32),
        'fp16': _run_tflops_test(M, K, N, warmup, iterations, dtypes.float16),
        'int8': _run_tflops_test(M, K, N, warmup, iterations, dtypes.int8),
    }

def _run_tflops_test(M, K, N, warmup, iterations, dtype) -> Dict[str, Any]:
    a = Tensor.randn(M, K, dtype=dtype).realize()
    b = Tensor.randn(K, N, dtype=dtype).realize()

    @TinyJit
    def run_test(a, b):
        return (a @ b).realize()

    for _ in range(warmup):
        run_test(a, b)
        device.Device.default.synchronize()
    
    start_time = time.perf_counter()
    for _ in range(iterations):
        run_test(a, b)
        device.Device.default.synchronize()

    end_time = time.perf_counter()
    
    avg_time = (end_time - start_time) / iterations
    total_flops = 2 * M * N * K
    tflops = (total_flops / avg_time) / 1e12
    
    return {
        "tflops": tflops,
        "avg_time_s": avg_time
    }

@Context(DEV="HIP")
def benchmark_bandwidth(R: int = 8192, C: int = 8192, warmup: int = 10, iterations: int = 50) -> Dict[str, Any]:
    a = Tensor.randn(R, C, dtype=dtypes.float32).realize()
    b = Tensor.randn(R, C, dtype=dtypes.float32).realize()
    
    @TinyJit
    def run_test(a, b):
        (a + b).realize()
    
    for _ in range(warmup):
        run_test(a, b)
        device.Device.default.synchronize()
        
    GlobalCounters.reset()
        
    start_time = time.perf_counter()
    for _ in range(iterations):
        run_test(a, b)
        device.Device.default.synchronize()
    end_time = time.perf_counter()
    
    avg_time = (end_time - start_time) / iterations
    # (Read A, Read B, Write C) * 4 bytes (float32)
    total_bytes_moved = 3 * (R * C) * 4
    gb_s = (total_bytes_moved / avg_time) / 1e9
    
    return {
        "gb_s": gb_s,
        "avg_time_s": avg_time
    }