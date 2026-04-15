import subprocess
import matplotlib.pyplot as plt

sizes = [512, 1024, 2048, 4096, 6144, 8192]
int8_times = []
fp32_times = []
mse_list = []
tflops_list = []

print("Starting FP32 vs INT8 Benchmark Suite...")

# test different matrix sizes
for size in sizes:
    print(f"Running size {size}x{size}...")
    result = subprocess.run(['../build/cuQuant8', str(size)], capture_output=True, text=True)
    
    try:
        lines = result.stdout.split('\n')

        # extract values
        int8_ms = float([line for line in lines if "INT8 Time:" in line][0].split(": ")[1].replace(" ms", ""))
        fp32_ms = float([line for line in lines if "FP32 Time:" in line][0].split(": ")[1].replace(" ms", ""))
        matmul_ms = float([line for line in lines if "MatMul Time:" in line][0].split(": ")[1].replace(" ms", ""))
        mse = float([line for line in lines if "MSE:" in line][0].split(": ")[1])
        
        # compute TFLOPS
        ops = 2.0 * (size ** 3)
        tflops = (ops / (matmul_ms / 1000.0)) / (10**12)

        # add to lists to build graph
        int8_times.append(int8_ms)
        fp32_times.append(fp32_ms)
        mse_list.append(mse)
        tflops_list.append(tflops)
        
        print(f"  -> MatMul: {tflops:.2f} TFLOPS | MSE: {mse:.4f}")
        
    except Exception as e:
        print(f"Error parsing output for size {size}: {e}")
        break

# plot results
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))

# plot exec time comparison 
ax1.plot(sizes, int8_times, marker='o', label="INT8 Pipeline")
ax1.plot(sizes, fp32_times, marker='s', label="FP32 Baseline")
ax1.set_title('Total Pipeline Latency (ms)')
ax1.legend()
ax1.grid(True)

# plot correctness (MSE)
ax2.plot(sizes, mse_list, marker='^', color='green')
ax2.set_title('Quantization Error (MSE)')
ax2.set_yscale('log')
ax2.grid(True)

# plot throughput (TFLOPS)
ax3.plot(sizes, tflops_list, marker='D', color='red')
ax3.set_title('Isolated Kernel Throughput (TFLOPS)')
ax3.set_ylabel('TFLOPS')
ax3.grid(True)

fig.tight_layout()
plt.savefig('final_benchmark.png')
print("\nDone! saved as 'final_benchmark.png'.")
