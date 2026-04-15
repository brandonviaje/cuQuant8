import subprocess
import matplotlib.pyplot as plt

sizes = [512, 1024, 2048, 4096, 6144, 8192]
times_ms = []
tflops_list = []

print("Starting Benchmark Suite...")

# test out each size
for size in sizes:
    print(f"Running size {size}x{size}...")
    
    # run c++ exec
    result = subprocess.run(['../build/cuQuant8', str(size)], capture_output=True, text=True)
    
    try:
        # grab number in pipeline
        time_str = result.stdout.split("Pipeline completed in: ")[1].split(" ms")[0]
        time_ms = float(time_str)
        times_ms.append(time_ms)
        
        # compute TFLOPS
        time_sec = time_ms / 1000.0
        operations = 2 * (size ** 3)
        tflops = operations / (time_sec * (10**12))
        tflops_list.append(tflops)
        
    except IndexError:
        print(f"Error parsing output for size {size}. Did the program crash?")
        print("Raw output:", result.stdout)
        break

# plot
fig, ax1 = plt.subplots(figsize=(10, 6))

# plot exec time
color = 'tab:red'
ax1.set_xlabel('Matrix Size (M = N = K)')
ax1.set_ylabel('Execution Time (ms)', color=color)
ax1.plot(sizes, times_ms, marker='o', color=color, linewidth=2, label="Time (ms)")
ax1.tick_params(axis='y', labelcolor=color)
ax1.grid(True, linestyle='--', alpha=0.6)

# plot TFLOPS 
ax2 = ax1.twinx()  
color = 'tab:blue'
ax2.set_ylabel('Throughput (TFLOPS)', color=color)
ax2.plot(sizes, tflops_list, marker='s', color=color, linewidth=2, label="TFLOPS")
ax2.tick_params(axis='y', labelcolor=color)

plt.title('INT8 Tensor Core Pipeline Performance')
fig.tight_layout()

# save graph as image
plt.savefig('benchmark_results.png', dpi=300)
print("Benchmarking complete! Graph saved as 'benchmark_results.png'.")
