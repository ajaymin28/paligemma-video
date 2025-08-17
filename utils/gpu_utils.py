import torch
import psutil
import os

getPercent = lambda current,total: round((current*100)/total,2) 

# def covert_to_percent(total, current):
#     return round((current*100)/total,2)

def memory_stats(get_dict=False, print_mem_usage=True, device=None, custom_label=""):
    """
    Provides memory stats for, cpu%, ram% for process, 
    """

    MB_eval_exp = 1024 ** 2
    GB_eval_exp = 1024 ** 3

    # Initialize stats dictionary
    stats = {}

    # Process-specific stats (for the current Python process)
    try:
        process = psutil.Process(os.getpid())
        stats["ram/percent"] = round(process.memory_percent(), 2)  # Process RAM usage in %
        stats["ram/usage_mb"] = round(process.memory_info().rss / (MB_eval_exp), 2)  # Process RAM usage in megabytes
    except Exception as e:
        stats["ram/percent"] = 0
        stats["ram/usage_mb"] = 0
        print(f"Error getting process stats: {e}")

    # System-wide CPU usage
    try:
        stats["cpu/percent"] = psutil.cpu_percent()  # CPU usage % over 1-second interval
        stats["cpu/count"] = psutil.cpu_count(logical=True)  # Number of logical CPU cores
    except Exception as e:
        stats["cpu/percent"] = 0
        stats["cpu/count"] = 0
        print(f"Error getting CPU stats: {e}")

    # System-wide RAM usage
    try:
        ram = psutil.virtual_memory()
        stats["ram/total"] = round(ram.total / (MB_eval_exp), 2)  # Total RAM in megabytes
        stats["ram/used"] = round(ram.used / (MB_eval_exp), 2)  # Used RAM in megabytes
        stats["ram/free"] = round(ram.free / (MB_eval_exp), 2)  # Free RAM in megabytes
        stats["ram/system_percent"] = ram.percent  # System RAM usage in %
    except Exception as e:
        stats["ram/total"] = 0
        stats["ram/used"] = 0
        stats["ram/free"] = 0
        stats["ram/system_percent"] = 0
        print(f"Error getting system RAM stats: {e}")

    # System-wide disk usage
    try:
        disk = psutil.disk_usage('/')
        stats["disk/total"] = round(disk.total / (GB_eval_exp), 2)  # Total disk in gigabytes
        stats["disk/used"] = round(disk.used / (GB_eval_exp), 2)  # Used disk in gigabytes
        stats["disk/free"] = round(disk.free / (GB_eval_exp), 2)  # Free disk in gigabytes
        stats["disk/percent"] = disk.percent  # Disk usage in %
    except Exception as e:
        stats["disk/total"] = 0
        stats["disk/used"] = 0
        stats["disk/free"] = 0
        stats["disk/percent"] = 0
        print(f"Error getting disk stats: {e}")

    # GPU stats (if CUDA is available)
    if torch.cuda.is_available():
        try:
            device = torch.device("cuda")
            cuda_freeMem, cuda_total = torch.cuda.mem_get_info()
            stats["cuda/total"] = round(cuda_total / MB_eval_exp, 2)  # Total GPU memory in megabytes
            stats["cuda/free"] = round(cuda_freeMem / MB_eval_exp, 2)  # Free GPU memory in megabytes
            stats["cuda/free_percent"] = round(stats["cuda/free"]*100 / stats["cuda/total"], 2) if stats["cuda/total"]> 0 else 0  # Free GPU memory in percent
            stats["cuda/used"] = round((cuda_total - cuda_freeMem) / MB_eval_exp, 2)  # Used GPU memory in megabytes
            stats["cuda/used_percent"] = round((stats["cuda/used"] / stats["cuda/total"]) * 100, 2) if stats["cuda/total"] > 0 else 0
            stats["cuda/allocated"] = round(torch.cuda.memory_allocated() / MB_eval_exp, 3)  # Allocated GPU memory in megabytes
            stats["cuda/allocated_percent"] = round((stats["cuda/allocated"] / stats["cuda/total"]) * 100, 2) if stats["cuda/total"] > 0 else 0
            stats["cuda/reserved"] = round(torch.cuda.memory_reserved() / MB_eval_exp, 3)  # Reserved GPU memory in megabytes
            stats["cuda/reserved_percent"] = round((stats["cuda/reserved"] / stats["cuda/total"]) * 100, 2) if stats["cuda/total"] > 0 else 0
            stats["cuda/peak_vram_allocated"] = round(torch.cuda.max_memory_allocated(device) / MB_eval_exp, 3)  # Peak VRAM in megabytes
            stats["cuda/peak_vram_allocated_percent"] = round((stats["cuda/peak_vram_allocated"] / stats["cuda/total"]) * 100, 2) if stats["cuda/total"] > 0 else 0
        except Exception as e:
            stats["cuda/total"] = 0
            stats["cuda/free"] = 0
            stats["cuda/free_percent"] = 0
            stats["cuda/used"] = 0
            stats["cuda/used_percent"] = 0
            stats["cuda/allocated"] = 0
            stats["cuda/allocated_percent"] = 0
            stats["cuda/reserved"] = 0
            stats["cuda/reserved_percent"] = 0
            stats["cuda/peak_vram_allocated"] = 0
            stats["cuda/peak_vram_allocated_percent"] = 0
            print(f"Error getting GPU stats: {e}")

    custom_label_string = f"[{custom_label}]" if custom_label!="" else ""
    if print_mem_usage:
        print(
            f"{custom_label_string}"
            f"CPU: {stats['cpu/percent']:.2f}% | "
            f"RAM: {stats['ram/percent']:.2f}% ({stats['ram/usage_mb']:.2f} MB) | "
            f"GPU: T {stats['cuda/total']:.2f} MB ({stats['cuda/used_percent']:.2f}%) | "
            f"F {stats['cuda/free']:.2f} MB ({stats['cuda/free_percent']:.2f}%) | "
            f"A {stats['cuda/allocated']:.3f} MB ({stats['cuda/allocated_percent']:.2f}%) | "
            f"R {stats['cuda/reserved']:.3f} MB ({stats['cuda/reserved_percent']:.2f}%) | "
            f"P {stats['cuda/peak_vram_allocated']:.3f} MB ({stats['cuda/peak_vram_allocated_percent']:.2f}%)"
        )
    if get_dict:
        return stats