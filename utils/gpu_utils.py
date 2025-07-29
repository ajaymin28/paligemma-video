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
        stats["ram_percent"] = round(process.memory_percent(), 2)  # Process RAM usage in %
        stats["ram"] = round(process.memory_info().rss / (MB_eval_exp), 2)  # Process RAM usage in megabytes
    except Exception as e:
        stats["ram_percent"] = 0
        stats["ram"] = 0
        print(f"Error getting process stats: {e}")

    # System-wide CPU usage
    try:
        stats["cpu_percent"] = psutil.cpu_percent()  # CPU usage % over 1-second interval
        stats["cpu_count"] = psutil.cpu_count(logical=True)  # Number of logical CPU cores
    except Exception as e:
        stats["cpu_percent"] = 0
        stats["cpu_count"] = 0
        print(f"Error getting CPU stats: {e}")

    # System-wide RAM usage
    try:
        ram = psutil.virtual_memory()
        stats["ram_total"] = round(ram.total / (MB_eval_exp), 2)  # Total RAM in megabytes
        stats["ram_used"] = round(ram.used / (MB_eval_exp), 2)  # Used RAM in megabytes
        stats["ram_free"] = round(ram.free / (MB_eval_exp), 2)  # Free RAM in megabytes
        stats["ram_system_percent"] = ram.percent  # System RAM usage in %
    except Exception as e:
        stats["ram_total"] = 0
        stats["ram_used"] = 0
        stats["ram_free"] = 0
        stats["ram_system_percent"] = 0
        print(f"Error getting system RAM stats: {e}")

    # System-wide disk usage
    try:
        disk = psutil.disk_usage('/')
        stats["disk_total"] = round(disk.total / (GB_eval_exp), 2)  # Total disk in gigabytes
        stats["disk_used"] = round(disk.used / (GB_eval_exp), 2)  # Used disk in gigabytes
        stats["disk_free"] = round(disk.free / (GB_eval_exp), 2)  # Free disk in gigabytes
        stats["disk_percent"] = disk.percent  # Disk usage in %
    except Exception as e:
        stats["disk_total"] = 0
        stats["disk_used"] = 0
        stats["disk_free"] = 0
        stats["disk_percent"] = 0
        print(f"Error getting disk stats: {e}")

    # GPU stats (if CUDA is available)
    if torch.cuda.is_available():
        try:
            device = torch.device("cuda")
            cuda_freeMem, cuda_total = torch.cuda.mem_get_info()
            stats["cuda_total"] = round(cuda_total / MB_eval_exp, 2)  # Total GPU memory in megabytes
            stats["cuda_free"] = round(cuda_freeMem / MB_eval_exp, 2)  # Free GPU memory in megabytes
            stats["cuda_free_percent"] = round(stats["cuda_free"]*100 / stats["cuda_total"], 2) if stats["cuda_total"]> 0 else 0  # Free GPU memory in percent
            stats["cuda_used"] = round((cuda_total - cuda_freeMem) / MB_eval_exp, 2)  # Used GPU memory in megabytes
            stats["cuda_used_percent"] = round((stats["cuda_used"] / stats["cuda_total"]) * 100, 2) if stats["cuda_total"] > 0 else 0
            stats["cuda_allocated"] = round(torch.cuda.memory_allocated() / MB_eval_exp, 3)  # Allocated GPU memory in megabytes
            stats["cuda_allocated_percent"] = round((stats["cuda_allocated"] / stats["cuda_total"]) * 100, 2) if stats["cuda_total"] > 0 else 0
            stats["cuda_reserved"] = round(torch.cuda.memory_reserved() / MB_eval_exp, 3)  # Reserved GPU memory in megabytes
            stats["cuda_reserved_percent"] = round((stats["cuda_reserved"] / stats["cuda_total"]) * 100, 2) if stats["cuda_total"] > 0 else 0
            stats["cuda_peak_vram_allocated"] = round(torch.cuda.max_memory_allocated(device) / MB_eval_exp, 3)  # Peak VRAM in megabytes
            stats["cuda_peak_vram_allocated_percent"] = round((stats["cuda_peak_vram_allocated"] / stats["cuda_total"]) * 100, 2) if stats["cuda_total"] > 0 else 0
        except Exception as e:
            stats["cuda_total"] = 0
            stats["cuda_free"] = 0
            stats["cuda_used"] = 0
            stats["cuda_used_percent"] = 0
            stats["cuda_allocated"] = 0
            stats["cuda_allocated_percent"] = 0
            stats["cuda_reserved"] = 0
            stats["cuda_reserved_percent"] = 0
            stats["cuda_peak_vram_allocated"] = 0
            stats["cuda_peak_vram_allocated_percent"] = 0
            print(f"Error getting GPU stats: {e}")

    custom_label_string = f"[{custom_label}]" if custom_label!="" else ""
    if print_mem_usage:
        print(
            f"{custom_label_string}"
            f"CPU: {stats['cpu_percent']:.2f}% | "
            f"RAM: {stats['ram_percent']:.2f}% ({stats['ram']:.2f} MB) | "
            f"GPU: T {stats['cuda_total']:.2f} MB ({stats['cuda_used_percent']:.2f}%) | "
            f"F {stats['cuda_free']:.2f} MB ({stats['cuda_free_percent']:.2f}%) | "
            f"A {stats['cuda_allocated']:.3f} MB ({stats['cuda_allocated_percent']:.2f}%) | "
            f"R {stats['cuda_reserved']:.3f} MB ({stats['cuda_reserved_percent']:.2f}%) | "
            f"P {stats['cuda_peak_vram_allocated']:.3f} MB ({stats['cuda_peak_vram_allocated_percent']:.2f}%)"
        )
    if get_dict:
        return stats