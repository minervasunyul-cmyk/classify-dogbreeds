"""
Script to check computer specifications important for ML model training.
Outputs results to specs.md in README-style markdown format.
"""

import platform
import psutil
import sys
from datetime import datetime

def get_cpu_info():
    """Get CPU information."""
    cpu_info = {
        'name': platform.processor(),
        'cores_physical': psutil.cpu_count(logical=False),
        'cores_logical': psutil.cpu_count(logical=True),
        'frequency': psutil.cpu_freq().current if psutil.cpu_freq() else 'N/A'
    }
    return cpu_info

def get_ram_info():
    """Get RAM information."""
    ram = psutil.virtual_memory()
    return {
        'total_gb': round(ram.total / (1024**3), 2),
        'available_gb': round(ram.available / (1024**3), 2),
        'used_gb': round(ram.used / (1024**3), 2),
        'percent': ram.percent
    }

def get_gpu_info():
    """Get GPU information (NVIDIA, AMD, Intel)."""
    gpu_info = []
    
    # Check for NVIDIA GPU
    try:
        import pynvml
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            memory_total_gb = round(memory_info.total / (1024**3), 2)
            memory_used_gb = round(memory_info.used / (1024**3), 2)
            
            # Get CUDA version
            try:
                cuda_version = pynvml.nvmlSystemGetCudaDriverVersion()
                cuda_version_str = f"{cuda_version // 1000}.{(cuda_version % 1000) // 10}"
            except:
                cuda_version_str = "N/A"
            
            gpu_info.append({
                'type': 'NVIDIA',
                'name': name,
                'memory_total_gb': memory_total_gb,
                'memory_used_gb': memory_used_gb,
                'cuda_version': cuda_version_str
            })
        
        pynvml.nvmlShutdown()
    except ImportError:
        pass
    except Exception as e:
        pass
    
    # Check for PyTorch GPU
    try:
        import torch
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                memory_total = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                # Only add if not already found via pynvml
                if not any(gpu['name'] == gpu_name for gpu in gpu_info):
                    gpu_info.append({
                        'type': 'NVIDIA (via PyTorch)',
                        'name': gpu_name,
                        'memory_total_gb': round(memory_total, 2),
                        'memory_used_gb': 'N/A',
                        'cuda_version': f"{torch.version.cuda}" if hasattr(torch.version, 'cuda') else "N/A"
                    })
    except ImportError:
        pass
    
    # Check for TensorFlow GPU
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                gpu_info.append({
                    'type': 'GPU (via TensorFlow)',
                    'name': gpu.name,
                    'memory_total_gb': 'N/A',
                    'memory_used_gb': 'N/A',
                    'cuda_version': 'N/A'
                })
    except ImportError:
        pass
    
    return gpu_info

def get_storage_info():
    """Get storage information."""
    storage_info = []
    partitions = psutil.disk_partitions()
    
    for partition in partitions:
        try:
            usage = psutil.disk_usage(partition.mountpoint)
            storage_info.append({
                'device': partition.device,
                'mountpoint': partition.mountpoint,
                'fstype': partition.fstype,
                'total_gb': round(usage.total / (1024**3), 2),
                'used_gb': round(usage.used / (1024**3), 2),
                'free_gb': round(usage.free / (1024**3), 2),
                'percent': usage.percent
            })
        except PermissionError:
            continue
    
    return storage_info

def get_python_info():
    """Get Python and ML library versions."""
    info = {
        'python_version': sys.version.split()[0],
        'pytorch_version': None,
        'tensorflow_version': None,
        'cuda_available': False
    }
    
    try:
        import torch
        info['pytorch_version'] = torch.__version__
        info['cuda_available'] = torch.cuda.is_available()
        if info['cuda_available']:
            info['cuda_version'] = torch.version.cuda if hasattr(torch.version, 'cuda') else 'N/A'
    except ImportError:
        pass
    
    try:
        import tensorflow as tf
        info['tensorflow_version'] = tf.__version__
    except ImportError:
        pass
    
    return info

def generate_markdown():
    """Generate markdown file with all specifications."""
    
    cpu = get_cpu_info()
    ram = get_ram_info()
    gpus = get_gpu_info()
    storage = get_storage_info()
    python_info = get_python_info()
    
    md_content = f"""# Computer Specifications

> Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## System Information

- **OS**: {platform.system()} {platform.release()}
- **Architecture**: {platform.machine()}
- **Platform**: {platform.platform()}

## CPU

- **Processor**: {cpu['name']}
- **Physical Cores**: {cpu['cores_physical']}
- **Logical Cores**: {cpu['cores_logical']}
- **Frequency**: {cpu['frequency']} MHz

## Memory (RAM)

- **Total**: {ram['total_gb']} GB
- **Available**: {ram['available_gb']} GB
- **Used**: {ram['used_gb']} GB ({ram['percent']}%)

## GPU

"""
    
    if gpus:
        for i, gpu in enumerate(gpus, 1):
            md_content += f"""### GPU {i}: {gpu['name']}

- **Type**: {gpu['type']}
- **Total Memory**: {gpu['memory_total_gb']} GB
"""
            if gpu['memory_used_gb'] != 'N/A':
                md_content += f"- **Used Memory**: {gpu['memory_used_gb']} GB\n"
            if gpu['cuda_version'] != 'N/A':
                md_content += f"- **CUDA Version**: {gpu['cuda_version']}\n"
            md_content += "\n"
    else:
        md_content += "No GPU detected.\n\n"
    
    md_content += """## Storage

"""
    
    for disk in storage:
        md_content += f"""### {disk['device']} ({disk['mountpoint']})

- **File System**: {disk['fstype']}
- **Total**: {disk['total_gb']} GB
- **Used**: {disk['used_gb']} GB ({disk['percent']}%)
- **Free**: {disk['free_gb']} GB

"""
    
    md_content += """## Python & ML Libraries

"""
    
    md_content += f"- **Python Version**: {python_info['python_version']}\n"
    
    if python_info['pytorch_version']:
        md_content += f"- **PyTorch Version**: {python_info['pytorch_version']}\n"
        if python_info['cuda_available']:
            md_content += f"  - CUDA Available: Yes\n"
            if 'cuda_version' in python_info:
                md_content += f"  - CUDA Version: {python_info['cuda_version']}\n"
        else:
            md_content += f"  - CUDA Available: No\n"
    
    if python_info['tensorflow_version']:
        md_content += f"- **TensorFlow Version**: {python_info['tensorflow_version']}\n"
    
    if not python_info['pytorch_version'] and not python_info['tensorflow_version']:
        md_content += "- No ML frameworks detected (PyTorch/TensorFlow not installed)\n"
    
    md_content += "\n---\n\n*This file was automatically generated by check_specs.py*\n"
    
    return md_content

def main():
    """Main function to generate and save specs.md."""
    print("Checking computer specifications...")
    
    try:
        md_content = generate_markdown()
        
        with open('specs.md', 'w', encoding='utf-8') as f:
            f.write(md_content)
        
        print("Specifications saved to specs.md")
        
    except Exception as e:
        print(f"Error generating specs: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
