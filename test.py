import torch
import time

def check_torch_gpu():
    print("🔍 PyTorch 檢測報告：")

    if torch.cuda.is_available():
        print("✅ CUDA 可用！")
        print(f"📟 GPU 名稱：{torch.cuda.get_device_name(0)}")
        print(f"💾 GPU 記憶體總量：{round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)} GB")
        print(f"🔥 GPU 數量：{torch.cuda.device_count()}")

        # Tensor 運算測試
        a = torch.rand(10000, 10000, device="cuda")
        b = torch.rand(10000, 10000, device="cuda")
        torch.cuda.synchronize()
        start = time.time()
        c = torch.matmul(a, b)
        torch.cuda.synchronize()
        end = time.time()
        print(f"⚙️ GPU 運算時間：{end - start:.4f} 秒")

        # PYNVML 狀態檢查
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            print(f"📊 GPU 使用記憶體：{round(mem_info.used / (1024**3), 2)} / {round(mem_info.total / (1024**3), 2)} GB")
            print(f"🔋 GPU 溫度：{pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)} °C")
            pynvml.nvmlShutdown()
        except Exception as e:
            print(f"❌ pynvml 錯誤：{e}")
    else:
        print("❌ 沒有偵測到可用的 GPU，快去裝顯卡或檢查 CUDA 環境！🥲")

check_torch_gpu()

import pandas as pd
with open('final_dataset.csv', 'r', encoding='utf-8') as f:
    head = [next(f) for _ in range(5)]
print(''.join(head))