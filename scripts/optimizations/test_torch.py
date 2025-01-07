import torch
import torch
import time
import numpy as np

n = 10
model_paths = (
    "dependence/ts/kl-f4_encoder.ts",
    "dependence/ts/kl-f4_decoder.ts",
    "dependence/ts/kl-f16_encoder.ts",
    "dependence/ts/kl-f16_decoder.ts",
)
test_sizes = (
    (1, 3, 512, 512),
    (1, 3, 128, 128),
    (1, 3, 512, 512),
    (1, 16, 32, 32),
)

print("Приступаем к испытаниям...")
for model_path, test_size in zip(model_paths, test_sizes):
    model = torch.jit.load(model_path).type(torch.float16).cuda()
    print()
    print("=== ЗАГРУЖЕНА МОДЕЛЬ: {} ===".format(model_path))

    all_res = []
    for i in range(n):
        print("- Испытание {} -".format(i))
        input_data = torch.rand(*test_size, dtype=torch.float16, device="cuda")
        torch.cuda.synchronize()
        a = time.time()
        output_data = model(input_data)
        torch.cuda.synchronize()
        b = time.time()
        res = (b - a) * 1000
        all_res.append(res)
        print(res, "мс")
    all_res = np.array(all_res)
    print("-> Среднее время выполнения модели: {} мс".format(all_res.mean()))
    print("-> Среднее время выполнения модели с пятого: {} мс".format(all_res[5:].mean()))

print()
print("Испытания успешно завершены.")
