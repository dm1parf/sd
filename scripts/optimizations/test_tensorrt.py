import onnx
import onnxruntime as ort
import numpy as np
import time
import os

n = 10
model_names = (
    "kl-f4_encoder.onnx",
    "kl-f4_decoder.onnx",
    "kl-f16_encoder.onnx",
    "kl-f16_decoder.onnx",
)
# dependence/onnx/kl-f4_encoder.onnx
# dependence/onnx/kl-f4_decoder.onnx
# dependence/onnx/kl-f16_encoder.onnx
# dependence/onnx/kl-f16_decoder.onnx
model_path = "dependence/onnx"
optimized_model_path = "dependence/onnx/optimized"
engine_cache_path = "./dependence/onnx/engine_cache"
timing_cache_path = "./dependence/onnx/timing_cache"
this_provider = "TensorrtExecutionProvider"

test_sizes = (
    (1, 3, 512, 512),
    (1, 3, 128, 128),
    (1, 3, 512, 512),
    (1, 16, 32, 32),
)

# ORIN
## 9_961_472_000,  # 9500 Mb
# A100
## 37_580_963_840,  # 35 Gb

providers = [
    ('TensorrtExecutionProvider', {
        'device_id': 0,
        'trt_max_workspace_size': 37_580_963_840,  # 35 Gb
        'trt_fp16_enable': True,

        "trt_timing_cache_enable": True,
        "trt_timing_cache_path": timing_cache_path,
        "trt_engine_cache_enable": True,
        "trt_engine_cache_path": engine_cache_path,
    }),
]


print("Приступаем к испытаниям...")
for model_name, test_size in zip(model_names, test_sizes):
    model_fullpath = os.path.join(model_path, model_name)

    model = onnx.load(model_fullpath)
    onnx.checker.check_model(model)

    ort_sess = ort.InferenceSession(model_fullpath, providers=providers)

    print()
    print("=== ЗАГРУЖЕНА МОДЕЛЬ: {} ===".format(model_name))

    all_res = []
    for i in range(n):
        print("- Испытание {} -".format(i))
        input_data = np.random.random(size=test_size).astype(np.float16)
        a = time.time()
        output_data = ort_sess.run(None, {"input": input_data})[0]
        b = time.time()
        print(output_data.shape)
        res = (b - a) * 1000
        all_res.append(res)
        print(res, "мс")
    all_res = np.array(all_res)
    print("-> Среднее время выполнения модели: {} мс".format(all_res.mean()))
    print("-> Среднее время выполнения модели без первого: {} мс".format(all_res[1:].mean()))

print()
print("Испытания успешно завершены.")
