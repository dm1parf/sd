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
model_path = "dependence/onnx"
optimized_model_path = "dependence/onnx/optimized"
this_provider = "CUDAExecutionProvider"

test_sizes = (
    (1, 3, 512, 512),
    (1, 3, 128, 128),
    (1, 3, 512, 512),
    (1, 16, 32, 32),
)

print("Все провайдеры:", ort.get_all_providers())

# ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'MIGraphXExecutionProvider',
# 'ROCMExecutionProvider', 'OpenVINOExecutionProvider', 'DnnlExecutionProvider', 'TvmExecutionProvider',
# 'VitisAIExecutionProvider', 'QNNExecutionProvider', 'NnapiExecutionProvider', 'JsExecutionProvider',
# 'CoreMLExecutionProvider', 'ArmNNExecutionProvider', 'ACLExecutionProvider', 'DmlExecutionProvider',
# 'RknpuExecutionProvider', 'WebNNExecutionProvider', 'XnnpackExecutionProvider', 'CANNExecutionProvider',
# 'AzureExecutionProvider', 'CPUExecutionProvider']

print("Поддерживаемые провайдеры:", ort.get_available_providers())

# ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']


print("Приступаем к испытаниям...")
for model_name, test_size in zip(model_names, test_sizes):
    model_fullpath = os.path.join(model_path, model_name)
    optimized_model_fullpath = os.path.join(optimized_model_path, this_provider + "_" + model_name)

    sess_opt = ort.SessionOptions()
    if os.path.isfile(optimized_model_fullpath):
        # Оптимизации уже были проведены.
        print("= Модель {} уже оптимизирована. Загружаем... =".format(model_name))

        model = onnx.load(optimized_model_fullpath)
        onnx.checker.check_model(model)
        sess_opt.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL

        ort_sess = ort.InferenceSession(optimized_model_fullpath, sess_opt, providers=[this_provider])
    else:
        # Оптимизации требуется провести.
        print("= Модель {} требует оптимизации. Загружаем и оптимизируем... =".format(model_name))

        model = onnx.load(model_fullpath)
        onnx.checker.check_model(model)
        sess_opt.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # sess_opt.log_severity_level = 0  # Verbose
        sess_opt.optimized_model_filepath = optimized_model_fullpath

        ort_sess = ort.InferenceSession(model_fullpath, sess_opt, providers=[this_provider])

    print()
    print("=== ЗАГРУЖЕНА МОДЕЛЬ: {} ===".format(model_name))
    print("Входы:")
    for inper in ort_sess.get_inputs():
        print(">", inper.name, inper.shape)
    print("Выходы:")
    for outer in ort_sess.get_outputs():
        print(">", outer.name, outer.shape)
    print("Пусть к исходной модели:")
    print(model_fullpath)
    print("Пусть к оптимизированной исходной модели:")
    print(optimized_model_fullpath)

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
