#### 1.small dataset 样本集

```shell
✅ CUDA available: NVIDIA L20
   GPU Memory: 44.4 GB
🔧 Loss weights: SYS_W=0.05, USER_W=0.2, ASSIST_W=1.0
🔹 Loading tokenizer from Qwen/Qwen3-1.7B...
2026-01-19 16:28:32,628 - modelscope - INFO - Intra-cloud acceleration enabled for downloading from Qwen/Qwen3-1.7B
Downloading Model from https://www.modelscope.cn to directory: /home/yushaolong/.cache/modelscope/hub/models/Qwen/Qwen3-1.7B
🔹 Loading model from Qwen/Qwen3-1.7B...
   Trainable tokens (assistant only): 93,604 (96.7%)
🔹 Applying LoRA...
trainable params: 6,422,528 || all params: 1,726,997,504 || trainable%: 0.3719
🔹 Initializing Trainer...
The model is already on multiple devices. Skipping the move to device specified in `args`.
🚀 Starting training...
{'loss': 5.5423, 'grad_norm': 7.162075996398926, 'learning_rate': 0.0, 'epoch': 0.2}
{'loss': 5.155, 'grad_norm': 5.865916728973389, 'learning_rate': 0.0001, 'epoch': 0.4}
{'loss': 4.6727, 'grad_norm': 5.424260139465332, 'learning_rate': 9.874639560909117e-05, 'epoch': 0.6}
{'loss': 4.7072, 'grad_norm': 6.196091175079346, 'learning_rate': 9.504844339512095e-05, 'epoch': 0.8}
{'loss': 4.2288, 'grad_norm': 4.964895248413086, 'learning_rate': 8.90915741234015e-05, 'epoch': 1.0}
{'loss': 4.0907, 'grad_norm': 4.816958427429199, 'learning_rate': 8.117449009293668e-05, 'epoch': 1.2}
{'loss': 3.7, 'grad_norm': 4.345840930938721, 'learning_rate': 7.169418695587791e-05, 'epoch': 1.4}
{'loss': 3.5568, 'grad_norm': 3.705782175064087, 'learning_rate': 6.112604669781572e-05, 'epoch': 1.6}
{'loss': 3.3405, 'grad_norm': 3.225646495819092, 'learning_rate': 5e-05, 'epoch': 1.8}
{'loss': 3.1537, 'grad_norm': 2.5605733394622803, 'learning_rate': 3.887395330218429e-05, 'epoch': 2.0}
{'loss': 3.264, 'grad_norm': 2.4719455242156982, 'learning_rate': 2.8305813044122097e-05, 'epoch': 2.2}
{'loss': 2.9979, 'grad_norm': 2.23502516746521, 'learning_rate': 1.8825509907063327e-05, 'epoch': 2.4}
{'loss': 3.2287, 'grad_norm': 2.1159286499023438, 'learning_rate': 1.090842587659851e-05, 'epoch': 2.6}
{'loss': 3.1506, 'grad_norm': 2.0219106674194336, 'learning_rate': 4.951556604879048e-06, 'epoch': 2.8}
{'loss': 3.14, 'grad_norm': 2.0334067344665527, 'learning_rate': 1.2536043909088191e-06, 'epoch': 3.0}
{'train_runtime': 57.4716, 'train_samples_per_second': 4.176, 'train_steps_per_second': 0.261, 'train_loss': 3.861918576558431, 'epoch': 3.0}
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 15/15 [00:57<00:00,  3.83s/it]

✅ Done. LoRA adapter saved to: qwen_lora_adapter_0119_s
```

显卡状态:
```
nvidia-smi
Mon Jan 19 16:10:39 2026
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 555.42.02              Driver Version: 555.42.02      CUDA Version: 12.5     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA L20                     On  |   00000000:00:03.0 Off |                  Off |
| N/A   27C    P8             25W /  350W |       4MiB /  49140MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
|   1  NVIDIA L20                     On  |   00000000:00:04.0 Off |                    0 |
| N/A   57C    P0            288W /  350W |   25307MiB /  46068MiB |     54%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI        PID   Type   Process name                              GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    1   N/A  N/A   4085961      C   python                                      25298MiB |
+-----------------------------------------------------------------------------------------+
```


#### 2.large dataset样本集
```shell
CUDA_VISIBLE_DEVICES=1 python finetune_gpu_weighted.py
✅ CUDA available: NVIDIA L20
   GPU Memory: 44.4 GB
🔧 Loss weights: SYS_W=0.05, USER_W=0.2, ASSIST_W=1.0
🔹 Loading tokenizer from Qwen/Qwen3-14B...
2026-01-19 16:28:32,628 - modelscope - INFO - Intra-cloud acceleration enabled for downloading from Qwen/Qwen3-14B
Downloading Model from https://www.modelscope.cn to directory: /home/yushaolong/.cache/modelscope/hub/models/Qwen/Qwen3-14B
🔹 Loading model from Qwen/Qwen3-14B...
2026-01-19 16:28:33,586 - modelscope - INFO - Intra-cloud acceleration enabled for downloading from Qwen/Qwen3-14B
Downloading Model from https://www.modelscope.cn to directory: /home/yushaolong/.cache/modelscope/hub/models/Qwen/Qwen3-14B
`torch_dtype` is deprecated! Use `dtype` instead!
Loading checkpoint shards: 100%|███████████████████████████████████████████████████████████████████████████████████| 8/8 [00:03<00:00,  2.27it/s]
🔹 Loading dataset from datasets/train_0119_l.jsonl...
   Total samples: 400
   Total tokens: 480,747
   Token mix (approx): assistant=400,892 (83.4%), user=63,855 (13.3%), system=16,000 (3.3%)
🔹 Applying LoRA...
trainable params: 20,971,520 || all params: 14,789,278,720 || trainable%: 0.1418
🔹 Initializing WeightedLossTrainer...
The model is already on multiple devices. Skipping the move to device specified in `args`.
🚀 Starting training...
{'loss': 44.7395, 'grad_norm': 22.035781860351562, 'learning_rate': 0.0, 'epoch': 0.04}
{'loss': 44.5203, 'grad_norm': 20.738832473754883, 'learning_rate': 3.3333333333333335e-05, 'epoch': 0.08}
{'loss': 43.0458, 'grad_norm': 20.302322387695312, 'learning_rate': 6.666666666666667e-05, 'epoch': 0.12}
{'loss': 44.9156, 'grad_norm': 23.323503494262695, 'learning_rate': 0.0001, 'epoch': 0.16}
{'loss': 43.175, 'grad_norm': 22.33148956298828, 'learning_rate': 9.99524110790929e-05, 'epoch': 0.2}
{'loss': 40.1386, 'grad_norm': 20.250566482543945, 'learning_rate': 9.980973490458728e-05, 'epoch': 0.24}
{'loss': 37.6271, 'grad_norm': 16.953012466430664, 'learning_rate': 9.957224306869053e-05, 'epoch': 0.28}
{'loss': 36.5452, 'grad_norm': 13.785959243774414, 'learning_rate': 9.924038765061042e-05, 'epoch': 0.32}
{'loss': 33.4176, 'grad_norm': 8.969555854797363, 'learning_rate': 9.881480035599667e-05, 'epoch': 0.36}
{'loss': 33.2645, 'grad_norm': 6.952634334564209, 'learning_rate': 9.829629131445342e-05, 'epoch': 0.4}
{'loss': 33.4478, 'grad_norm': 5.294229984283447, 'learning_rate': 9.768584753741134e-05, 'epoch': 0.44}
{'loss': 31.055, 'grad_norm': 4.776330947875977, 'learning_rate': 9.698463103929542e-05, 'epoch': 0.48}
{'loss': 30.8342, 'grad_norm': 3.8331894874572754, 'learning_rate': 9.619397662556435e-05, 'epoch': 0.52}
{'loss': 31.0211, 'grad_norm': 4.155395984649658, 'learning_rate': 9.53153893518325e-05, 'epoch': 0.56}
{'loss': 30.4904, 'grad_norm': 4.467333793640137, 'learning_rate': 9.435054165891109e-05, 'epoch': 0.6}
{'loss': 31.5327, 'grad_norm': 5.293240070343018, 'learning_rate': 9.330127018922194e-05, 'epoch': 0.64}
{'loss': 31.8237, 'grad_norm': 5.042154312133789, 'learning_rate': 9.21695722906443e-05, 'epoch': 0.68}
{'loss': 29.6595, 'grad_norm': 4.297292232513428, 'learning_rate': 9.09576022144496e-05, 'epoch': 0.72}
{'loss': 30.937, 'grad_norm': 4.0857720375061035, 'learning_rate': 8.966766701456177e-05, 'epoch': 0.76}
{'loss': 29.059, 'grad_norm': 3.571591854095459, 'learning_rate': 8.83022221559489e-05, 'epoch': 0.8}
{'loss': 29.7147, 'grad_norm': 3.1964430809020996, 'learning_rate': 8.68638668405062e-05, 'epoch': 0.84}
{'loss': 30.2798, 'grad_norm': 3.068286895751953, 'learning_rate': 8.535533905932738e-05, 'epoch': 0.88}
{'loss': 29.7123, 'grad_norm': 2.860661745071411, 'learning_rate': 8.377951038078302e-05, 'epoch': 0.92}
{'loss': 28.9664, 'grad_norm': 2.702151298522949, 'learning_rate': 8.213938048432697e-05, 'epoch': 0.96}
{'loss': 29.837, 'grad_norm': 2.7535438537597656, 'learning_rate': 8.043807145043604e-05, 'epoch': 1.0}
{'loss': 29.2397, 'grad_norm': 2.5620779991149902, 'learning_rate': 7.86788218175523e-05, 'epoch': 1.04}
{'loss': 28.7344, 'grad_norm': 2.4864745140075684, 'learning_rate': 7.68649804173412e-05, 'epoch': 1.08}
{'loss': 28.2632, 'grad_norm': 2.528272867202759, 'learning_rate': 7.500000000000001e-05, 'epoch': 1.12}
{'loss': 29.1445, 'grad_norm': 2.734110116958618, 'learning_rate': 7.308743066175172e-05, 'epoch': 1.16}
{'loss': 28.9018, 'grad_norm': 2.4826009273529053, 'learning_rate': 7.113091308703498e-05, 'epoch': 1.2}
{'loss': 29.2566, 'grad_norm': 2.5434858798980713, 'learning_rate': 6.91341716182545e-05, 'epoch': 1.24}
{'loss': 29.3802, 'grad_norm': 2.556025266647339, 'learning_rate': 6.710100716628344e-05, 'epoch': 1.28}
{'loss': 28.5473, 'grad_norm': 2.8074989318847656, 'learning_rate': 6.503528997521366e-05, 'epoch': 1.32}
{'loss': 27.5826, 'grad_norm': 2.5049993991851807, 'learning_rate': 6.294095225512603e-05, 'epoch': 1.36}
{'loss': 28.4289, 'grad_norm': 2.7788357734680176, 'learning_rate': 6.0821980696905146e-05, 'epoch': 1.4}
{'loss': 28.6298, 'grad_norm': 2.66839599609375, 'learning_rate': 5.868240888334653e-05, 'epoch': 1.44}
{'loss': 27.5341, 'grad_norm': 2.415025234222412, 'learning_rate': 5.6526309611002594e-05, 'epoch': 1.48}
{'loss': 29.3907, 'grad_norm': 2.743967294692993, 'learning_rate': 5.435778713738292e-05, 'epoch': 1.52}
{'loss': 27.8345, 'grad_norm': 2.6551859378814697, 'learning_rate': 5.218096936826681e-05, 'epoch': 1.56}
{'loss': 28.8319, 'grad_norm': 2.617722988128662, 'learning_rate': 5e-05, 'epoch': 1.6}
{'loss': 28.6168, 'grad_norm': 2.8000078201293945, 'learning_rate': 4.781903063173321e-05, 'epoch': 1.64}
{'loss': 28.0522, 'grad_norm': 2.7135696411132812, 'learning_rate': 4.564221286261709e-05, 'epoch': 1.68}
{'loss': 28.5631, 'grad_norm': 2.8135716915130615, 'learning_rate': 4.347369038899744e-05, 'epoch': 1.72}
{'loss': 28.4275, 'grad_norm': 2.7928481101989746, 'learning_rate': 4.131759111665349e-05, 'epoch': 1.76}
{'loss': 27.6891, 'grad_norm': 2.633587121963501, 'learning_rate': 3.917801930309486e-05, 'epoch': 1.8}
{'loss': 29.6804, 'grad_norm': 2.7111868858337402, 'learning_rate': 3.705904774487396e-05, 'epoch': 1.84}
{'loss': 27.4914, 'grad_norm': 2.7424368858337402, 'learning_rate': 3.4964710024786354e-05, 'epoch': 1.88}
{'loss': 27.2723, 'grad_norm': 2.8502166271209717, 'learning_rate': 3.289899283371657e-05, 'epoch': 1.92}
{'loss': 28.3726, 'grad_norm': 2.802563190460205, 'learning_rate': 3.086582838174551e-05, 'epoch': 1.96}
{'loss': 28.7767, 'grad_norm': 2.8514747619628906, 'learning_rate': 2.886908691296504e-05, 'epoch': 2.0}
{'loss': 28.1488, 'grad_norm': 2.7524795532226562, 'learning_rate': 2.6912569338248315e-05, 'epoch': 2.04}
{'loss': 27.394, 'grad_norm': 2.626378297805786, 'learning_rate': 2.500000000000001e-05, 'epoch': 2.08}
{'loss': 28.1236, 'grad_norm': 2.666848659515381, 'learning_rate': 2.3135019582658802e-05, 'epoch': 2.12}
{'loss': 26.8054, 'grad_norm': 2.586392879486084, 'learning_rate': 2.132117818244771e-05, 'epoch': 2.16}
{'loss': 27.5549, 'grad_norm': 2.886051893234253, 'learning_rate': 1.9561928549563968e-05, 'epoch': 2.2}
{'loss': 28.5234, 'grad_norm': 3.2526156902313232, 'learning_rate': 1.7860619515673033e-05, 'epoch': 2.24}
{'loss': 27.1454, 'grad_norm': 2.7967071533203125, 'learning_rate': 1.622048961921699e-05, 'epoch': 2.28}
{'loss': 26.9648, 'grad_norm': 3.001215696334839, 'learning_rate': 1.4644660940672627e-05, 'epoch': 2.32}
{'loss': 28.4659, 'grad_norm': 2.7390246391296387, 'learning_rate': 1.3136133159493802e-05, 'epoch': 2.36}
{'loss': 27.4011, 'grad_norm': 2.6751747131347656, 'learning_rate': 1.1697777844051105e-05, 'epoch': 2.4}
{'loss': 28.6096, 'grad_norm': 2.771030902862549, 'learning_rate': 1.0332332985438248e-05, 'epoch': 2.44}
{'loss': 28.9081, 'grad_norm': 2.7364416122436523, 'learning_rate': 9.042397785550405e-06, 'epoch': 2.48}
{'loss': 28.7498, 'grad_norm': 2.8686349391937256, 'learning_rate': 7.830427709355725e-06, 'epoch': 2.52}
{'loss': 27.2436, 'grad_norm': 2.727646827697754, 'learning_rate': 6.698729810778065e-06, 'epoch': 2.56}
{'loss': 26.6863, 'grad_norm': 2.8000833988189697, 'learning_rate': 5.649458341088915e-06, 'epoch': 2.6}
{'loss': 27.4548, 'grad_norm': 2.9961183071136475, 'learning_rate': 4.684610648167503e-06, 'epoch': 2.64}
{'loss': 27.1663, 'grad_norm': 3.0859694480895996, 'learning_rate': 3.8060233744356633e-06, 'epoch': 2.68}
{'loss': 27.3843, 'grad_norm': 2.9410879611968994, 'learning_rate': 3.0153689607045845e-06, 'epoch': 2.72}
{'loss': 28.4133, 'grad_norm': 2.750465154647827, 'learning_rate': 2.314152462588659e-06, 'epoch': 2.76}
{'loss': 27.5139, 'grad_norm': 2.7588133811950684, 'learning_rate': 1.70370868554659e-06, 'epoch': 2.8}
{'loss': 27.7183, 'grad_norm': 2.777634382247925, 'learning_rate': 1.1851996440033319e-06, 'epoch': 2.84}
{'loss': 28.741, 'grad_norm': 2.8054134845733643, 'learning_rate': 7.596123493895991e-07, 'epoch': 2.88}
{'loss': 27.3396, 'grad_norm': 2.60774564743042, 'learning_rate': 4.277569313094809e-07, 'epoch': 2.92}
{'loss': 28.5711, 'grad_norm': 2.6314098834991455, 'learning_rate': 1.9026509541272275e-07, 'epoch': 2.96}
{'loss': 27.6173, 'grad_norm': 2.9611477851867676, 'learning_rate': 4.7588920907110094e-08, 'epoch': 3.0}
{'train_runtime': 1523.0039, 'train_samples_per_second': 0.788, 'train_steps_per_second': 0.049, 'train_loss': 30.227289123535158, 'epoch': 3.0}
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 75/75 [25:23<00:00, 20.31s/it]

✅ Done. LoRA adapter saved to: qwen_lora_adapter_0119_lw
```