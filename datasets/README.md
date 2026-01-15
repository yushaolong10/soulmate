#### 1.small dataset (1490样本, 3epoch)
```
 python finetune.py
Downloading Model from https://www.modelscope.cn to directory: /Users/yushaolong/.cache/modelscope/hub/models/Qwen/Qwen3-1.7B
2026-01-15 11:30:45,273 - modelscope - INFO - Target directory already exists, skipping creation.
Downloading Model from https://www.modelscope.cn to directory: /Users/yushaolong/.cache/modelscope/hub/models/Qwen/Qwen3-1.7B
2026-01-15 11:30:46,086 - modelscope - INFO - Target directory already exists, skipping creation.
`torch_dtype` is deprecated! Use `dtype` instead!
Loading checkpoint shards: 100%|███████████████████████████████████████████████████████████████████████████████████| 2/2 [00:07<00:00,  3.95s/it]
Generating train split: 1490 examples [00:00, 32441.41 examples/s]
Map: 100%|██████████████████████████████████████████████████████████████████████████████████████████| 1490/1490 [00:00<00:00, 2275.60 examples/s]
Truncating train dataset: 100%|███████████████████████████████████████████████████████████████████| 1490/1490 [00:00<00:00, 512549.25 examples/s]
The tokenizer has new PAD/BOS/EOS tokens that differ from the model config and generation config. The model config and generation config were aligned accordingly, being updated with the tokenizer's values. Updated tokens: {'bos_token_id': None, 'pad_token_id': 151643}.
  0%|                                                                                                                    | 0/282 [00:00<?, ?it/s]/Users/yushaolong/Desktop/work/ai_program/soulmate/venv/lib/python3.11/site-packages/torch/utils/data/dataloader.py:692: UserWarning: 'pin_memory' argument is set as true but not supported on MPS now, device pinned memory won't be used.
  warnings.warn(warn_msg)
{'loss': 5.8893, 'grad_norm': 5.191494464874268, 'learning_rate': 0.0001, 'entropy': 1.530438232421875, 'num_tokens': 16858.0, 'mean_token_accuracy': 0.32799972118809817, 'epoch': 0.11}
{'loss': 3.5701, 'grad_norm': 1.6658929586410522, 'learning_rate': 9.966929928028133e-05, 'entropy': 2.28489990234375, 'num_tokens': 32999.0, 'mean_token_accuracy': 0.3929405600763857, 'epoch': 0.21}
{'loss': 2.9203, 'grad_norm': 1.0390846729278564, 'learning_rate': 9.868157163976622e-05, 'entropy': 2.94166259765625, 'num_tokens': 49870.0, 'mean_token_accuracy': 0.45076825506985185, 'epoch': 0.32}
{'loss': 2.7682, 'grad_norm': 1.0074023008346558, 'learning_rate': 9.704988276811883e-05, 'entropy': 2.7728515625, 'num_tokens': 67130.0, 'mean_token_accuracy': 0.4631983643397689, 'epoch': 0.43}
{'loss': 2.6508, 'grad_norm': 1.1403886079788208, 'learning_rate': 9.479581669270759e-05, 'entropy': 2.76092529296875, 'num_tokens': 83866.0, 'mean_token_accuracy': 0.4678344836458564, 'epoch': 0.54}
{'loss': 2.6354, 'grad_norm': 1.1507043838500977, 'learning_rate': 9.19491902644698e-05, 'entropy': 2.74027099609375, 'num_tokens': 100305.0, 'mean_token_accuracy': 0.4810102773830295, 'epoch': 0.64}
{'loss': 2.6063, 'grad_norm': 1.4233546257019043, 'learning_rate': 8.854765873974898e-05, 'entropy': 2.7252197265625, 'num_tokens': 116823.0, 'mean_token_accuracy': 0.4817782023921609, 'epoch': 0.75}
{'loss': 2.5352, 'grad_norm': 1.2214453220367432, 'learning_rate': 8.463621767547998e-05, 'entropy': 2.71317138671875, 'num_tokens': 133168.0, 'mean_token_accuracy': 0.49220794159919024, 'epoch': 0.86}
{'loss': 2.5709, 'grad_norm': 1.470194935798645, 'learning_rate': 8.026660772666642e-05, 'entropy': 2.64248046875, 'num_tokens': 149806.0, 'mean_token_accuracy': 0.4929128268733621, 'epoch': 0.97}
{'loss': 2.4525, 'grad_norm': 1.4186941385269165, 'learning_rate': 7.54966302195068e-05, 'entropy': 2.703780500856164, 'num_tokens': 164969.0, 'mean_token_accuracy': 0.5044764968222135, 'epoch': 1.06}
{'loss': 2.4927, 'grad_norm': 1.6271265745162964, 'learning_rate': 7.038938255378751e-05, 'entropy': 2.658740234375, 'num_tokens': 181774.0, 'mean_token_accuracy': 0.5012675948441029, 'epoch': 1.17}
{'loss': 2.4334, 'grad_norm': 1.4419245719909668, 'learning_rate': 6.501242354866194e-05, 'entropy': 2.636181640625, 'num_tokens': 198623.0, 'mean_token_accuracy': 0.505122335255146, 'epoch': 1.28}
{'loss': 2.4308, 'grad_norm': 1.6731239557266235, 'learning_rate': 5.943687977264584e-05, 'entropy': 2.6178466796875, 'num_tokens': 214917.0, 'mean_token_accuracy': 0.5102606443688273, 'epoch': 1.39}
{'loss': 2.441, 'grad_norm': 1.551530122756958, 'learning_rate': 5.373650467932122e-05, 'entropy': 2.62900390625, 'num_tokens': 231563.0, 'mean_token_accuracy': 0.5062695536762476, 'epoch': 1.49}
{'loss': 2.444, 'grad_norm': 1.576484203338623, 'learning_rate': 4.798670299452926e-05, 'entropy': 2.62635498046875, 'num_tokens': 248713.0, 'mean_token_accuracy': 0.5056773917749524, 'epoch': 1.6}
{'loss': 2.417, 'grad_norm': 1.6646183729171753, 'learning_rate': 4.226353326048593e-05, 'entropy': 2.63134765625, 'num_tokens': 264949.0, 'mean_token_accuracy': 0.503875277750194, 'epoch': 1.71}
{'loss': 2.4132, 'grad_norm': 1.5971559286117554, 'learning_rate': 3.664270173119611e-05, 'entropy': 2.6142333984375, 'num_tokens': 281335.0, 'mean_token_accuracy': 0.5081624800339342, 'epoch': 1.82}
{'loss': 2.4361, 'grad_norm': 1.55985426902771, 'learning_rate': 3.1198560927945905e-05, 'entropy': 2.63167724609375, 'num_tokens': 298008.0, 'mean_token_accuracy': 0.49596401806920765, 'epoch': 1.92}
{'loss': 2.4034, 'grad_norm': 1.417194128036499, 'learning_rate': 2.6003126102010695e-05, 'entropy': 2.620665667808219, 'num_tokens': 313556.0, 'mean_token_accuracy': 0.5064643852106513, 'epoch': 2.02}
{'loss': 2.3242, 'grad_norm': 1.6272653341293335, 'learning_rate': 2.112512261483801e-05, 'entropy': 2.58914794921875, 'num_tokens': 329600.0, 'mean_token_accuracy': 0.5224992288276553, 'epoch': 2.13}
{'loss': 2.3281, 'grad_norm': 1.4724522829055786, 'learning_rate': 1.6629076836987784e-05, 'entropy': 2.586572265625, 'num_tokens': 346216.0, 'mean_token_accuracy': 0.5224415114149451, 'epoch': 2.24}
{'loss': 2.3151, 'grad_norm': 1.8642715215682983, 'learning_rate': 1.257446259144494e-05, 'entropy': 2.58572998046875, 'num_tokens': 362912.0, 'mean_token_accuracy': 0.5253308081999422, 'epoch': 2.34}
{'loss': 2.3441, 'grad_norm': 1.7463430166244507, 'learning_rate': 9.014914432176792e-06, 'entropy': 2.59095458984375, 'num_tokens': 379264.0, 'mean_token_accuracy': 0.5193719357252121, 'epoch': 2.45}
{'loss': 2.3894, 'grad_norm': 1.4679211378097534, 'learning_rate': 5.997518164709076e-06, 'entropy': 2.592724609375, 'num_tokens': 396214.0, 'mean_token_accuracy': 0.5122475702315569, 'epoch': 2.56}
{'loss': 2.3844, 'grad_norm': 1.5289775133132935, 'learning_rate': 3.5621879937348836e-06, 'entropy': 2.60286865234375, 'num_tokens': 413038.0, 'mean_token_accuracy': 0.505672406591475, 'epoch': 2.67}
{'loss': 2.3189, 'grad_norm': 1.6851886510849, 'learning_rate': 1.7411385368659937e-06, 'entropy': 2.57261962890625, 'num_tokens': 429719.0, 'mean_token_accuracy': 0.5222833707928658, 'epoch': 2.77}
{'loss': 2.397, 'grad_norm': 1.7166872024536133, 'learning_rate': 5.584586887435739e-07, 'entropy': 2.6115478515625, 'num_tokens': 446344.0, 'mean_token_accuracy': 0.5031122887507081, 'epoch': 2.88}
{'loss': 2.3288, 'grad_norm': 1.5358495712280273, 'learning_rate': 2.9792972446479605e-08, 'entropy': 2.585546875, 'num_tokens': 463100.0, 'mean_token_accuracy': 0.5203175120055675, 'epoch': 2.99}
{'train_runtime': 2222.5196, 'train_samples_per_second': 2.011, 'train_steps_per_second': 0.127, 'train_loss': 2.627381867550789, 'entropy': 2.6243489583333335, 'num_tokens': 464922.0, 'mean_token_accuracy': 0.5267686810758379, 'epoch': 3.0}
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████| 282/282 [37:02<00:00,  7.88s/it]

✅ Done. LoRA adapter saved to: qwen_lora_adapter_0115_s
```


#### 2.medium dataset (4598样本, 2epoch)

```
python finetune.py
Downloading Model from https://www.modelscope.cn to directory: /Users/yushaolong/.cache/modelscope/hub/models/Qwen/Qwen3-1.7B
2026-01-15 16:30:09,030 - modelscope - INFO - Target directory already exists, skipping creation.
Downloading Model from https://www.modelscope.cn to directory: /Users/yushaolong/.cache/modelscope/hub/models/Qwen/Qwen3-1.7B
2026-01-15 16:30:09,793 - modelscope - INFO - Target directory already exists, skipping creation.
`torch_dtype` is deprecated! Use `dtype` instead!
Loading checkpoint shards: 100%|███████████████████████████████████████████████████████████████████████████████████| 2/2 [00:09<00:00,  4.50s/it]
The tokenizer has new PAD/BOS/EOS tokens that differ from the model config and generation config. The model config and generation config were aligned accordingly, being updated with the tokenizer's values. Updated tokens: {'bos_token_id': None, 'pad_token_id': 151643}.
  0%|                                                                                                                    | 0/576 [00:00<?, ?it/s]/Users/yushaolong/Desktop/work/ai_program/soulmate/venv/lib/python3.11/site-packages/torch/utils/data/dataloader.py:692: UserWarning: 'pin_memory' argument is set as true but not supported on MPS now, device pinned memory won't be used.
  warnings.warn(warn_msg)
{'loss': 7.1574, 'grad_norm': 5.900953769683838, 'learning_rate': 5e-05, 'entropy': 1.486376953125, 'num_tokens': 15309.0, 'mean_token_accuracy': 0.29728787755593655, 'epoch': 0.03}
{'loss': 4.7839, 'grad_norm': 2.2967121601104736, 'learning_rate': 9.999920755303033e-05, 'entropy': 1.89576416015625, 'num_tokens': 31007.0, 'mean_token_accuracy': 0.34408407686278225, 'epoch': 0.07}
{'loss': 3.085, 'grad_norm': 1.193611741065979, 'learning_rate': 9.990414430676715e-05, 'entropy': 2.983544921875, 'num_tokens': 46611.0, 'mean_token_accuracy': 0.4330693380907178, 'epoch': 0.1}
{'loss': 2.9682, 'grad_norm': 1.118050217628479, 'learning_rate': 9.965093687129668e-05, 'entropy': 2.94327392578125, 'num_tokens': 61981.0, 'mean_token_accuracy': 0.46019143536686896, 'epoch': 0.14}
{'loss': 2.8032, 'grad_norm': 1.2717361450195312, 'learning_rate': 9.924038765061042e-05, 'entropy': 2.9320068359375, 'num_tokens': 78236.0, 'mean_token_accuracy': 0.4695726968348026, 'epoch': 0.17}
{'loss': 2.7109, 'grad_norm': 1.2766650915145874, 'learning_rate': 9.867379765837236e-05, 'entropy': 2.88511962890625, 'num_tokens': 94037.0, 'mean_token_accuracy': 0.4777309441938996, 'epoch': 0.21}
{'loss': 2.7819, 'grad_norm': 1.3653537034988403, 'learning_rate': 9.795296239506012e-05, 'entropy': 2.93455810546875, 'num_tokens': 110179.0, 'mean_token_accuracy': 0.468700279481709, 'epoch': 0.24}
{'loss': 2.6988, 'grad_norm': 1.5178940296173096, 'learning_rate': 9.708016615809729e-05, 'entropy': 2.8153564453125, 'num_tokens': 125927.0, 'mean_token_accuracy': 0.48076201397925616, 'epoch': 0.28}
{'loss': 2.6798, 'grad_norm': 1.3791486024856567, 'learning_rate': 9.605817480300862e-05, 'entropy': 2.81947021484375, 'num_tokens': 141822.0, 'mean_token_accuracy': 0.4831177067011595, 'epoch': 0.31}
{'loss': 2.623, 'grad_norm': 1.3335206508636475, 'learning_rate': 9.489022697853709e-05, 'entropy': 2.79749755859375, 'num_tokens': 157226.0, 'mean_token_accuracy': 0.49421586357057096, 'epoch': 0.35}
{'loss': 2.624, 'grad_norm': 1.4239726066589355, 'learning_rate': 9.358002386349863e-05, 'entropy': 2.81148681640625, 'num_tokens': 173234.0, 'mean_token_accuracy': 0.4845303278416395, 'epoch': 0.38}
{'loss': 2.5886, 'grad_norm': 1.4432809352874756, 'learning_rate': 9.21317174378982e-05, 'entropy': 2.781256103515625, 'num_tokens': 189361.0, 'mean_token_accuracy': 0.4876081386581063, 'epoch': 0.42}
{'loss': 2.5318, 'grad_norm': 1.5029547214508057, 'learning_rate': 9.054989732547506e-05, 'entropy': 2.72347412109375, 'num_tokens': 204788.0, 'mean_token_accuracy': 0.5014538913965225, 'epoch': 0.45}
{'loss': 2.519, 'grad_norm': 1.4609863758087158, 'learning_rate': 8.883957624937333e-05, 'entropy': 2.69290771484375, 'num_tokens': 220342.0, 'mean_token_accuracy': 0.5095364410430193, 'epoch': 0.49}
{'loss': 2.6355, 'grad_norm': 1.515071988105774, 'learning_rate': 8.700617414702745e-05, 'entropy': 2.77569580078125, 'num_tokens': 236433.0, 'mean_token_accuracy': 0.4872701123356819, 'epoch': 0.52}
{'loss': 2.6075, 'grad_norm': 1.6140224933624268, 'learning_rate': 8.505550099460265e-05, 'entropy': 2.7998779296875, 'num_tokens': 252624.0, 'mean_token_accuracy': 0.4912933690473437, 'epoch': 0.56}
{'loss': 2.5849, 'grad_norm': 1.534812569618225, 'learning_rate': 8.299373839541829e-05, 'entropy': 2.7874755859375, 'num_tokens': 268467.0, 'mean_token_accuracy': 0.49450206737965346, 'epoch': 0.59}
{'loss': 2.5589, 'grad_norm': 2.129318952560425, 'learning_rate': 8.082741999070029e-05, 'entropy': 2.71370849609375, 'num_tokens': 283856.0, 'mean_token_accuracy': 0.5040526276454329, 'epoch': 0.63}
{'loss': 2.4572, 'grad_norm': 1.61896812915802, 'learning_rate': 7.856341075473962e-05, 'entropy': 2.726806640625, 'num_tokens': 299573.0, 'mean_token_accuracy': 0.5071036711335182, 'epoch': 0.66}
{'loss': 2.5319, 'grad_norm': 1.6795828342437744, 'learning_rate': 7.620888524007e-05, 'entropy': 2.67711181640625, 'num_tokens': 315219.0, 'mean_token_accuracy': 0.5015517903491855, 'epoch': 0.7}
{'loss': 2.5182, 'grad_norm': 1.661049485206604, 'learning_rate': 7.377130484160475e-05, 'entropy': 2.745880126953125, 'num_tokens': 331136.0, 'mean_token_accuracy': 0.49548952113837, 'epoch': 0.73}
{'loss': 2.4936, 'grad_norm': 1.6205928325653076, 'learning_rate': 7.125839415178204e-05, 'entropy': 2.71927490234375, 'num_tokens': 346943.0, 'mean_token_accuracy': 0.5055271102115512, 'epoch': 0.77}
{'loss': 2.5572, 'grad_norm': 1.5142056941986084, 'learning_rate': 6.867811648164769e-05, 'entropy': 2.72181396484375, 'num_tokens': 363023.0, 'mean_token_accuracy': 0.49477991200983523, 'epoch': 0.8}
{'loss': 2.4821, 'grad_norm': 1.5574238300323486, 'learning_rate': 6.603864862544878e-05, 'entropy': 2.74544677734375, 'num_tokens': 378540.0, 'mean_token_accuracy': 0.5065610017627478, 'epoch': 0.84}
{'loss': 2.4916, 'grad_norm': 1.5102086067199707, 'learning_rate': 6.334835494870758e-05, 'entropy': 2.693096923828125, 'num_tokens': 394179.0, 'mean_token_accuracy': 0.5030592609196901, 'epoch': 0.87}
{'loss': 2.5093, 'grad_norm': 1.7121374607086182, 'learning_rate': 6.0615760881889804e-05, 'entropy': 2.6944580078125, 'num_tokens': 410047.0, 'mean_token_accuracy': 0.5000030586495996, 'epoch': 0.9}
{'loss': 2.5078, 'grad_norm': 1.937254548072815, 'learning_rate': 5.784952590366464e-05, 'entropy': 2.721533203125, 'num_tokens': 425697.0, 'mean_token_accuracy': 0.5000421661883593, 'epoch': 0.94}
{'loss': 2.4757, 'grad_norm': 1.7500319480895996, 'learning_rate': 5.505841609937161e-05, 'entropy': 2.70926513671875, 'num_tokens': 441253.0, 'mean_token_accuracy': 0.5164028745144605, 'epoch': 0.97}
 50%|█████████████████████████████████████████████████████                                                     | 288/576 [36:10<49:05, 10.23s/it]/Users/yushaolong/Desktop/work/ai_program/soulmate/venv/lib/python3.11/site-packages/torch/utils/data/dataloader.py:692: UserWarning: 'pin_memory' argument is set as true but not supported on MPS now, device pinned memory won't be used.
  warnings.warn(warn_msg)
{'loss': 2.4177, 'grad_norm': 2.0270743370056152, 'learning_rate': 5.225127638165514e-05, 'entropy': 2.7013151041666665, 'num_tokens': 455948.0, 'mean_token_accuracy': 0.5073400501410167, 'epoch': 1.01}
{'loss': 2.4479, 'grad_norm': 1.7636042833328247, 'learning_rate': 4.943700246129871e-05, 'entropy': 2.67532958984375, 'num_tokens': 471489.0, 'mean_token_accuracy': 0.5023141017183661, 'epoch': 1.04}
{'loss': 2.4136, 'grad_norm': 1.7948801517486572, 'learning_rate': 4.662451265708174e-05, 'entropy': 2.671240234375, 'num_tokens': 487418.0, 'mean_token_accuracy': 0.5117261715233326, 'epoch': 1.08}
{'loss': 2.4072, 'grad_norm': 1.752000093460083, 'learning_rate': 4.3822719633992684e-05, 'entropy': 2.6419189453125, 'num_tokens': 503179.0, 'mean_token_accuracy': 0.5122636588290334, 'epoch': 1.11}
{'loss': 2.3447, 'grad_norm': 2.0811901092529297, 'learning_rate': 4.1040502159358746e-05, 'entropy': 2.64412841796875, 'num_tokens': 518620.0, 'mean_token_accuracy': 0.5285080384463072, 'epoch': 1.15}
{'loss': 2.4616, 'grad_norm': 1.7654168605804443, 'learning_rate': 3.82866769663959e-05, 'entropy': 2.70047607421875, 'num_tokens': 534484.0, 'mean_token_accuracy': 0.5117103312164545, 'epoch': 1.18}
{'loss': 2.4284, 'grad_norm': 1.8754584789276123, 'learning_rate': 3.556997081434248e-05, 'entropy': 2.6574951171875, 'num_tokens': 550161.0, 'mean_token_accuracy': 0.5121151655912399, 'epoch': 1.22}
{'loss': 2.4007, 'grad_norm': 1.8755627870559692, 'learning_rate': 3.289899283371657e-05, 'entropy': 2.66146240234375, 'num_tokens': 565700.0, 'mean_token_accuracy': 0.515607376024127, 'epoch': 1.25}
{'loss': 2.3738, 'grad_norm': 1.6909230947494507, 'learning_rate': 3.0282207244334082e-05, 'entropy': 2.64766845703125, 'num_tokens': 581247.0, 'mean_token_accuracy': 0.5252019265666604, 'epoch': 1.29}
{'loss': 2.422, 'grad_norm': 1.7875972986221313, 'learning_rate': 2.772790653254278e-05, 'entropy': 2.66578369140625, 'num_tokens': 596854.0, 'mean_token_accuracy': 0.5161840505897999, 'epoch': 1.32}
{'loss': 2.3702, 'grad_norm': 1.927504539489746, 'learning_rate': 2.524418517267283e-05, 'entropy': 2.64569091796875, 'num_tokens': 612665.0, 'mean_token_accuracy': 0.5228612747043371, 'epoch': 1.35}
{'loss': 2.4736, 'grad_norm': 1.7228206396102905, 'learning_rate': 2.283891397597908e-05, 'entropy': 2.6486083984375, 'num_tokens': 628984.0, 'mean_token_accuracy': 0.5090604413300752, 'epoch': 1.39}
{'loss': 2.3811, 'grad_norm': 1.9532105922698975, 'learning_rate': 2.0519715148362585e-05, 'entropy': 2.67696533203125, 'num_tokens': 644769.0, 'mean_token_accuracy': 0.5174464022740721, 'epoch': 1.42}
{'loss': 2.4296, 'grad_norm': 2.0597994327545166, 'learning_rate': 1.8293938135912476e-05, 'entropy': 2.646826171875, 'num_tokens': 660874.0, 'mean_token_accuracy': 0.5126217156648636, 'epoch': 1.46}
{'loss': 2.3634, 'grad_norm': 1.9171936511993408, 'learning_rate': 1.6168636334812125e-05, 'entropy': 2.63214111328125, 'num_tokens': 676471.0, 'mean_token_accuracy': 0.5267171211540699, 'epoch': 1.49}
{'loss': 2.3595, 'grad_norm': 2.0653445720672607, 'learning_rate': 1.4150544739415756e-05, 'entropy': 2.62127685546875, 'num_tokens': 692039.0, 'mean_token_accuracy': 0.5181246595457196, 'epoch': 1.53}
{'loss': 2.3963, 'grad_norm': 1.7269763946533203, 'learning_rate': 1.224605859932702e-05, 'entropy': 2.63118896484375, 'num_tokens': 707925.0, 'mean_token_accuracy': 0.5220922818407416, 'epoch': 1.56}
{'loss': 2.413, 'grad_norm': 1.824412226676941, 'learning_rate': 1.046121315311508e-05, 'entropy': 2.6731201171875, 'num_tokens': 724110.0, 'mean_token_accuracy': 0.515910635702312, 'epoch': 1.6}
{'loss': 2.375, 'grad_norm': 1.7795859575271606, 'learning_rate': 8.801664502890722e-06, 'entropy': 2.65767822265625, 'num_tokens': 740088.0, 'mean_token_accuracy': 0.5217986611649394, 'epoch': 1.63}
{'loss': 2.3487, 'grad_norm': 1.9420173168182373, 'learning_rate': 7.27267169035053e-06, 'entropy': 2.641357421875, 'num_tokens': 755813.0, 'mean_token_accuracy': 0.5211648056283593, 'epoch': 1.67}
{'loss': 2.3955, 'grad_norm': 1.9283866882324219, 'learning_rate': 5.879080031089046e-06, 'entropy': 2.6445556640625, 'num_tokens': 771524.0, 'mean_token_accuracy': 0.5145618369802832, 'epoch': 1.7}
{'loss': 2.3861, 'grad_norm': 1.9613897800445557, 'learning_rate': 4.625305759992205e-06, 'entropy': 2.6470458984375, 'num_tokens': 787163.0, 'mean_token_accuracy': 0.5142383834347128, 'epoch': 1.74}
{'loss': 2.3864, 'grad_norm': 1.8209816217422485, 'learning_rate': 3.5153220363698226e-06, 'entropy': 2.66583251953125, 'num_tokens': 803096.0, 'mean_token_accuracy': 0.5093262894079089, 'epoch': 1.77}
{'loss': 2.3833, 'grad_norm': 1.8572112321853638, 'learning_rate': 2.5526463531765465e-06, 'entropy': 2.668231201171875, 'num_tokens': 818703.0, 'mean_token_accuracy': 0.5159433891996741, 'epoch': 1.81}
{'loss': 2.4183, 'grad_norm': 1.9099087715148926, 'learning_rate': 1.740329390220685e-06, 'entropy': 2.66123046875, 'num_tokens': 834196.0, 'mean_token_accuracy': 0.5157493766397238, 'epoch': 1.84}
{'loss': 2.3691, 'grad_norm': 1.791279673576355, 'learning_rate': 1.0809453466849029e-06, 'entropy': 2.632733154296875, 'num_tokens': 850274.0, 'mean_token_accuracy': 0.5267206624150276, 'epoch': 1.88}
{'loss': 2.4352, 'grad_norm': 2.105327844619751, 'learning_rate': 5.765837835944309e-07, 'entropy': 2.646875, 'num_tokens': 865690.0, 'mean_token_accuracy': 0.5150044744834303, 'epoch': 1.91}
{'loss': 2.4099, 'grad_norm': 1.8889821767807007, 'learning_rate': 2.2884300208378396e-07, 'entropy': 2.651416015625, 'num_tokens': 881388.0, 'mean_token_accuracy': 0.5162023862823844, 'epoch': 1.95}
{'loss': 2.3968, 'grad_norm': 1.8035075664520264, 'learning_rate': 3.88249784459227e-08, 'entropy': 2.641888427734375, 'num_tokens': 897274.0, 'mean_token_accuracy': 0.5188917553052306, 'epoch': 1.98}
{'train_runtime': 4827.9661, 'train_samples_per_second': 1.905, 'train_steps_per_second': 0.119, 'train_loss': 2.621161472466257, 'entropy': 2.6218432049418605, 'num_tokens': 905788.0, 'mean_token_accuracy': 0.5227171231147855, 'epoch': 2.0}
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████| 576/576 [1:20:27<00:00,  8.38s/it]

✅ Done. LoRA adapter saved to: qwen_lora_adapter_0115_x