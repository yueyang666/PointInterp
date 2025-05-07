```
讀取時間：3.9848 s
16線點雲數量: 148
32線點雲數量: 148
64線點雲數量: 148
開始計算所有幀的 Chamfer Distance...
已計算 50 / 148 幀...
已計算 100 / 148 幀...
全部計算完成，耗時 24.35 秒。
全部幀的平均 Chamfer Distance: 0.094510
```


```
全部計算完成，總耗時 44.11 秒
全部幀的平均 Chamfer Distance: 0.115723
```
```
讀取時間：3.9937 s
資料筆數：148 筆 32線點雲
開始插值並計算 Chamfer Distance...
Processing Frames: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 148/148 [00:33<00:00,  4.46it/s]
全部計算完成，總耗時 33.18 秒
全部幀的平均 Chamfer Distance: 0.094642
```

讀取時間：0.0041 s
16線點雲數量: 1
第一筆16線 shape: (28181, 4)
32線點雲數量: 1
第一筆32線 shape: (56420, 4)
64線Ground Truth數量: 1
第一筆64線 shape: (116770, 4)
隨機選擇第 0 幀
第 0 幀 Chamfer Distance: 0.126592

=== 16線插值32線 Chamfer Distance 分析報告 ===
總Frame數量: 148
平均 Chamfer Distance: 0.419356
標準差: 0.302266
最大 Chamfer Distance: 1.653970 (Frame 25)
最小 Chamfer Distance: 0.045367 (Frame 36)

16線插值32線 Chamfer Distance 曲線圖已儲存到: figures/loss_curve_20250506_040309_16to32.png

=== 32線插值64線 Chamfer Distance 分析報告 ===
總Frame數量: 148
平均 Chamfer Distance: 0.094641
標準差: 0.062948
最大 Chamfer Distance: 0.413230 (Frame 8)
最小 Chamfer Distance: 0.012574 (Frame 36)


32線插值64線 Chamfer Distance 曲線圖已儲存到: figures/loss_curve_20250506_040326_32to64.png