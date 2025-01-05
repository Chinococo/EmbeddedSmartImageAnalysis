
### 期中考專案總結

---

#### 模型設定

- 模型架構：ResNet-18
- 原始影像大小：
  - **224x224** (原始版本)
  - **224x130** (改進版本)
- 訓練目標：車道辨識，預測路徑落點
- 平均辨識速率：60ms~80ms (提提升版本)、50ms~110ms (提升版本) / 150ms~200ms (原始版本)
- 相機FPS：
  - 5fps（224x224 原始版本）
  - 9~13fps（224x130 改進版本）
- 參數
- ![image.png](image.png)
---

#### 訓練資料集與結果

| 版本名稱                                                                                                                     | 描述                               | 資料集                      |
|--------------------------------------------------------------------------------------------------------------------------|----------------------------------|-----------------------------|
| [best_steering_model_xy.pth](Midterm%2FResult%2Fbest_steering_model_xy.pth)                                              | 第一個成功版本，使用初始的 1100 資料集進行訓練       | [1100](Midterm/Train/1100)          |
| [resnet18_1600.pth](Midterm%2FResult%2Fresnet18_1600.pth)                                                                | 新增樣本數以涵蓋更多角度                     | [1600](Midterm/Train/1600)          |
| [resnet18_1600_v2.pth](Midterm%2FResult%2Fresnet18_1600_v2.pth)                                                          | 重新標記資料以避免過於極端的標記，減少誤判            |    [1600-v2](Midterm%2FTrain%2F1600-v2)     |
| [resnet18_1600_v3.pth](Midterm%2FResult%2Fresnet18_1600_v3.pth)                                                          | 增加 100 張靠左直線空白區域樣本，增強對直線的辨識      |    [1600-v3](Midterm%2FTrain%2F1600-v3)     |
| [resnet18_1600_v4.pth](Midterm%2FResult%2Fresnet18_1600_v4.pth)                                                          | 增加 300 張靠右直線空白區域樣本，加強模型在右側車道的穩定性 | [1600-v4](Midterm%2FTrain%2F1600-v4)        |
| [resnet18_finetuned_fp16.pth](Midterm%2FResult%2Fresnet18_finetuned_fp16.pth) (看起來fintuning太少)                           | 半精度加速(量化)                        |  [1600-v4](Midterm%2FTrain%2F1600-v4)      |
| [resnet18_finetuned_fp16_pruned.pth](Midterm%2FResult%2Fresnet18_finetuned_fp16_pruned.pth) (預測軟體看起來不錯)(以測試可以加速10% 效果不差) | 半精度加速(量化)+10%剪枝                  |  [1600-v4](Midterm%2FTrain%2F1600-v4)      |

---

#### 運行環境

- **平台**：JetBot (依賴相機)
- **訓練文件**： [TrainRestnet_TensorRT.ipynb](Midterm/Train/TrainRestnet_TensorRT.ipynb)
- **運行程式**：
  - **道路跟隨程式 (原始影像)**： [NormalRoadFollowing.ipynb](Midterm/Result/ipynb/NormalRoadFollowing.ipynb)
  - **道路跟隨程式 (截去影像上部)**： [NormalRoadFollowing-CutImage.ipynb](Midterm/Result/ipynb/NormalRoadFollowing-CutImage.ipynb)
  - **預測落點並顯示**： 
    - 原始影像：[Predict (1).ipynb](Result/ipynb/Predict%20(1).ipynb)
    - 截去影像上部：[PredictCutImage.ipynb](Result/ipynb/PredictCutImage.ipynb)
  - **測試相機處理速度**：
    - 原始影像：[TestExcuteSpeed.ipynb](Result/ipynb/TestExcuteSpeed.ipynb)
    - 截去影像上部：[TestExcuteSpeed-CutImage.ipynb](Result/ipynb/TestExcuteSpeed-CutImage.ipynb)
---

#### 改進點與優化

- **平視拍攝**：
  - 拍攝角度較接近人眼平視，避免僅依賴路面特徵，有助於提前預測動向。
  - 平視需更大量樣本以忽略上半部分背景，否則易受雜訊影響。
- **影像裁剪**：
  - 去除影像上半部分 40% 的資料，避免背景干擾，僅保留車道特徵，有助於提高模型穩定性。
  - 對 ResNet-18 進行輸入適配 (224x130) 訓練，顯著減少辨識延遲。
- 半精度(量化)10%~20%剪枝
  - 將訓練好的模型減少精度FP32->FP16 加速計算
  - 並重新fintuning [減小模型.ipynb](Midterm%2FTrain%2F%E6%B8%9B%E5%B0%8F%E6%A8%A1%E5%9E%8B.ipynb)
  - 提升到60ms~80ms
- TensorRT
  - 執行腳本(本地我搞不出來)[Colab腳本]("https://colab.research.google.com/drive/1i6E-K5drZ0D1g93dLfZ-GSJjIvasuc9K#scrollTo=usF1VG4o2J6-")
---

#### 影片演示
##### resnet18_1600.pth 結果
- [示例影片 1](https://youtube.com/shorts/N_F7avbfFk4?feature=share)
- [示例影片 2](https://youtube.com/shorts/S1-lvRKiw_E?feature=share)
- [示例影片 3](https://youtube.com/shorts/tDnq7fe2e6c?feature=share)
##### resnet18_1600.pth(剪枝/半精度) 結果
- [示例影片 1](https://youtube.com/shorts/TTU1k0pBeaM?feature=share)

---

#### 最終訓練模型

- 原始版本：[lane_detection_model.pth](Result/lane_detection_model.pth)
- 改進版本：[best_steering_model_xy.pth](Result/best_steering_model_xy.pth)
- 其他版本：
  - [resnet18_1600.pth](Result/resnet18_1600.pth)
  - [resnet18_1600_v2.pth](Result/resnet18_1600_v2.pth)
  - [resnet18_1600_v3.pth](Result/resnet18_1600_v3.pth)
  - [resnet18_1600_v4.pth](Result/resnet18_1600_v4.pth)
  - [resnet18_finetuned_fp16.pth](Midterm%2FResult%2Fresnet18_finetuned_fp16.pth)
  - [resnet18_finetuned_fp16_pruned.pth](Midterm%2FResult%2Fresnet18_finetuned_fp16_pruned.pth)
  - (x)[resnet18_finetuned_fp16_pruned_TensorRT.pth](Midterm%2FResult%2Fresnet18_finetuned_fp16_pruned_TensorRT.pth)
  - [resnet18_finetuned_fp16_pruned_20.pth](Midterm%2FResult%2Fresnet18_finetuned_fp16_pruned_20.pth)
---


## YoloV4_tiny Car object detect
> train.txt is form [dataset](https://www.kaggle.com/code/balraj98/yolo-v5-car-object-detection/input) 
> 
> train2.txt is base on train.txt add
> [dataset](https://www.kaggle.com/datasets/saumyapatel/traffic-vehicles-object-detection/suggestions?status=pending&yourSuggestions=true) fliter by car anntoion
### Main Code(Running on colab)
[YoloV4.ipynb](Hw3_YoloV4_tiny%2FYoloV4.ipynb)
[colab](https://colab.research.google.com/drive/1wLqsLnIht2j7tjL7eopB47cOAi5-Je1x?usp=sharing)
#### Version1
1. [coco.data](Hw3_YoloV4_tiny%2Fcoco.data)
2. [obj.name](Hw3_YoloV4_tiny%2Fobj.name)
3. [train.txt](Hw3_YoloV4_tiny%2Ftrain.txt)
4. [yolov4-tiny.weights](Hw3_YoloV4_tiny%2Fyolov4-tiny.weights)
5. [yolov4-tiny-obj.cfg](Hw3_YoloV4_tiny%2Fyolov4-tiny-obj.cfg)
##### Result
- [yolov4-tiny-obj_final (2).weights](Hw3_YoloV4_tiny%2Fv1%2Fyolov4-tiny-obj_final%20%282%29.weights)
- [predict Result Viedo](https://www.youtube.com/watch?v=JSLVMdWxJfk)
---
#### Version2
1. [coco.data](Hw3_YoloV4_tiny%2Fcoco.data)
2. [obj.name](Hw3_YoloV4_tiny%2Fobj.name)
3. [train2.txt](Hw3_YoloV4_tiny%2Ftrain2.txt)
4. [yolov4-tiny.weights](Hw3_YoloV4_tiny%2Fyolov4-tiny.weights)
5. [yolov4-tiny-obj.cfg](Hw3_YoloV4_tiny%2Fyolov4-tiny-obj.cfg)
##### Result
- [yolov4-tiny-obj_last.weights](Hw3_YoloV4_tiny%2Fv2%2Fyolov4-tiny-obj_last.weights)
- [predict Result Viedo](https://www.youtube.com/watch?v=HIHQPvlyBII)
- [Final Good Result](https://youtu.be/IBy0CXA6RpY)
### Project5
- [訓練執行檔](https://colab.research.google.com/drive/1wLqsLnIht2j7tjL7eopB47cOAi5-Je1x#scrollTo=ZRh5epEz6YhB)
  - [訓練資料夾](https://drive.google.com/drive/folders/1JtiYKp36zMfkaHJohwD2RDVhOyoeQQq7?usp=drive_link)
  - [訓練配置檔](Project5%2Fyolov4-tiny-224.cfg)
  - [訓練結果](Project5%2Fyolov4-tiny-224.weights)
- [Demo code](Project5%2Fproject5.ipynb)
  - [trt權重](Project5%2Fyolov4-tiny-224.trt)
### Lab6

#### 功能概述
1. **文字分析**  
   - 使用 OpenAI GPT API 將文字轉換為動作清單。  
   - 路由：`/analyze` (POST)

2. **圖片分析**  
   - 將圖片上傳到 Imgur，並用 OpenAI API 分析圖片內容。  
   - 路由：`/analyze-image` (POST)

3. **主頁與圖片頁面**  
   - 主頁：`/`  
   - 圖片頁面：`/image`

---

#### 必要配置
請建立 `.env` 文件，在Lab6底下，內容如下：
```env
IMGUR_CLIENT_ID=???
OPENAI_API_KEY=??
```
#### 結果
##### 文字分析
![img.png](Lab6%2Fimg%2Fimg.png)
##### 圖片分析
> 輸入:  
> ![00c4731c-967a-47bf-9f9c-9940b8443921.jpg](Lab6%2Fimg%2F00c4731c-967a-47bf-9f9c-9940b8443921.jpg)

>結果:
>![img_1.png](Lab6%2Fimg%2Fimg_1.png)
