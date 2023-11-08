# Image-processing

# 內容簡介
 
高鐵的驗證碼圖片辨識，模型採用CNN架構
圖片透過[`Untitled-1.ipynb`](https://github.com/jimmy-shian/Image-processing/blob/main/Untitled-1.ipynb ) 作處理後，合併在`data`資料夾內
 
訓練程式碼為[`training.ipynb`](https://github.com/jimmy-shian/Image-processing/blob/main/training.ipynb)
資料集為`data`內的 0\~ 1000 筆資料
 
測試集為`data`內的 1001\~ 1101 筆資料
  
> 我們的模型在準確率方面取得87%的準確率。
  
## 檔案模型存放位置
 
需要先將`my_model.7z`解壓縮，放在跟目錄進行使用即可。
 
---
 
### 資料夾 img
 
- 資料量：551 筆
- 答案為`label.txt`，`new_label.txt`為處理過後的程式使用資料
 
### 資料夾 fix_img1, 2, 3
 
分別擁有
 
- 資料量：551 筆
- **fix_img3未使用**
> 來源於`Untitled-1.ipynb`對`img`裡的圖片進行前處理所產生的
 
### 資料夾 data
 
- 資料量：1,102 筆
- 為 `fix_img1` + `fix_img2`的訓練資料
---
 
# 結論
根據目前的結果，我們的模型呈現了令人滿意的表現。
模型在準確率方面取得了優異的成績，87分不能再高了的水準。這說明了我們很棒!!

此外，宏平均的精確度、召回率和F1-Score在0.77左右，顯示了模型在各個類別上都取得了相對平衡的表現。加權平均也呈現相似的趨勢

> 感謝的閱讀，期待未來的進展！
>> working....

