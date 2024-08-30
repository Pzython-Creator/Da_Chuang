# 测试：

## 	启动环境：

cd进STIL(以下都在STIL文件夹下进行)

```shell
#activate venv
source venv/bin/activate
```

## 	推理预测：

用来测试的视频数据在STIL/samples/manipulated_sequences/videos/c23/videos,我写了三个方法的运行脚本，在STIL文件夹中：

配置文件是config里的ffpp.yaml

predict_DF.sh是检测Deepfakes方法，predict_F2F.sh是检测Face2Face方法，predict_FS.sh是检测FaceSwap方法。

```shell
#run Deepfakes
bash predict_DF.sh
```

```shell
#run Face2Face
bash predict_F2F.sh
```

```shell
#run FS
bash predict_FS.sh
```

输出将为"true video" 或 "fake video"

测试完要真正开始跑的话要把测试的那个视频删掉，samples下面的那个.pkl文件 测试完可以删可以不删。

事实上每检测完一个要检测下一个视频时，都要把上一个检测的视频删掉，因为我不会写一直保存还能继续往下检测的方法

# 需求：

用户上传视频，能得到该视频是真或假的判断。

一次只能测一个，

首先接受视频需要存入STIL/samples/manipulated_sequences/videos/c23/videos,路径比较绕，要改也行，但是我不想改了。

然后数据预处理

```命令行
#create a pkl file
python FaceDetector.py
```

这样会在samples下创造一个.pkl文件，保存的是人脸的边框信息。

然后测试的三个方法的脚本：predict_DF.sh，predict_F2F.sh，predict_FS.sh，都跑一遍，输出的true 或者 false哪个更多那么结果就是哪个