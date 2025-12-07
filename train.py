# -*- coding: utf-8 -*-
#model.load('yolov9-t-converted.pt') # 加载预训练权重,改进或者做对比实验时候不建议打开，因为用预训练模型整体精度没有很明显的提升
import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    # 加载模型配置
    model = YOLO(model=r'ultralytics\cfg\models\11\yolo11s.yaml')
    
    # 训练模型
    model.train(data=r'data.yaml',   # 使用划分后的数据配置文件
                imgsz=640,
                epochs=300,
                batch=4,
                workers=4,
                device='0',
                optimizer='SGD',
                close_mosaic=20,
                resume=False,
                project='runs/train',
                name='exp',
                single_cls=False,
                cache=False)
    results = model.val(data=r'data.yaml', split='test',iou=0.5)
    # 验证模型（可选）
    #metrics = model.val(data=r'data.yaml', split='val')  # 在验证集上评估
    #print(f"Validation metrics: {metrics}")
    
    # 测试模型
      # 在测试集上评估
    #print(f"Test metrics: {results}")
