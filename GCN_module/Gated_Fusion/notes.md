### Gated fusion module

##### 将关节点数据 pose 速度数据velocity 通过交叉注意力得到pose_att

##### 将速度数据velocity通过1d卷积得到v1，pool为权重与关节点数据mul 进而得到pose_mul

作为input1 与 input2 通过 1d卷积调整为两个输入 x与y

则	$z = Sigmoid(x + y)$

即最终	$output = ReLU(z * x + (1-z)*y + bias)$

## 00-模型结果为：80.7964

[File 01](https://github.com/sellenzh/doing_project/tree/main/GCN_module/Gated_Fusion)

![pic.png](/Users/sellenz/Desktop/project/gated_fusion/80.7964-ori/80.7964.png)

## 01-00去除gated fusion 模型中 最后的两层2d卷积层，结果为：83.94

[File 02](https://github.com/sellenzh/doing_project/blob/main/GCN_module/Gated_Fusion/file02.py)

```python
@Delete
self.layer = nn.ModuleList()
        self.conv = nn.Sequential(
                        nn.Conv2d(dims, dims, kernel_size=1, stride=1),
                        nn.BatchNorm2d(dims), nn.ReLU()
        )
        for _ in range(self.layers):
            self.layer.append(self.conv)

for i in range(self.layers):
            result = self.layer[i](result)
```

![pic.png](/Users/sellenz/Desktop/project/gated_fusion/83.94-without2conv/83.94.png)

## 02-01调整 Cross Attention模块中 注意力头的个数为4/8:结果为 85.87

[File 03](https://github.com/sellenzh/doing_project/blob/main/GCN_module/Gated_Fusion/file03.py)

![](/Users/sellenz/Desktop/project/gated_fusion/85.87-4h/85.87.png)

## 03-01调整 Cross Attention模块中 注意力头的个数为2:结果为84.52

![](/Users/sellenz/Desktop/project/gated_fusion/84.52-2h/84.52.png)

## 04-02batch_size = 32:		85.9666

![](/Users/sellenz/Desktop/project/gated_fusion/85.9666-32b/85.9666.png)

## 05-02batch_size = 64:		72.286448



## 06-02调整gated Fusion模块中 FC的2d卷积为linear，结果为：84.01

[File04](https://github.com/sellenzh/doing_project/blob/main/GCN_module/Gated_Fusion/file04.py)

```python
self.conv = nn.Conv2d(dims, dims, kernel_size=1, stride=1)
->
self.linear = nn.Linear(dims, dims)
```

![](/Users/sellenz/Desktop/project/gated_fusion/84.01-linear/84.01.png)

## 07-02调整gated Fusion模块中 FC的2d卷积为mlp，结果为:

[file05](https://github.com/sellenzh/doing_project/blob/main/GCN_module/Gated_Fusion/file05.py)

## version 1 -> 85.55:

$y = W_2(W_1x + bias_1) + bias_2$ 

$W_1\in(dims*hidden)$,$W_2\in(hidden*dims)$ ,$hidden=dims*2$

```python
self.hidden = dims * 2
self.layers = nn.Sequential(
            nn.Linear(dims, self.hidden), nn.ReLU(),
            nn.Linear(self.hidden, dims), nn.ReLU())
```

![](/Users/sellenz/Desktop/project/gated_fusion/85.55-mlp/85.55.png)

## Version2 -> 85.61:

$y = W_3(W_2(W_1x + bias_1) + bias_2) + bias_3$

$W_1\in(dims*hidden)$,$W_2\in(hidden*hidden)$ ,$W_3\in(hidden*dims)$ ,$hidden=dims*2$

```python
self.hidden = dims * 2
self.layers = nn.Sequential(
            nn.Linear(dims, self.hidden), nn.ReLU(),
  					nn.Linear(self.hidden, self.hidden), nn.ReLU(),
            nn.Linear(self.hidden, dims), nn.ReLU())
```

![](/Users/sellenz/Desktop/project/gated_fusion/85.61-3lmlp/85.61.png)

## version 3 -> 86.83:

$y = W_3(W_2(W_1x + bias_1) + bias_2) + bias_3$

$W_1\in(dims*hidden)$,$W_2\in(hidden*hidden)$ ,$W_3\in(hidden*dims)$ ,$hidden=dims*3$

```python
self.hidden = dims * 3
self.layers = nn.Sequential(
            nn.Linear(dims, self.hidden), nn.ReLU(),
  					nn.Linear(self.hidden, self.hidden), nn.ReLU(),
            nn.Linear(self.hidden, dims), nn.ReLU())
```



![](/Users/sellenz/Desktop/project/gated_fusion/86.83-3dims3mlp/86.83.png)

## version 4->86.73 :

$y = W_3(W_2(W_1x + bias_1) + bias_2) + bias_3$

$W_1\in(dims*hidden)$,$W_2\in(hidden*hidden)$ ,$W_3\in(hidden*dims)$ ,$hidden=dims*4$

![](/Users/sellenz/Desktop/project/gated_fusion/86.74-4dims3mlp/86.74.png)

## version5 -> 72.61:

$y = W_3(W_2(W_1x + bias_1) + bias_2) + bias_3$

$W_1\in(dims*hidden)$,$W_2\in(hidden*hidden)$ ,$W_3\in(hidden*dims)$ ,$hidden=dims*5$

## Version6 -> 84.68:

$y = W_4(W_3(W_2(W_1x + bias_1) + bias_2) + bias_3)+bias_4$

$W_1\in(dims*hidden)$,$W_2\in(hidden*hidden)$ ,$W_3\in(hidden*hidden)$ ,$W_4\in(hidden *dims)$,$hidden=dims*3$

```python
self.hidden = dims * 3
self.layers = nn.Sequential(
            nn.Linear(dims, self.hidden), nn.ReLU(),
  					nn.Linear(self.hidden, self.hidden), nn.ReLU(),
  					nn.Linear(self.hidden, self.hidden), nn.ReLU(),
            nn.Linear(self.hidden, dims), nn.ReLU())
```

![](/Users/sellenz/Desktop/project/gated_fusion/84.68-3dims4mlp/84.68.png)

### version7 -> 86.26:

$y = W_5(W_4(W_3(W_2(W_1x + bias_1) + bias_2) + bias_3)+bias_4)+bias_5$

$W_1\in(dims*hidden)$,$W_2\in(hidden*hidden)$ ,$W_3\in(hidden*hidden)$ ,$W_4\in(hidden*hidden$,$W_5\in(hidden *dims)$,$hidden=dims*3$

```python
self.hidden = dims * 3
self.layers = nn.Sequential(
            nn.Linear(dims, self.hidden), nn.ReLU(),
  					nn.Linear(self.hidden, self.hidden), nn.ReLU(),
					  nn.Linear(self.hidden, self.hidden), nn.ReLU(),
  					nn.Linear(self.hidden, self.hidden), nn.ReLU(),
            nn.Linear(self.hidden, dims), nn.ReLU())
```

![](/Users/sellenz/Desktop/project/gated_fusion/86.26-3dims5mlp/86.26.png)

## 08-07v3将速度vel拆分为19关节向量87.38:

[File06](https://github.com/sellenzh/doing_project/blob/main/GCN_module/Gated_Fusion/file06.py)

### version 1 -> 87.38:

```python
self.mlp = nn.Sequential(
            nn.Linear(1, inputs), nn.ReLU(),
            nn.Linear(inputs, 19), nn.ReLU())
```

![](/Users/sellenz/Desktop/project/gated_fusion/87.38-vel_resize/87.38.png)

### version 2 -> 86.095:

[file07](https://github.com/sellenzh/doing_project/blob/main/GCN_module/Gated_Fusion/file07.py)

corss attention 中mlp 3层：

```python
self.mlp = nn.Sequential(
            nn.Linear(1, inputs), nn.ReLU(),
            nn.Linear(inputs, inputs), nn.ReLU(),
            nn.Linear(inputs, 19), nn.ReLU())
```

![](/Users/sellenz/Desktop/project/gated_fusion/86.095-cross3mlp/86.095.png)

### version 3 -> 84.81:

[file08](https://github.com/sellenzh/doing_project/blob/main/GCN_module/Gated_Fusion/file08.py)

Ped_graph 模型中 设置velocity的mlp为3layers，2hidden units。

```python
self.vel1 = nn.Sequential(
                    nn.Linear(2, self.ch1*2), nn.ReLU(),
                    nn.Linear(self.ch1*2, self.ch1*2), nn.ReLU(),
                    nn.Linear(self.ch1*2, self.ch1), nn.ReLU())
```

![](/Users/sellenz/Desktop/project/gated_fusion/84.81-crossmlp2hidden/84.81.png)

### version4->84.39:

Mlp: 3layers, 3hidden units.

```python
self.vel1 = nn.Sequential(
                    nn.Linear(2, self.ch1*3), nn.ReLU(),
                    nn.Linear(self.ch1*3, self.ch1*3), nn.ReLU(),
                    nn.Linear(self.ch1*3, self.ch1), nn.ReLU())
```

![](/Users/sellenz/Desktop/project/gated_fusion/84.39-3mlp3hidden/84.39.png)

### version5 -> 88.05:

mlp : 2layers, 3hidden units.

```python
self.vel1 = nn.Sequential(
                    nn.Linear(2, self.ch1*3), nn.ReLU(),
                    nn.Linear(self.ch1*3, self.ch1), nn.ReLU())
```

![](/Users/sellenz/Desktop/project/gated_fusion/88.05-2mlp3hidden/88.05.png)

### version 6 -> 85.87:

[File09](https://github.com/sellenzh/doing_project/blob/main/GCN_module/Gated_Fusion/file09.py) 

MLP -> Conv1D

```python
self.vel1 = nn.Sequential(
    nn.Conv1d(2, self.ch1, kernel_size=1, stride=1, padding=0, bias=True),
    nn.BatchNorm1d(self.ch1), nn.SiLU())
```

![](/Users/sellenz/Desktop/project/gated_fusion/85.87-conv1d/85.87.png)

### version 7. -->   88.567759:

Conv1d(in, out\*2) ->  Conv1d(out\*2, out)

```python
self.vel1 = nn.Sequential(
    nn.Conv1d(2, self.ch1 * 2, kernel_size=1, stride=1, padding=0, bias=True),
    nn.BatchNorm1d(self.ch1 * 2), nn.SiLU(),
		nn.Conv1d(self.ch1 * 2, self.ch1, kernel_size=1, stride=1, padding=0, bias=True),
		nn.BatchNorm1d(self.ch1), nn.ReLU())
```

![](/Users/sellenz/Desktop/project/gated_fusion/88.567759-2conv/88.567759.png)

### version7-1 :

[file11](https://github.com/sellenzh/doing_project/blob/main/GCN_module/Gated_Fusion/file11.py)

Conv units由2times 更改为 3times

![]()

### version 8 -> 87.41：

Conv1d(in, out\*2) ->  Conv1d(out\*2, out*3) -> Conv1d(out\*3, out)

````python
self.vel1 = nn.Sequential(
		nn.Conv1d(2, self.ch1 * 2, kernel_size=1, stride=1, padding=0, bias=True),
    nn.BatchNorm1d(self.ch1 * 2), nn.SiLU(),
		nn.Conv1d(self.ch1 * 2, self.ch1 * 3, kernel_size=1, stride=1, padding=0, bias=True),
		nn.BatchNorm1d(self.ch1 * 3), nn.ReLU(),
  	nn.Conv1d(self.ch1 * 3, self.ch1, kernel_size=1, stride=1, padding=0, bias=True),
  	nn.BatchNorm1d(self.ch1), nn.ReLU())
````

![](/Users/sellenz/Desktop/project/gated_fusion/87.41-3conv/87.41.png)

## 09--08v7:

### VERSION 1->87.09:

[file10](https://github.com/sellenzh/doing_project/blob/main/GCN_module/Gated_Fusion/file10.py)

`Pose 与 <q,k>·v 经过gated1 fusion 得到gate1`

`gate1 与 velocity 经过gated2 fusion得到y`

`vel2由vel1 经过Linear增加Nodes 使Size与pose匹配`

Conv1d(in, out\*2) ->  Conv1d(out\*2, out)

$y = Gated_2[Gated_1(pose  + Att(q,k,vel_1) ) + vel_2)]$

$vel_2 = MLP(vel_1.unsqueeze(-1))$ ，$units = ch_i * 3$

```python
self.vel1 = nn.Sequential(
    nn.Conv1d(2, self.ch1 * 2, kernel_size=1, stride=1, padding=0, bias=True),
    nn.BatchNorm1d(self.ch1 * 2), nn.ReLU(),
   	nn.Conv1d(self.ch1 * 2, self.ch1, kernel_size=1, stride=1, padding=0, bias=True),
    nn.BatchNorm1d(self.ch1), nn.ReLU())
self.vel_node = nn.Sequential(
    nn.Linear(1, self.ch1*3), nn.ReLU(),
    nn.Linear(self.ch1*3, self.ch1*3), nn.ReLU(),
    nn.Linear(self.ch1*3, 19), nn.ReLU())

pose_att = self.cross1(pose, velocity)
velocity_node = self.vel_node(velocity.unsqueeze(-1))
gate1 = self.gated1(pose, pose_att)
pose = self.gated11(gate1, velocity_node)
```

![](/Users/sellenz/Desktop/project/gated_fusion/87.09-fusion2/87.09.png)

### version 2 -> 78.3558:

$y = Gated_2[Gated_1(pose  + Att(q,k,vel_1) ) + vel_2)]$

$vel_2 = MLP(vel_1.unsqueeze(-1))$ ，$units = ch_i * 2$

![](/Users/sellenz/Desktop/project/gated_fusion/78.3558-fu2-2times/78.3558.png)

### version 3 -> 85.228:

$y = Gated_2[Gated_1(pose  + Att(q,k,vel_1) ) + vel_2)]$

$vel_2 = MLP(vel_1.unsqueeze(-1))$ ，$units = ch_i * 3$

![](/Users/sellenz/Desktop/project/gated_fusion/85.228-fu2-3units/85.228.png)

## Summary:

1. gated fusion 最后去除2layers Conv

2. cross attention module ‘s head 设为 4/8
3. cross attention 中mlp设为3layers ，1->units->19
4. Batch_size设为32 or16
5. gated Fusion ’s fc layer 为MLP 3layers ,$hidden = dims * 3$
6. ped_graph中Velocity 使用mlp拓展维度与nodes维度一致。
7. vel使用Conv1d替代Linear， 2 layers， 2 units   in->out*2->out （3units？？）



