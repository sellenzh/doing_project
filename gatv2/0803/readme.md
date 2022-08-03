put graph attention module and temporal attention module in one cycle
in version-3(v080303) set:
  batch_size: 2
  head: 8
  cycle times for attention module: 4
  learning_rate: 0.002
  optim: Adam
  auto_find_lr: False
  
  WHEN batch_size=2, the balanced_score is 0.5. means the model is not learning anything in traing steps. 
  the next modify in the model would to set the batch_size to 8 or 16/32/64/128 etc.
  
