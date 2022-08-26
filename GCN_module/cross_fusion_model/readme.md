pose --->\
          --(mul)-->pose.mul(vel)         --->\
vel  --->/                                      \
                                                 ----> gated fusion    
pose --->\                                      /
          --(cross_attention)--> pose+vel --->/ 
vel  --->/
