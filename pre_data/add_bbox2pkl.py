import numpy as np
import pickle5 as pk
import os

np.set_printoptions(threshold=np.inf)

def load_xml_coordinates(path, file, pid, fid, i):
    #print(file)
    #vid = file.split('_')[1][:4]
    #print(vid)
    file = path + file
    with open(file, 'rb') as f:
        data = str(f.read())
    for ped in data.split('<track label=')[1:]:# 每个行人数据#"ped(estrian)">...
        p = str(ped.split('name="id">')[1].split('</attribute>')[0])# ...name="id">*x_x_x*</attribute>...
        if p != pid:
            continue
        for fr in ped.split('<box frame="')[1:]: #每帧数据：...<box frame=*xxx*" keyframe=...
            if int(fr.split('" keyframe')[0]) == int(fid) + i:
                if fr.split('xbr="')[1].split('"')[0] is not None and fr.split('xtl="')[1].split('"')[0] is not None and fr.split('ybr="')[1].split('"')[0] is not None and fr.split('ytl="')[1].split('"')[0] is not None:
                    return fr.split('xbr="')[1].split('"')[0], fr.split('xtl="')[1].split('"')[0], fr.split('ybr="')[1].split('"')[0], fr.split('ytl="')[1].split('"')[0]
            else:
                return '0.0', '0.0', '0.0', '0.0'

def readbbox(vid, pid, fid, i):
    file_name = 'video_' + vid + '.xml'
    return load_xml_coordinates(jaad_annotations_path, file_name, pid, fid, i)
    #return xbr, xtl, ybr, ytl
    
def load_write_pkl(path, file):
    #print(path)
    #print(file)
    vid = str(file.split('_')[1])
    #print(vid)
    pid = str(file.split('_')[3]) + '_' + str(file.split('_')[4]) + '_' + str(file.split('_')[5])
    #print(pid)
    fid = str(file.split('_')[-1].split('.')[0])
    #print(fid)
    filename = path + file
    coordinates = []
    for i in range(32):
        '''if i == 29:
            print('ok.')'''
        xbr, xtl, ybr, ytl = readbbox(vid, pid, fid, i)
        coordinates.append([float(xbr), float(xtl), float(ybr), float(ytl)])
    with open(filename, 'rb') as f:
        data = pk.load(f)
        #print(type(data)) # data's type is: dict -> {key: values, ...}
    data['bbox'] = coordinates
    with open(filename, 'wb') as f:
        pk.dump(data, f)
    #print(str(vid) + '_' + str(pid) + '_' + str(fid) + 'Finished!')

jaad_annotations_path = os.getcwd() + '/JAAD/annotations/'
#jaad_annotations_path = os.getcwd() + '/test/xml/'
annot = os.walk(jaad_annotations_path)
kps_path = os.getcwd() + '/data/JAAD/data/'
#kps_path = os.getcwd() + '/test/pkl/'
kps = os.walk(kps_path)
count = 0
#flag = False
for path, dir_list, file_list in kps:
    for file in file_list:
        #print(file) #get file as: video_0001.xml
        #load_xml(path, file)
        #if flag:
        #print(file)
        load_write_pkl(path, file)
        count += 1
        if (count % 100 == 0):
            print(str(count) + ' files finished.')
            '''if count == 11300:
                flag = True'''
print('   ' + str(count) + ' files finished!')
