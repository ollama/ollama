import h5py
import collections

ids = {}
with open("string_ids.txt") as fi:
    for x in fi:
        p = x.strip().split("|")
        ids[p[0]] = p[1]
#print(ids)
# from https://stackoverflow.com/a/53340677
def descend_obj(obj,sep='\t'):
    """
    Iterate through groups in a HDF5 file and prints the groups and datasets names and datasets attributes
    """
    if type(obj) in [h5py._hl.group.Group,h5py._hl.files.File]:
        #print("FILE")
        for key in obj.keys():
            #print ("KEY",sep,'-',key,':',obj[key])
            descend_obj(obj[key],sep=sep+'\t')
    elif type(obj)==h5py._hl.dataset.Dataset:
        #print("ds")
        #print( obj.name, obj.shape, obj.size, obj.dtype)

        report = collections.Counter()
        objs = obj.size
        for i in range(objs):
            #print("OBJ",i, obj[i])
            data = obj[i]
            name = ids.get(str(data[1]),"oops")
            print("\t".join(map(str,data)),name)
            # 1 [('id', '<i8'), 
            # 2 ('symbol', '<u4'),
            # 3 ('module', '<u4'),
            # 4 ('unresolved', 'u1'),
            # 5 ('originalIP', '<u8'),
            # 6 ('stackDepth', '<i4')]
            #ip = obj[i][4]
            #print("DEB",j,f)
        #    report[ip] += 1
        #else:
        #    print(obj)
        #print(report.most_common(100))
        #if obj.chunks:
            #for c in obj.chunks:
            #    print("CHUNK",c)
        #    print("ATTR",x)
        #for x in obj.fields():
        #    print("FIELD",x)
        #print(dir(obj))
        #print(dir(obj))
        #'_cache_props', '_d', '_dcpl', '_dxpl', '_e', '_extent_type', '_fast_read_ok', '_fast_reader', '_filters', '_id', '_is_empty', '_lapl', '_lcpl', '_readonly', '_selector',
        #'asstr', 'astype', 'attrs', 'chunks', 'compression', 'compression_opts', 'dims', 'dtype', 'external', 'fields', 'file', 'fillvalue', 'fletcher32', 'flush', 'id', 'is_scale', 'is_virtual', 'iter_chunks', 'len', 'make_scale', 'maxshape', 'name', 'nbytes', 'ndim', 'parent', 'read_direct', 'ref', 'refresh', 'regionref', 'resize', 'scaleoffset', 'shape', 'shuffle', 'size', 'virtual_sources', 'write_direct'
        
        for key in obj.attrs.keys():
            print ("DS",sep+'\t','-',key,':',obj.attrs[key])
    else:
        print(obj)

def h5dump(path,group='/'):
    """
    print HDF5 file metadata

    group: you can give a specific group, defaults to the root group
    """
    with h5py.File(path,'r') as f:
         descend_obj(f[group])
if __name__=="__main__":
    h5dump("report13.h5","/CUDA_CALLCHAINS")
