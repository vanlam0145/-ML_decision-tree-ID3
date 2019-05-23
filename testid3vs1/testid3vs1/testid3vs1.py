
import pandas as pd
from sklearn.model_selection import train_test_split
import math
from collections import deque
class Node(object):
    def __init__(self):
        self.value = None
        self.next = None
        self.childs = None
def printTree(root):
    ##in ra cay
    if root:
        roots = deque()
        roots.append(root)
        while len(roots) > 0:
            root = roots.popleft()
            if root.childs:
                print("node:",root.value)##cac node
                for child in root.childs:
                    print("child node:",'({})'.format(child.value))##cac dieu kien
                    roots.append(child.next)
            if(root.childs==None and root.next==None):
                print("leaf:",root.value)##cac node la
    return True
def id3(df_train,root):##trả về một cây
    index = entropycategorys(df_train)##tra ve cot có entropy nho nhat trong category
    #print("indexx",index)
    root = Node()
    if(len(df_train) == 1):##khi category còn có 1 giá trị duy nhất thì nó sẽ là node leaf
        root.value = df_train.iloc[0,-1]
        #print("add leaf",root.value)
        return root
    if(len(socum(df_train.iloc[:,-1])) == 1):##khi cột cuối cùng của category chỉ có một giá trị duy nhất thì nó sẽ là node
                                                  ##leaf
        root.value = df_train.iloc[0,-1]
        #print("add leaf",root.value)
        return root
    root.value = df_train.columns[index]
    #print("add root",root.value)
    root.childs = []
    ##cot Index
    cIndex=df_train.iloc[:,index]
    socums = socum(cIndex)
    #print(socums)
    for socumin in socums:
        child = Node()
        child.value = socumin
        root.childs.append(child)
        cataa = df_train[cIndex == socumin]
        child.next = id3(cataa,child.next)
    return root
def entropycategorys(df_train):##trả về cột có entropy nhỏ nhất
    entropyr = []
    NumColumns = df_train.shape[1] - 1
    i = 0
    while i < NumColumns:
        entropyr.append(entropycategoryR(df_train,i))
        i+=1
    return entropyr.index(min(entropyr))
def entropycategoryR(df_train,index):##trả về số entropy của một cột
    cIndex=df_train.iloc[:,index]
    socums = socum(cIndex)
    entroys = []
    count = len(df_train)
    for socumin in socums:
        cateIndex=df_train[cIndex == socumin]
        counts=len(cateIndex)
        kq = counts * entropycategoryC(cateIndex) / count
        entroys.append(kq)
    kq = sum(entroys)
    return kq
def entropycategoryC(df_train_index):##trả về số entropy của mảng data thuộc cùng một cụm
    t = len(df_train_index)
    cIndex=df_train_index.iloc[:,-1]
    socums = socum(cIndex)
    values = []
    ans = []
    for scum in socums:
        value = len(df_train_index[cIndex == scum])
        values.append(value)
    for value in values:
        if(value == t):
            kq = 0
            return kq
        else:
            ans.append(-value * (math.log2(value / t)))
    kq = 0
    for an in ans:
        kq+=an / t
    return kq
def socum(df_train_Cindex):##tìm mảng dữ liệu trùng nhau trong một cột category
    array = []
    for cate in df_train_Cindex:
        if(kttontai(array,cate)):##xác định dòng dữ liệu(cate) đã xuất hiện trong mảng hay chưa
            array.append(cate)
        else:
            continue
    return array
def kttontai(array,i):
    for arr in array:
        if(arr == i):
            return 0
        else:
            continue
    return 1
def test(df_test,root):##test new data, sử dụng cây(root)
    sodong = len(df_test)
    labels = [0] * sodong
    for ii in range(sodong):##test trên từng dòng data
        x = df_test.iloc[ii,:]
        #print("data for test",x);
        node = Node()
        node = root
        while node.childs:##nếu node có childs
            for jj in range(len(node.childs)):
                if(node.childs[jj].value == x[node.value]):##kiểm tra theo cây, với dữ liệu x
                    #print("node jum",node.value)
                    node = node.childs[jj].next
                    break
        #print("data add for label",node.value)
        labels[ii] = node.value##
    ##print(labels)##xuất mảng kết quả với thứ tự của newdata
    return labels
def f1_score(y,y_pred):
    sca = socum(y)
    scs = range(len(sca))
    c = range(len(y_pred))
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    #for sc in scs:
    for ii in c:
        if(y.array[ii] == sca[0]):
            if(y.array[ii] == y_pred[ii]):
                tp+=1
            else:
                fn+=1
        else:
            if(y.array[ii] == y_pred[ii]):
                tn+=1
            else:
                fp+=1
    recall=tp/(tp+fn);
    precision=tp/(tp+fp);

    print("TP:",tp," FN:",fn," TN:",tn," FP:",fp)
    f1=(2*recall*precision)/(recall+precision)
    print("F1_SCORE:",f1)
def getData():
    df = pd.DataFrame.from_csv('weather.csv')
    #df=pd.read_csv('heart.csv')
    #print(df)
    X=df.iloc[:,:-1]
    y=df.iloc[:,-1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    df_train=pd.concat([X_train,y_train],axis=1)
    df_test=pd.concat([X_test,y_test],axis=1)
    return df,df_train,df_test
def main():
    df,df_train,df_test=getData()
    ##train và test trên toàn bộ dữ liệu df
    root = Node()
    root = id3(df,root)##học với dữ liệu là df
    y_pred = test(df.iloc[:,:-1],root)##test với dữ liệu là 4 dòng cuối cùng
    print(df.iloc[:,-1].array)
    print(y_pred)
    printTree(root)
    f1_score(df.iloc[:,-1],y_pred)##Tính hiệu xuất
    
    ##train và test trên những tập được lấy ra random từ dữ liệu df
    root=Node()
    root=id3(df_train,root)
    y_pred=test(df_test.iloc[:,:-1],root)
    print(df_test.iloc[:,-1].array)
    print(y_pred)
    printTree(root)
    f1_score(df_test.iloc[:,-1],y_pred)  
main()
