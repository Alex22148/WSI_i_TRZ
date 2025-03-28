import json


def getJsonObjFromFile(path):
    jsonObj={}
    try:
        f = open(path, encoding="utf-8")
        jsonObj = json.load(f)
    except:
        print("prawdopodobnie brak pliku")
    return jsonObj

def getJsonObjFromString(text):
    jsonObj={}
    try:
        jsonObj = json.loads(text)
    except:
        print("prawdopodobnie błąd danych jsonString")
    return jsonObj

def getSizeJsonObj(jsonObj):
    return len(jsonObj)

def getFieldNameFromJsonObj(jsonObj):
    field=[]
    for id in jsonObj:
        field.append(id)
    return field

def shortInfoJsonFileStruct_type_data(path):
    jsonObj=getJsonObjFromFile(path)
    fields=getFieldNameFromJsonObj(jsonObj)
    sizeFields=[]
    subFields=[]
    for id in fields:
        x=str(type(jsonObj[id]))
        if x=="<class 'str'>":
            sizeFields.append(jsonObj[id])
        if x=="<class 'list'>":
            q=getSizeJsonObj(jsonObj[id])
            sizeFields.append(q)
            if q>0:
                subFields = getFieldNameFromJsonObj(jsonObj[id][0])

    print('structure: ', fields)
    print('type structure, sizedata: ', sizeFields)
    print('type fielsData: ', subFields)
    return fields, sizeFields, subFields



def getListfromJsonObj(jsonObj,fieldName):
    lista=[]
    try:
        for i in jsonObj[fieldName]:
            lista.append(i)
    except:
        print('prawdopodobnie brak listy')
    return lista

def writeJson2file(jsonObj,path,type=0):
    '''
    Example: JsonObj ={
    "name" : "sathiyajith",
    "rollno" : 56,
    "cgpa" : 8.6,
    "phonenumber" : "9976770500"
}
    :param jsonObj:
    :param path:
    :param type:
    :return:
    '''
    if type==0:
        with open(path, 'w') as f:
            json.dump(jsonObj, f)
    if type==1:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(jsonObj, f, ensure_ascii=False, indent=4)

def writeJsonObj2string(jsonObj,type=0):
    jsonStr = json.dumps(jsonObj)
    return jsonStr