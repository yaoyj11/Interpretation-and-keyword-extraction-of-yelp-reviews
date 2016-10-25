import re
import json
def parseReview(line):
    res=json.loads(line)
    return res["business_id"],(res["stars"],res["text"])

def parseBusiness(line):
    res=json.loads(line)
    return res["business_id"],res["categories"]

def remove_punctuation(line):
    res=re.sub(r"\p{P}"," ",line.lower())
    res=re.sub(r"[^a-zA-Z]"," ",res)
    return res

