import math, random
from operator import xor
message = input("Enter the message \n")
string = str(message)
result = [(ord(c)) for c in string]
print("ASCII code...")
print(result)
binary = [bin(value)[2:].zfill(7) for value in result]
print("Binary values of respected ASCII codes...")
print(binary)
merged="".join(binary)
print("Complete Binary value of message...")
print(merged)

def OTPgenerator():
    digits = "01"
    OTP = ""
    rangeval=len(merged)
    for i in range(rangeval):
        OTP += digits[math.floor(random.random() * 2)]

    return OTP

key=OTPgenerator()
print("Key by OTP...")
print(key)

counter1=0
cipherlist=[""]
while counter1<len(merged):
    cipher=xor(int(merged[counter1]), int(key[counter1]))
    cipherlist.append(cipher)
    counter1 +=1
cipherlist2=map(str,cipherlist)
CipherText="".join(cipherlist2)
print("Cipher Text...")
print(CipherText)
print("Encrypton has been completed")

incr2=0
declist=[]
while incr2<len(CipherText):
    dec=xor(int(CipherText[incr2]), int(key[incr2]))
    declist.append(dec)
    incr2 +=1
declist2=map(str,declist)
decText="".join(declist2)
print("Decrepted Binary")
print(decText)
n=7
chunks=[decText[i:i+n] for i in range(0,len(decText),n)]
print(chunks)

decryptascii=[int(z,2) for z in chunks]
print("Decrypted ASCII...")
print(decryptascii)
decrypt=[chr(s) for s in decryptascii]
decryptedmessage = "".join(decrypt)
print("Decrypted message...")
print(decryptedmessage)