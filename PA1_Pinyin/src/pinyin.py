
import time
from model import *

mode = ''
gram = 3
lamb = 0.99
gamma = 10
k = 3

for cmd in sys.argv:
    if cmd == 'pinyin.py':
        continue
    elif '=' in cmd:
        token = cmd.split('=')
        if len(token)!=2:
            continue
        if token[0] == 'gram':
            gram = int(token[1])
            if gram not in [2,3]:
                print('illegal gram!')
                gram = 3
        elif token[0] == 'k':
            k = int(token[1])
        elif token[0] == 'gamma':
            gamma = int(token[1])
        elif token[0] == 'lamb':
            lamb = float(token[1])

if gram == 3:
    print('model: gram=3','gamma='+str(gamma),'k='+str(k))
else:
    print('model: gram=2','lamb='+str(lamb))

if '.txt' in sys.argv[len(sys.argv)-1]:
    input = sys.argv[len(sys.argv)-2]
    output = sys.argv[len(sys.argv)-1]
    mode = 'file'
elif sys.argv[len(sys.argv)-1] == 'test':
    mode = 'test'
else:
    mode = 'interactive'
print('mode:',mode)

print('Loading model...')
st = time.time()
pinyin_dict, duoyin_dict = load_pinyin()
model = myModel(lamb=lamb, gamma=gamma)
ed = time.time()
print('Model loaded in', ed - st, 'seconds')

        
if mode == 'interactive':
    while True:
        pinyin = input('>')
        if pinyin == 'exit' or pinyin == 'q':
            print('Bye')
            break
        elif len(pinyin) == 0:
            continue
        pinyin = pinyin.lower()
        st = time.time()
        ans = convert(pinyin, model, pinyin_dict,
                      duoyin_dict, gram=gram, k=k)
        ed = time.time()
        if ans:
            print(ans)
            print('time usage:', (ed - st) * 1000, 'ms')
        else:
            print('illegal input!')
elif mode == 'file':
    convert_file(model, pinyin_dict,
    duoyin_dict, gram=gram, k=k, input=input, output=output)
elif mode == 'test':
    test(model, pinyin_dict, duoyin_dict, gram=gram, k=k)
