import matlab.engine

if __name__ == '__main__':
    eng = matlab.engine.start_matlab()
    tf = eng.isprime(37)
    print(tf)