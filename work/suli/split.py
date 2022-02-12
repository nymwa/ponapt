import sys

def main():
    data = []
    for x in sys.stdin:
        data.append(x.strip())

    for i in range(10):
        with open('splitdata/data.{}.txt'.format(i), 'w') as f:
            for x in data[100000 * i: 100000 * (i+1)]:
                print(x, file = f)

if __name__ == '__main__':
    main()

