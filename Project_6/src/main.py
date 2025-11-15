from game import Game
import sys
import os
import time

def main():
    # textfile = open("./src/data/expectiminimax/10.csv", "w")
    # textfile.write("dropped, rows\n")
    # for i in range(2):
    #     print(i)
    #     g = Game(sys.argv[1])
    #     dropped, rows = g.run_no_visual()
    #     textfile.write(str(dropped) + ", " + str(rows) + "\n")
    # textfile.close()
    g = Game(sys.argv[1])
    start = time.time() 
    g.run_no_visual()
    # g.run()
    end = time.time() 
    dur = end - start
    print(f'duration: {dur}')
    

    # for a visual, comment out line 16 and uncomment line 17 


if __name__ == "__main__":
    main()
