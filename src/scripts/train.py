import os
import sys
os.chdir('/home/jovyan/Co-Speech-Motion-Generation/src')
sys.path.append(os.getcwd())

from trainer import Trainer

def main():
    trainer = Trainer()
    trainer.train()

if __name__ == '__main__':
    main()