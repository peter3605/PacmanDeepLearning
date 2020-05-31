import os
def grid_search():
    starting_batch_size, starting_tau, starting_epsilon = 28, 0.04, 0.04
    print("GRID SEARCH OF DDQN AGENT")
    for a in range(1,10):
        batch_size = starting_batch_size + 2*a
        for b in range(3, 10):
            tau = round((starting_tau + 0.01 * b), 2)
            for c in range(7, 10):
                epsilon = round((starting_epsilon + 0.01*c), 2)
                print(f"Testing parameters batch_size={batch_size}, epsilon={epsilon}, tau={tau}")
                mode = "train"
                os.system(f'python pacman.py -p PacmanDDQN -n 5000 -x 5000 -l mediumClassic -a mode={mode},batch_size={batch_size},tau={tau},epsilon={epsilon}')
                mode = "test"
                os.system(f'python pacman.py -p PacmanDDQN -n 100 -x 95 -l mediumClassic -a mode={mode},batch_size={batch_size},tau={tau},epsilon={epsilon}')
                print("\n")


if __name__ == '__main__':
  grid_search()